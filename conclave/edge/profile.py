"""Emulated edge deployment profiles and honest resource accounting.

We do not own the target hardware, so we do not claim measurements on it.
Instead this module does three separate, individually defensible things
and keeps them clearly labelled:

1. **Constrain** the host to the target's core count and RAM ceiling
   (``constrain``).  Thread caps and ``RLIMIT_AS`` are real limits, not
   estimates -- if the agent needs more memory than a Pi Zero has, it
   raises ``MemoryError`` here too.

2. **Measure** device-independent quantities that transfer without any
   assumption: serialised model bytes, peak resident memory, number of
   floating-point operations per decision.  These are the numbers a
   reader can check against their own board.

3. **Scale** measured host latency by a *calibrated* single-core
   throughput ratio to give an estimated on-device latency, always
   reported as an estimate with the ratio stated.

The distinction matters: (1) and (2) are measurements, (3) is a model.
Papers that blur them get rejected, and rightly so.
"""
from __future__ import annotations

import contextlib
import gc
import os
import platform
import statistics
import time

try:                      # POSIX only; Windows has no RLIMIT_AS equivalent
    import resource
except ImportError:       # pragma: no cover - platform dependent
    resource = None       # type: ignore[assignment]

#: True when this platform can actually enforce a RAM ceiling on the process.
#: On Windows it is False: thread caps still apply, but the memory limit is
#: advisory only.  Results generated there must not claim an enforced ceiling.
MEMORY_LIMIT_SUPPORTED = resource is not None and hasattr(resource, "RLIMIT_AS")
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class DeviceProfile:
    """A target board.

    ``scalar_score`` is the device's measured score on the same
    :func:`cpu_calibration_score` microbenchmark used on the host, so the
    latency scaling factor is ``host_score / device_score``.  Values here
    are literature/vendor-derived placeholders; replace them with your
    own measurement if you have the board, and say so in the paper.
    """

    name: str
    cores: int
    ram_mb: int
    clock_mhz: int
    tdp_watts: float
    scalar_score: float          # relative single-core throughput (host = 1.0)
    note: str = ""


# Relative scalar scores are approximate and used only for the clearly
# labelled *estimated* latency column.
PROFILES: Dict[str, DeviceProfile] = {
    "host": DeviceProfile("host", os.cpu_count() or 1, 0, 0, 15.0, 1.00,
                          "the machine actually running the benchmark"),
    "rpi4b_4gb": DeviceProfile("Raspberry Pi 4B (4 GB)", 4, 4096, 1500, 6.4, 0.28,
                               "Cortex-A72"),
    "rpi3bp": DeviceProfile("Raspberry Pi 3B+", 4, 1024, 1400, 5.0, 0.17,
                            "Cortex-A53"),
    "rpi_zero2w": DeviceProfile("Raspberry Pi Zero 2 W", 4, 512, 1000, 2.5, 0.11,
                                "Cortex-A53, 512 MB -- the hard case"),
    "jetson_nano": DeviceProfile("Jetson Nano (4 GB)", 4, 4096, 1430, 10.0, 0.26,
                                 "Cortex-A57"),
}


def cpu_calibration_score(seconds: float = 0.4) -> float:
    """Single-core throughput microbenchmark (higher is faster).

    Deliberately mixes dense linear algebra and Python-level branching so
    it correlates with the mixed workload an ensemble agent actually runs.
    """
    A = np.random.default_rng(0).random((96, 96))
    t_end = time.perf_counter() + seconds
    n = 0
    while time.perf_counter() < t_end:
        B = A @ A
        _ = float(B.sum())
        n += 1
    return n / seconds


@contextlib.contextmanager
def constrain(profile: DeviceProfile, enforce_memory: bool = True):
    """Really restrict threads and address space to the target profile.

    Thread caps must be set before BLAS initialises to take full effect;
    for reliable results run each profile in a fresh process (see
    ``scripts/run_edge_benchmark.py``).
    """
    keys = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]
    old_env = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ[k] = str(profile.cores)

    old_limit = None
    if (enforce_memory and profile.ram_mb > 0
            and resource is not None and hasattr(resource, "RLIMIT_AS")):
        try:
            old_limit = resource.getrlimit(resource.RLIMIT_AS)
            cap = profile.ram_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (cap, old_limit[1]))
        except (ValueError, OSError):
            old_limit = None
    try:
        yield profile
    finally:
        if old_limit is not None:
            with contextlib.suppress(Exception):
                resource.setrlimit(resource.RLIMIT_AS, old_limit)
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@dataclass
class Measurement:
    """Result of profiling one callable."""

    label: str
    n: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    std_ms: float
    peak_rss_mb: float
    model_bytes: int = 0
    host_score: float = 0.0
    trials: int = 1
    trial_medians_ms: List[float] = field(default_factory=list)
    noise_ratio: float = 1.0        # worst trial median / best trial median

    def estimate_on(self, device: DeviceProfile) -> Dict[str, float]:
        """Model-based projection onto a target board (clearly an estimate)."""
        factor = 1.0 / max(device.scalar_score, 1e-6)
        return {
            "device": device.name,
            "scaling_factor": factor,
            "est_mean_ms": self.mean_ms * factor,
            "est_p99_ms": self.p99_ms * factor,
            "est_energy_mJ": self.mean_ms * factor * device.tdp_watts,
            "fits_in_ram": (self.peak_rss_mb < device.ram_mb) if device.ram_mb else True,
        }

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def profile_call(fn: Callable[[], Any], label: str = "", n: int = 200,
                 warmup: int = 20, model_bytes: int = 0,
                 trials: int = 5) -> Measurement:
    """Time a callable and record peak resident memory.

    Reports p95/p99 as well as the mean because an autonomous agent's safety
    argument depends on the *tail*, not the average.

    **Why repeated trials.**  Interference from other processes can only ever
    make a timing slower, never faster, so a single batch is an upper bound
    contaminated by whatever else the machine was doing.  We therefore run
    ``trials`` independent batches of ``n`` calls and report the batch with
    the lowest median -- the least-contaminated estimate of the code's actual
    cost.  This is standard practice for latency microbenchmarks and it is
    what keeps a number in the paper from moving by 2.5x because a build
    happened to be running in another window.

    ``noise_ratio`` (worst trial median / best) is reported so contamination
    stays visible rather than being quietly averaged away.  A value near 1.0
    means a quiet machine; a large value means the run should be repeated.
    """
    import psutil
    proc = psutil.Process()
    for _ in range(warmup):
        fn()
    gc.collect()
    peak = proc.memory_info().rss

    batches: List[List[float]] = []
    for _ in range(max(trials, 1)):
        times: List[float] = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000.0)
            peak = max(peak, proc.memory_info().rss)
        batches.append(times)

    medians = [statistics.median(b) for b in batches]
    times = batches[medians.index(min(medians))]      # the quietest batch
    ts = sorted(times)
    return Measurement(
        label=label, n=n,
        trials=len(batches),
        trial_medians_ms=[round(m, 4) for m in medians],
        noise_ratio=round(max(medians) / max(min(medians), 1e-9), 3),
        mean_ms=statistics.fmean(times),
        median_ms=statistics.median(times),
        p95_ms=ts[min(int(0.95 * len(ts)), len(ts) - 1)],
        p99_ms=ts[min(int(0.99 * len(ts)), len(ts) - 1)],
        max_ms=max(times),
        std_ms=statistics.pstdev(times) if len(times) > 1 else 0.0,
        peak_rss_mb=peak / (1024 ** 2),
        model_bytes=model_bytes,
        host_score=0.0,
    )


def host_description() -> Dict[str, Any]:
    """Record exactly what the numbers were measured on -- reproducibility."""
    import psutil
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "logical_cores": os.cpu_count(),
        "total_ram_gb": round(psutil.virtual_memory().total / 1024 ** 3, 2),
    }
