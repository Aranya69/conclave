---
name: Hardware measurement
about: You ran it on a real board - this is the most useful report you can file
labels: hardware, help wanted
---

The paper's on-device latency figures are *scaled estimates*, not measurements,
because no physical board was used. Real numbers are the single most valuable
contribution to this project.

**Board**
- Model and revision (e.g. Raspberry Pi 4B rev 1.5, 4 GB):
- OS and bitness (`uname -m` — `aarch64` or `armv7l`):
- Cooling (passive / heatsink / fan):
- Governor (`cpufreq-info | grep governor`):

**Output of `python deploy/pi_check.py`**
```
```

**Temperature during the run** (`vcgencmd measure_temp`)
```
```

**Anything that failed to install**
