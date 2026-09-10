"""Feature extraction under an explicit memory budget.

An edge agent cannot afford a 100 MB vocabulary.  ``BudgetedTextFeaturizer``
uses the hashing trick so the feature space is fixed at construction time
and the serialised featuriser is *constant size* regardless of corpus --
the single most important property for deploying on a microcontroller-class
device.  ``TabularFeaturizer`` handles sensor readings.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer


class BudgetedTextFeaturizer(BaseEstimator, TransformerMixin):
    """Fixed-memory text features via hashing + TF-IDF reweighting.

    Parameters
    ----------
    n_word_features, n_char_features:
        Size of each hashed space.  Total footprint is bounded by
        ``(n_word_features + n_char_features)`` regardless of how much
        text is seen -- there is no stored vocabulary.
    """

    def __init__(
        self,
        n_word_features: int = 2 ** 16,
        n_char_features: int = 2 ** 16,
        word_ngrams: tuple = (1, 2),
        char_ngrams: tuple = (3, 5),
        use_idf: bool = True,
    ) -> None:
        self.n_word_features = n_word_features
        self.n_char_features = n_char_features
        self.word_ngrams = word_ngrams
        self.char_ngrams = char_ngrams
        self.use_idf = use_idf

    def _build(self) -> None:
        self._w = HashingVectorizer(
            n_features=self.n_word_features, ngram_range=self.word_ngrams,
            alternate_sign=False, norm=None, strip_accents="unicode",
            lowercase=True,
        )
        self._c = HashingVectorizer(
            n_features=self.n_char_features, analyzer="char_wb",
            ngram_range=self.char_ngrams, alternate_sign=False, norm=None,
            lowercase=True,
        )
        self._tw = TfidfTransformer(sublinear_tf=True) if self.use_idf else None
        self._tc = TfidfTransformer(sublinear_tf=True) if self.use_idf else None

    def fit(self, X: Sequence[str], y=None):
        self._build()
        Xw, Xc = self._w.transform(X), self._c.transform(X)
        if self.use_idf:
            self._tw.fit(Xw)
            self._tc.fit(Xc)
        self.n_features_out_ = self.n_word_features + self.n_char_features
        return self

    def transform(self, X: Sequence[str]):
        Xw, Xc = self._w.transform(X), self._c.transform(X)
        if self.use_idf:
            Xw, Xc = self._tw.transform(Xw), self._tc.transform(Xc)
        return sparse.hstack([Xw, Xc]).tocsr()

    def nbytes(self) -> int:
        """Approximate in-memory footprint (IDF vectors only -- no vocab)."""
        n = 0
        for t in (getattr(self, "_tw", None), getattr(self, "_tc", None)):
            if t is not None and hasattr(t, "idf_"):
                n += t.idf_.nbytes
        return n


class TabularFeaturizer(BaseEstimator, TransformerMixin):
    """Ordered, missing-safe extraction of numeric signals from dicts."""

    def __init__(self, keys: Optional[Sequence[str]] = None,
                 fill: float = 0.0, standardize: bool = True) -> None:
        self.keys = list(keys) if keys else None
        self.fill = fill
        self.standardize = standardize

    def fit(self, X: Sequence[Mapping[str, float]], y=None):
        if self.keys is None:
            seen: List[str] = []
            for row in X:
                for k in row:
                    if k not in seen:
                        seen.append(k)
            self.keys_ = sorted(seen)
        else:
            self.keys_ = list(self.keys)
        A = self._raw(X)
        self.mean_ = A.mean(axis=0) if self.standardize else np.zeros(A.shape[1])
        self.scale_ = A.std(axis=0) if self.standardize else np.ones(A.shape[1])
        self.scale_[self.scale_ == 0] = 1.0
        self.n_features_out_ = len(self.keys_)
        return self

    def _raw(self, X: Sequence[Mapping[str, float]]) -> np.ndarray:
        return np.array(
            [[float(r.get(k, self.fill)) for k in self.keys_] for r in X],
            dtype=np.float32,
        )

    def transform(self, X: Sequence[Mapping[str, float]]) -> np.ndarray:
        return ((self._raw(X) - self.mean_) / self.scale_).astype(np.float32)


class FusedFeaturizer(BaseEstimator, TransformerMixin):
    """Concatenates tabular and text views of an :class:`Observation`."""

    def __init__(self, text: Optional[BudgetedTextFeaturizer] = None,
                 tabular: Optional[TabularFeaturizer] = None) -> None:
        self.text = text
        self.tabular = tabular

    def fit(self, obs: Sequence, y=None):
        if self.text is not None:
            self.text.fit([o.text for o in obs])
        if self.tabular is not None:
            self.tabular.fit([o.features for o in obs])
        return self

    def transform(self, obs: Sequence):
        blocks = []
        if self.tabular is not None:
            blocks.append(sparse.csr_matrix(self.tabular.transform([o.features for o in obs])))
        if self.text is not None:
            blocks.append(self.text.transform([o.text for o in obs]))
        if not blocks:
            raise ValueError("FusedFeaturizer needs at least one view")
        return sparse.hstack(blocks).tocsr() if len(blocks) > 1 else blocks[0]
