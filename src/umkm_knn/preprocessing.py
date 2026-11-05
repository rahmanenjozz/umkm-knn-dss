import numpy as np


class MinMaxScaler:
    """Skala tiap fitur ke rentang 0–1 berdasarkan data latih."""

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        self.min_ = X.min(axis=0)
        rng = X.max(axis=0) - self.min_
        self.range_ = np.where(rng == 0, 1.0, rng)  # hindari bagi nol
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.min_) / self.range_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
