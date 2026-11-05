import numpy as np


class KNNClassifier:
    """K-Nearest Neighbors dari nol (Euclidean distance, majority vote)."""

    def __init__(self, k: int = 5, weighted: bool = False):
        if k < 1:
            raise ValueError("k harus >= 1")
        self.k = k
        self.weighted = weighted

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        if self.k > len(X):
            raise ValueError(f"k ({self.k}) lebih besar dari jumlah data ({len(X)})")
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y)
        self.classes_ = np.unique(self.y_)  # urut alfabet
        return self

    def kneighbors(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Jarak dan indeks k tetangga terdekat untuk tiap baris X."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        dist = np.sqrt(((X[:, None, :] - self.X_[None, :, :]) ** 2).sum(axis=2))
        idx = np.argsort(dist, axis=1, kind="stable")[:, : self.k]
        return np.take_along_axis(dist, idx, axis=1), idx

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Proporsi suara tiap kelas (urutan = self.classes_)."""
        dist, idx = self.kneighbors(X)
        w = 1.0 / (dist + 1e-9) if self.weighted else np.ones_like(dist)
        votes = np.zeros((len(idx), len(self.classes_)))
        for c, cls in enumerate(self.classes_):
            votes[:, c] = (w * (self.y_[idx] == cls)).sum(axis=1)
        return votes / votes.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # argmax ambil kelas pertama kalau seri -> sama dengan scikit-learn
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
