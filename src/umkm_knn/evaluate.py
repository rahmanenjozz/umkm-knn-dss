"""Evaluasi KNN: stratified k-fold, confusion matrix, dan pemilihan k.

Jalankan: python -m umkm_knn.evaluate
"""

import json
from pathlib import Path

import numpy as np

from umkm_knn.data import DATASETS, load_data
from umkm_knn.knn import KNNClassifier
from umkm_knn.preprocessing import MinMaxScaler

REPORT_DIR = Path(__file__).resolve().parents[2] / "reports"


def stratified_kfold(
    y: np.ndarray, n_splits: int = 5, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Bagi indeks jadi n_splits fold dengan proporsi kelas yang seimbang.

    Mengembalikan list (train_idx, test_idx).
    """
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(y), dtype=int)
    for cls in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == cls))
        fold_of[idx] = np.arange(len(idx)) % n_splits  # bagikan bergiliran
    return [(np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f)) for f in range(n_splits)]


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Baris = kelas sebenarnya, kolom = kelas tebakan."""
    pos = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[pos[t], pos[p]] += 1
    return cm


def metrics(cm: np.ndarray, classes: np.ndarray) -> dict:
    """Accuracy + precision/recall/F1 per kelas dan macro-F1."""
    tp = np.diag(cm).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.nan_to_num(tp / cm.sum(axis=0))
        recall = np.nan_to_num(tp / cm.sum(axis=1))
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall))
    return {
        "accuracy": float(tp.sum() / cm.sum()),
        "macro_f1": float(f1.mean()),
        "per_kelas": {
            str(c): {"precision": float(p), "recall": float(r), "f1": float(f)}
            for c, p, r, f in zip(classes, precision, recall, f1)
        },
    }


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    weighted: bool = False,
    n_splits: int = 5,
    seeds: range = range(10),
) -> dict:
    """Repeated stratified k-fold. Scaler di-fit di dalam tiap fold (tanpa data leakage)."""
    classes = np.unique(y)
    accs, f1s = [], []
    cm_total = np.zeros((len(classes), len(classes)), dtype=int)
    for seed in seeds:
        y_pred = np.empty_like(y)
        for train, test in stratified_kfold(y, n_splits, seed):
            scaler = MinMaxScaler().fit(X[train])
            model = KNNClassifier(k=k, weighted=weighted).fit(scaler.transform(X[train]), y[train])
            y_pred[test] = model.predict(scaler.transform(X[test]))
        cm = confusion_matrix(y, y_pred, classes)
        m = metrics(cm, classes)
        accs.append(m["accuracy"])
        f1s.append(m["macro_f1"])
        cm_total += cm
    return {
        "k": k,
        "weighted": weighted,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "classes": [str(c) for c in classes],
        "confusion_matrix": cm_total.tolist(),  # dijumlah dari semua pengulangan
    }


def pilih_k(
    X: np.ndarray, y: np.ndarray, ks=range(1, 16, 2), n_splits: int = 5, seeds: range = range(10)
) -> dict:
    """Coba tiap k (uniform & weighted), pilih yang macro-F1 rata-ratanya tertinggi."""
    hasil = [
        cross_validate(X, y, k, weighted, n_splits, seeds)
        for weighted in (False, True)
        for k in ks
        if k <= len(X) - len(X) // n_splits  # k tidak boleh > jumlah data latih per fold
    ]
    terbaik = max(hasil, key=lambda r: (r["macro_f1_mean"], -r["k"]))  # seri -> k kecil
    return {
        "n_data": len(y),
        "n_splits": n_splits,
        "n_repeats": len(seeds),
        "terbaik": {"k": terbaik["k"], "weighted": terbaik["weighted"]},
        "hasil": hasil,
    }


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    for nama in DATASETS:
        X, y = load_data(nama)
        report = pilih_k(X, y)
        report["dataset"] = nama

        print(f"\n=== Dataset: {nama} ({len(y)} baris) ===")
        print(f"{'k':>3} {'bobot':>8} {'accuracy':>16} {'macro-F1':>16}")
        for r in report["hasil"]:
            bobot = "jarak" if r["weighted"] else "sama"
            print(
                f"{r['k']:>3} {bobot:>8} "
                f"{r['accuracy_mean']:>9.3f} ± {r['accuracy_std']:.3f} "
                f"{r['macro_f1_mean']:>9.3f} ± {r['macro_f1_std']:.3f}"
            )
        t = report["terbaik"]
        print(f"Terbaik: k={t['k']}, weighted={t['weighted']}")

        path = REPORT_DIR / f"evaluasi_{nama}.json"
        path.write_text(json.dumps(report, indent=2))
        print(f"Disimpan ke reports/{path.name}")


if __name__ == "__main__":
    main()
