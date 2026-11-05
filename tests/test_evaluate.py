import numpy as np
import pytest
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler as SkMinMaxScaler

from umkm_knn.data import load_data
from umkm_knn.evaluate import (
    confusion_matrix,
    cross_validate,
    metrics,
    pilih_k,
    stratified_kfold,
)


@pytest.fixture
def data():
    return load_data()


def test_fold_tidak_tumpang_tindih_dan_lengkap(data):
    _, y = data
    folds = stratified_kfold(y, n_splits=5, seed=0)
    semua_test = np.concatenate([test for _, test in folds])
    assert sorted(semua_test) == list(range(len(y)))
    for train, test in folds:
        assert not set(train) & set(test)


def test_fold_stratified(data):
    _, y = data
    for _, test in stratified_kfold(y, n_splits=5, seed=0):
        for cls in np.unique(y):
            # tiap kelas tersebar rata: selisih antar fold maksimal 1
            assert abs((y[test] == cls).sum() - (y == cls).sum() / 5) < 1


def test_confusion_matrix_dan_metrics():
    classes = np.array(["A", "B"])
    y_true = np.array(["A", "A", "B", "B"])
    y_pred = np.array(["A", "B", "B", "B"])
    cm = confusion_matrix(y_true, y_pred, classes)
    assert cm.tolist() == [[1, 1], [0, 2]]
    m = metrics(cm, classes)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["per_kelas"]["B"]["precision"] == pytest.approx(2 / 3)
    assert m["per_kelas"]["A"]["recall"] == pytest.approx(0.5)


@pytest.mark.parametrize("k", [1, 3, 5])
def test_akurasi_sama_dengan_sklearn(data, k):
    X, y = data
    folds = stratified_kfold(y, n_splits=5, seed=0)
    ours = cross_validate(X, y, k=k, seeds=range(1))["accuracy_mean"]

    pipe = make_pipeline(SkMinMaxScaler(), KNeighborsClassifier(n_neighbors=k, algorithm="brute"))
    scores = cross_val_score(pipe, X, y, cv=folds)
    # bobot per fold: jumlah data uji per fold bisa beda 1
    sizes = [len(test) for _, test in folds]
    assert ours == pytest.approx(np.average(scores, weights=sizes))


def test_pilih_k(data):
    X, y = data
    report = pilih_k(X, y, ks=[1, 3], seeds=range(2))
    assert len(report["hasil"]) == 4
    assert report["terbaik"]["k"] in (1, 3)
