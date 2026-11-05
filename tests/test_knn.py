import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from umkm_knn.data import load_data
from umkm_knn.knn import KNNClassifier
from umkm_knn.preprocessing import MinMaxScaler


@pytest.fixture
def data():
    X, y = load_data()
    return MinMaxScaler().fit_transform(X), y


@pytest.mark.parametrize("k", [1, 3, 5, 7])
@pytest.mark.parametrize("weighted", [False, True])
def test_sama_dengan_sklearn(data, k, weighted):
    X, y = data
    rng = np.random.default_rng(0)
    X_test = rng.random((50, X.shape[1]))  # titik acak -> hampir tak ada jarak seri
    ours = KNNClassifier(k=k, weighted=weighted).fit(X, y).predict(X_test)
    weights = "distance" if weighted else "uniform"
    ref = KNeighborsClassifier(n_neighbors=k, weights=weights, algorithm="brute")
    ref = ref.fit(X, y).predict(X_test)
    assert (ours == ref).all()


def test_k1_menebak_data_latih_sendiri(data):
    X, y = data
    assert (KNNClassifier(k=1).fit(X, y).predict(X) == y).all()


def test_scaler_rentang_0_1(data):
    X, _ = data
    assert X.min() == 0 and X.max() == 1


def test_k_terlalu_besar(data):
    X, y = data
    with pytest.raises(ValueError):
        KNNClassifier(k=len(X) + 1).fit(X, y)
