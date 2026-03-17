import pytest
import numpy as np
from sklearn.datasets import make_blobs
from ocksvm.model import OCKSVM
from sklearn.utils.estimator_checks import check_estimator

def test_sklearn_compatibility():
    model = OCKSVM(random_state=42)
    check_estimator(model)

def test_ocksvm_fit_predict():
    """
    Test if OCKSVM can learn a simple 3-class separation.
    """
    # Synthetic data
    X, y = make_blobs(n_samples=150, centers=3, n_features=4, random_state=42)
    

    model = OCKSVM(n_clusters=3, oc_nu=0.1)
    model.fit(X, y)
    
    y_pred = model.predict(X)
    
    assert y_pred.shape == y.shape
    accuracy = np.mean(y_pred == y)
    assert accuracy > 0.8, f"Accuracy was too low: {accuracy}"
    
def test_ocksvm_single_class_cluster():
    """
    Test how the model handles a cluster that only contains one class.
    """
    # Dataset with 1 class
    X = np.array([[1, 1], [1.1, 1.1], [5, 5], [5.1, 5.1]])
    y = np.array([0, 0, 1, 1])
    
    model = OCKSVM(n_clusters=2)
    model.fit(X, y)
    
    preds = model.predict(np.array([[0.9, 0.9], [5.2, 5.2]]))
    np.testing.assert_array_equal(preds, [0, 1])

def test_ocksvm_unfitted_error():
    """
    Ensure predict() raises an error if fit() hasn't been called.
    """
    from sklearn.exceptions import NotFittedError
    model = OCKSVM()
    X = np.array([[1, 2], [3, 4]])
    
    with pytest.raises(NotFittedError):
        model.predict(X)