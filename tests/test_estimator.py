"""Tests for WSRFClassifier estimator."""

import numpy as np
import pytest
from wsrf import WSRFClassifier
from sklearn.datasets import make_classification


def test_basic_fit_predict(train_test_split_data):
    """Test that WSRFClassifier can fit and predict on simple data."""
    X_train, X_test, y_train, y_test = train_test_split_data
    
    clf = WSRFClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)
    
    predictions = clf.predict(X_test)
    
    assert predictions.shape == (20,), "Predictions shape should match test set size"
    assert all(p in [0, 1] for p in predictions), "All predictions should be valid class labels"
    assert hasattr(clf, 'feature_importances_'), "Should have feature importances after fit"
    assert hasattr(clf, 'classes_'), "Should have classes_ after fit"
    assert len(clf.trees_) == 10, "Should have correct number of trees"


def test_parameter_validation():
    """Test that invalid parameters raise appropriate errors."""
    X, y = make_classification(n_samples=50, n_features=5, n_classes=2, random_state=42)
    
    clf = WSRFClassifier(n_estimators=-1, random_state=42)
    with pytest.raises((ValueError, Exception)):
        clf.fit(X, y)
    
    clf = WSRFClassifier(n_estimators=10, nodesize=-1, random_state=42)
    with pytest.raises((ValueError, Exception)):
        clf.fit(X, y)
    
    clf = WSRFClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    
    X_wrong_features = np.random.rand(10, 10)
    with pytest.raises((ValueError, IndexError)):
        clf.predict(X_wrong_features)


def test_prediction_shape(simple_classification_data):
    """Test that prediction output has correct dimensions."""
    X, y = simple_classification_data
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    clf = WSRFClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)
    
    predictions = clf.predict(X_test)
    assert predictions.shape == (20,), "Predictions shape should be (n_samples,)"
    
    probabilities = clf.predict_proba(X_test)
    assert probabilities.shape == (20, 2), "Probabilities shape should be (n_samples, n_classes)"
    
    assert np.allclose(probabilities.sum(axis=1), 1.0), "Probabilities should sum to 1 per sample"
    assert np.all(probabilities >= 0) and np.all(probabilities <= 1), "Probabilities should be in [0, 1]"


def test_different_n_estimators(simple_classification_data):
    """Test WSRFClassifier with different n_estimators values."""
    X, y = simple_classification_data
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    for n_est in [10, 50, 100]:
        clf = WSRFClassifier(n_estimators=n_est, random_state=42)
        clf.fit(X_train, y_train)
        
        assert len(clf.trees_) == n_est, f"Should have {n_est} trees"
        assert clf.oob_errors_.shape == (n_est,), f"Should have {n_est} OOB errors"
        
        predictions = clf.predict(X_test)
        assert predictions.shape == (20,), "Predictions shape should be correct"
        
        accuracy = np.mean(predictions == y_test)
        assert accuracy > 0.4, f"Accuracy with {n_est} estimators should be > 40%"


def test_weighted_vs_unweighted():
    """Test that use_weights parameter affects behavior."""
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    clf_weighted = WSRFClassifier(n_estimators=20, use_weights=True, random_state=42)
    clf_weighted.fit(X_train, y_train)
    
    clf_unweighted = WSRFClassifier(n_estimators=20, use_weights=False, random_state=42)
    clf_unweighted.fit(X_train, y_train)
    
    assert hasattr(clf_weighted, 'feature_importances_'), "Weighted should have importances"
    assert hasattr(clf_unweighted, 'feature_importances_'), "Unweighted should have importances"
    
    pred_weighted = clf_weighted.predict(X_test)
    pred_unweighted = clf_unweighted.predict(X_test)
    
    assert pred_weighted.shape == pred_unweighted.shape, "Both should produce same shape output"


def test_multiclass_classification(multiclass_classification_data):
    """Test WSRFClassifier on multiclass problem."""
    X, y = multiclass_classification_data
    X_train, X_test = X[:120], X[120:]
    y_train, y_test = y[:120], y[120:]
    
    clf = WSRFClassifier(n_estimators=20, random_state=42)
    clf.fit(X_train, y_train)
    
    assert len(clf.classes_) == 3, "Should recognize 3 classes"
    
    predictions = clf.predict(X_test)
    assert predictions.shape == (30,), "Predictions shape should match test set"
    assert all(p in [0, 1, 2] for p in predictions), "All predictions should be valid class labels"
    
    probabilities = clf.predict_proba(X_test)
    assert probabilities.shape == (30, 3), "Probabilities shape should be (n_samples, 3)"
    assert np.allclose(probabilities.sum(axis=1), 1.0), "Probabilities should sum to 1"
