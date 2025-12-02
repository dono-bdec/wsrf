"""Tests for OOB (Out-of-Bag) functions."""

import numpy as np
import pytest
from wsrf import WSRFClassifier, oob_error_rate, oob_confusion_matrix


def test_oob_error_rate(simple_classification_data):
    """Test OOB error rate calculation."""
    X, y = simple_classification_data
    
    clf = WSRFClassifier(n_estimators=20, random_state=42)
    clf.fit(X, y)
    
    error_rate = oob_error_rate(clf, X, y)
    
    assert isinstance(error_rate, float), "OOB error rate should be a float"
    assert 0.0 <= error_rate <= 1.0, "OOB error rate should be between 0 and 1"
    
    assert hasattr(clf, 'oob_errors_'), "Model should have oob_errors_ attribute"
    assert len(clf.oob_errors_) == 20, "Should have OOB error for each tree"
    
    expected_mean = np.mean(clf.oob_errors_)
    assert np.isclose(error_rate, expected_mean), "Should return mean of per-tree OOB errors"


def test_oob_error_rate_unfitted_model():
    """Test that oob_error_rate raises error on unfitted model."""
    clf = WSRFClassifier(n_estimators=10, random_state=42)
    
    X = np.random.rand(50, 5)
    y = np.random.randint(0, 2, 50)
    
    with pytest.raises(ValueError, match="Model lacks oob_errors_"):
        oob_error_rate(clf, X, y)


def test_oob_confusion_matrix(simple_classification_data):
    """Test that OOB confusion matrix raises NotImplementedError."""
    X, y = simple_classification_data
    
    clf = WSRFClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    
    with pytest.raises(NotImplementedError, match="Exact OOB confusion requires storing OOB masks"):
        oob_confusion_matrix(clf, X, y)


def test_oob_error_rate_different_estimators(simple_classification_data):
    """Test OOB error rate with different numbers of estimators."""
    X, y = simple_classification_data
    
    for n_est in [10, 30, 50]:
        clf = WSRFClassifier(n_estimators=n_est, random_state=42)
        clf.fit(X, y)
        
        error_rate = oob_error_rate(clf, X, y)
        
        assert isinstance(error_rate, float), f"Error rate should be float for {n_est} estimators"
        assert 0.0 <= error_rate <= 1.0, f"Error rate should be in [0,1] for {n_est} estimators"
