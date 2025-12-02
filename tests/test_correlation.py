"""Tests for tree_correlation function."""

import numpy as np
import pytest
from wsrf import WSRFClassifier, tree_correlation


def test_tree_correlation_basic(simple_classification_data):
    """Test basic tree correlation calculation."""
    X, y = simple_classification_data
    
    clf = WSRFClassifier(n_estimators=20, random_state=42)
    clf.fit(X, y)
    
    corr = tree_correlation(clf, X)
    
    assert corr.shape == (20, 20), "Correlation matrix should be T x T"
    assert np.allclose(np.diag(corr), 1.0), "Diagonal should be 1.0 (self-correlation)"
    assert np.allclose(corr, corr.T), "Correlation matrix should be symmetric"


def test_tree_correlation_single_tree(simple_classification_data):
    """Test correlation with single tree."""
    X, y = simple_classification_data
    
    clf = WSRFClassifier(n_estimators=1, random_state=42)
    clf.fit(X, y)
    
    corr = tree_correlation(clf, X)
    
    assert corr.shape == (1, 1), "Should be 1x1 matrix"
    assert corr[0, 0] == 1.0, "Single tree correlation with itself is 1.0"


def test_tree_correlation_range(simple_classification_data):
    """Test that correlation values are in valid range."""
    X, y = simple_classification_data
    
    clf = WSRFClassifier(n_estimators=15, random_state=42)
    clf.fit(X, y)
    
    corr = tree_correlation(clf, X)
    
    assert np.all(corr >= -1.0) and np.all(corr <= 1.0), "Correlations should be in [-1, 1]"


def test_tree_correlation_fitted_model(simple_classification_data):
    """Test that correlation requires fitted model."""
    X, y = simple_classification_data
    
    clf = WSRFClassifier(n_estimators=10, random_state=42)
    
    with pytest.raises(AttributeError):
        tree_correlation(clf, X)
    
    clf.fit(X, y)
    corr = tree_correlation(clf, X)
    assert isinstance(corr, np.ndarray), "Should work after fitting"


def test_tree_correlation_different_data(simple_classification_data):
    """Test correlation on different data subsets."""
    X, y = simple_classification_data
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    clf = WSRFClassifier(n_estimators=15, random_state=42)
    clf.fit(X_train, y_train)
    
    corr_train = tree_correlation(clf, X_train)
    corr_test = tree_correlation(clf, X_test)
    
    assert corr_train.shape == (15, 15), "Train correlation should be T x T"
    assert corr_test.shape == (15, 15), "Test correlation should be T x T"
    
    assert not np.allclose(corr_train, corr_test), "Correlations should differ on different data"


def test_tree_correlation_multiclass(multiclass_classification_data):
    """Test tree correlation on multiclass problem."""
    X, y = multiclass_classification_data
    
    clf = WSRFClassifier(n_estimators=20, random_state=42)
    clf.fit(X, y)
    
    corr = tree_correlation(clf, X)
    
    assert corr.shape == (20, 20), "Should work for multiclass"
    assert np.allclose(np.diag(corr), 1.0), "Diagonal should be 1.0"
    assert np.allclose(corr, corr.T), "Should be symmetric"
