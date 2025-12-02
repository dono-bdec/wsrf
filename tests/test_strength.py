"""Tests for strength function."""

import numpy as np
import pytest
from wsrf import WSRFClassifier, strength


def test_strength_basic(simple_classification_data):
    """Test basic strength calculation."""
    X, y = simple_classification_data
    
    clf = WSRFClassifier(n_estimators=20, random_state=42)
    clf.fit(X, y)
    
    s = strength(clf, X, y)
    
    assert isinstance(s, float), "Strength should be a float"
    assert -1.0 <= s <= 1.0, "Strength should be between -1 and 1"


def test_strength_fitted_model(simple_classification_data):
    """Test that strength requires fitted model."""
    X, y = simple_classification_data
    
    clf = WSRFClassifier(n_estimators=10, random_state=42)
    
    with pytest.raises(AttributeError):
        strength(clf, X, y)
    
    clf.fit(X, y)
    s = strength(clf, X, y)
    assert isinstance(s, float), "Should work after fitting"


def test_strength_train_vs_test(simple_classification_data):
    """Test strength on training vs test data."""
    X, y = simple_classification_data
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    clf = WSRFClassifier(n_estimators=30, random_state=42)
    clf.fit(X_train, y_train)
    
    train_strength = strength(clf, X_train, y_train)
    test_strength = strength(clf, X_test, y_test)
    
    assert isinstance(train_strength, float), "Train strength should be float"
    assert isinstance(test_strength, float), "Test strength should be float"
    
    assert train_strength >= test_strength, "Train strength typically >= test strength"


def test_strength_different_n_estimators(simple_classification_data):
    """Test strength with varying number of estimators."""
    X, y = simple_classification_data
    
    strengths = []
    for n_est in [10, 30, 50]:
        clf = WSRFClassifier(n_estimators=n_est, random_state=42)
        clf.fit(X, y)
        s = strength(clf, X, y)
        strengths.append(s)
        assert isinstance(s, float), f"Strength should be float for {n_est} estimators"
    
    assert all(-1.0 <= s <= 1.0 for s in strengths), "All strengths should be in [-1, 1]"


def test_strength_multiclass(multiclass_classification_data):
    """Test strength calculation on multiclass problem."""
    X, y = multiclass_classification_data
    
    clf = WSRFClassifier(n_estimators=20, random_state=42)
    clf.fit(X, y)
    
    s = strength(clf, X, y)
    
    assert isinstance(s, float), "Strength should be float for multiclass"
    assert -1.0 <= s <= 1.0, "Strength should be in [-1, 1] for multiclass"
