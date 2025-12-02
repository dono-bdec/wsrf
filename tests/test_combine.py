"""Tests for combine_forests function."""

import numpy as np
import pytest
from wsrf import WSRFClassifier, combine_forests


def test_combine_two_forests(simple_classification_data):
    """Test combining two fitted forests."""
    X, y = simple_classification_data
    
    clf1 = WSRFClassifier(n_estimators=10, random_state=42)
    clf1.fit(X, y)
    
    clf2 = WSRFClassifier(n_estimators=15, random_state=43)
    clf2.fit(X, y)
    
    combined = combine_forests(clf1, clf2)
    
    assert len(combined.trees_) == 25, "Should have 10 + 15 = 25 trees"
    
    total_weight = sum(tr.weight for tr in combined.trees_)
    assert np.isclose(total_weight, 1.0), "Weights should sum to 1.0"
    
    assert hasattr(combined, 'classes_'), "Combined model should have classes_"
    assert hasattr(combined, 'feature_importances_'), "Combined model should have importances"


def test_combine_multiple_forests(simple_classification_data):
    """Test combining more than two forests."""
    X, y = simple_classification_data
    
    forests = []
    for i in range(5):
        clf = WSRFClassifier(n_estimators=8, random_state=42+i)
        clf.fit(X, y)
        forests.append(clf)
    
    combined = combine_forests(*forests)
    
    assert len(combined.trees_) == 40, "Should have 5 * 8 = 40 trees"
    
    total_weight = sum(tr.weight for tr in combined.trees_)
    assert np.isclose(total_weight, 1.0), "Weights should sum to 1.0"


def test_combine_single_forest(simple_classification_data):
    """Test combining a single forest (should return copy with normalized weights)."""
    X, y = simple_classification_data
    
    clf = WSRFClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    
    combined = combine_forests(clf)
    
    assert len(combined.trees_) == 10, "Should have same number of trees"
    
    total_weight = sum(tr.weight for tr in combined.trees_)
    assert np.isclose(total_weight, 1.0), "Weights should sum to 1.0"


def test_combine_forests_empty_input():
    """Test that combine_forests raises error with no arguments."""
    with pytest.raises(ValueError, match="Provide at least one forest"):
        combine_forests()


def test_combined_forest_predictions(simple_classification_data):
    """Test that combined forest can make predictions."""
    X, y = simple_classification_data
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    clf1 = WSRFClassifier(n_estimators=10, random_state=42)
    clf1.fit(X_train, y_train)
    
    clf2 = WSRFClassifier(n_estimators=10, random_state=43)
    clf2.fit(X_train, y_train)
    
    combined = combine_forests(clf1, clf2)
    
    predictions = combined.predict(X_test)
    assert predictions.shape == (20,), "Should make predictions on test set"
    assert all(p in combined.classes_ for p in predictions), "All predictions should be valid"
    
    probabilities = combined.predict_proba(X_test)
    assert probabilities.shape == (20, len(combined.classes_)), "Should output probabilities"
    assert np.allclose(probabilities.sum(axis=1), 1.0), "Probabilities should sum to 1"
