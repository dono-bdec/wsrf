"""Tests for subset_forest function."""

import numpy as np
import pytest
from wsrf import WSRFClassifier, subset_forest


def test_subset_forest_basic(simple_classification_data):
    """Test creating a subset of trees from a forest."""
    X, y = simple_classification_data
    
    clf = WSRFClassifier(n_estimators=20, random_state=42)
    clf.fit(X, y)
    
    indices = [0, 5, 10, 15]
    subset = subset_forest(clf, indices)
    
    assert len(subset.trees_) == 4, "Should have 4 trees in subset"
    
    total_weight = sum(tr.weight for tr in subset.trees_)
    assert np.isclose(total_weight, 1.0), "Weights should sum to 1.0"


def test_subset_forest_single_tree(simple_classification_data):
    """Test creating a subset with single tree."""
    X, y = simple_classification_data
    
    clf = WSRFClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    
    subset = subset_forest(clf, [3])
    
    assert len(subset.trees_) == 1, "Should have 1 tree"
    assert subset.trees_[0].weight == 1.0, "Single tree should have weight 1.0"


def test_subset_forest_all_trees(simple_classification_data):
    """Test creating subset with all trees (should be equivalent to full forest)."""
    X, y = simple_classification_data
    
    clf = WSRFClassifier(n_estimators=15, random_state=42)
    clf.fit(X, y)
    
    indices = list(range(15))
    subset = subset_forest(clf, indices)
    
    assert len(subset.trees_) == 15, "Should have all 15 trees"
    
    total_weight = sum(tr.weight for tr in subset.trees_)
    assert np.isclose(total_weight, 1.0), "Weights should sum to 1.0"


def test_subset_forest_predictions(simple_classification_data):
    """Test that subset forest can make predictions."""
    X, y = simple_classification_data
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    clf = WSRFClassifier(n_estimators=20, random_state=42)
    clf.fit(X_train, y_train)
    
    indices = [0, 2, 4, 6, 8, 10]
    subset = subset_forest(clf, indices)
    
    predictions = subset.predict(X_test)
    assert predictions.shape == (20,), "Should make predictions"
    assert all(p in subset.classes_ for p in predictions), "All predictions should be valid"
    
    probabilities = subset.predict_proba(X_test)
    assert probabilities.shape == (20, len(subset.classes_)), "Should output probabilities"
    assert np.allclose(probabilities.sum(axis=1), 1.0), "Probabilities should sum to 1"


def test_subset_forest_invalid_indices(simple_classification_data):
    """Test that invalid indices raise appropriate errors."""
    X, y = simple_classification_data
    
    clf = WSRFClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    
    with pytest.raises(IndexError):
        subset_forest(clf, [20])
    
    with pytest.raises(IndexError):
        subset_forest(clf, [-20])
