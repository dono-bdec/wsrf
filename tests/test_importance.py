"""Tests for importance function."""

import numpy as np
import pytest
from wsrf import WSRFClassifier, importance


def test_importance_fitted_model(simple_classification_data):
    """Test that importance returns feature importances from fitted model."""
    X, y = simple_classification_data
    
    clf = WSRFClassifier(n_estimators=20, random_state=42)
    clf.fit(X, y)
    
    importances = importance(clf)
    
    assert isinstance(importances, np.ndarray), "Should return numpy array"
    assert len(importances) == X.shape[1], "Should have importance for each feature"
    assert np.allclose(importances.sum(), 1.0), "Importances should sum to 1.0"
    assert np.all(importances >= 0), "All importances should be non-negative"


def test_importance_unfitted_model(simple_classification_data):
    """Test that importance raises error on unfitted model."""
    clf = WSRFClassifier(n_estimators=10, random_state=42)
    
    with pytest.raises(ValueError, match="Model has no feature_importances_"):
        importance(clf)


def test_importance_values_consistency(simple_classification_data):
    """Test that importance returns consistent values with model attribute."""
    X, y = simple_classification_data
    
    clf = WSRFClassifier(n_estimators=20, random_state=42)
    clf.fit(X, y)
    
    importances = importance(clf)
    
    assert np.array_equal(importances, clf.feature_importances_), "Should match model attribute"


def test_importance_different_datasets():
    """Test importance on different dataset sizes."""
    for n_features in [5, 10, 20]:
        X, y = np.random.rand(100, n_features), np.random.randint(0, 2, 100)
        
        clf = WSRFClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)
        
        importances = importance(clf)
        
        assert len(importances) == n_features, f"Should have {n_features} importances"
        assert np.allclose(importances.sum(), 1.0), "Importances should sum to 1.0"
