"""Pytest configuration and shared fixtures for wsrf tests."""

import numpy as np
import pytest
from sklearn.datasets import make_classification


@pytest.fixture
def simple_classification_data():
    """Generate simple binary classification dataset for testing."""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    return X, y


@pytest.fixture
def multiclass_classification_data():
    """Generate multiclass classification dataset for testing."""
    X, y = make_classification(
        n_samples=150,
        n_features=15,
        n_informative=10,
        n_redundant=3,
        n_classes=3,
        random_state=42
    )
    return X, y


@pytest.fixture
def train_test_split_data(simple_classification_data):
    """Split simple classification data into train/test sets."""
    X, y = simple_classification_data
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    return X_train, X_test, y_train, y_test
