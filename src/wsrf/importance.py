import numpy as np

def importance(model):
    """Return initial RF importances from a fitted WSRFClassifier."""
    if not hasattr(model, "feature_importances_") or model.feature_importances_ is None:
        raise ValueError("Model has no feature_importances_. Did you fit it?")
    return model.feature_importances_
