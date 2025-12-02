import numpy as np

def tree_correlation(model, X):
    """Compute correlation matrix of tree predictions (labels) across the ensemble.
    Returns (T x T) correlation of 0/1 match indicators per tree relative to first tree.
    """
    # Check if model is fitted
    if not hasattr(model, '_check_is_fitted'):
        # Fallback check if model doesn't have the method
        if not hasattr(model, 'trees_') or len(model.trees_) == 0:
            raise AttributeError("Model is not fitted yet.")
    else:
        model._check_is_fitted()
    
    X = np.asarray(X)
    T = len(model.trees_)
    if T < 2:
        return np.ones((T, T))
    # Collect predictions per tree (as integers mapped to [0..C-1])
    classes = model.classes_
    preds = []
    for tr in model.trees_:
        p = tr.model.predict(X[:, tr.features])
        # map to indices
        idx = np.array([np.where(classes == c)[0][0] for c in p])
        preds.append(idx)
    P = np.vstack(preds)  # T x n_samples
    # Correlation of tree-wise predictions
    corr = np.corrcoef(P)
    return corr
