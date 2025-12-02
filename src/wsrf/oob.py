import numpy as np

def oob_error_rate(model, X, y):
    """Compute global OOB error by aggregating OOB predictions across trees.
    We reconstruct OOB votes by re-checking each tree's bootstrap mask is not stored;
    thus this function requires passing (X, y) from training to approximate OOB.
    """
    # NOTE: For exact OOB, masks must be stored during fit. Here we approximate by
    # inferring OOB per tree is unknown; thus we fallback to per-tree stored error mean.
    if getattr(model, "oob_errors_", None) is None:
        raise ValueError("Model lacks oob_errors_.")
    return float(np.mean(model.oob_errors_))

def oob_confusion_matrix(model, X, y):
    """Placeholder: exact OOB aggregation would require storing masks. """
    raise NotImplementedError("Exact OOB confusion requires storing OOB masks during fit.")
