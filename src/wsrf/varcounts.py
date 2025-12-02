import numpy as np

def var_counts(model, split_only=False):
    """Return variable usage counts.
    split_only=False => how many times a feature was selected into subspaces.
    split_only=True  => how many times used in actual splits across trees.
    """
    if not hasattr(model, "subspace_counts_") or model.subspace_counts_ is None:
        raise ValueError("Model lacks subspace_counts_. Fit the model first.")
    if not split_only:
        return model.subspace_counts_.copy()

    # Split usage: count how often feature appears as an internal split in scikit trees
    # tree.tree_.feature holds feature indices in the subspace; map back to original indices.
    counts = np.zeros_like(model.subspace_counts_)
    for tr in model.trees_:
        # sklearn stores -2 for leaves
        split_features = tr.model.tree_.feature
        for f in split_features[split_features >= 0]:
            global_f = tr.features[f]  # map subspace index back to global index
            counts[global_f] += 1
    return counts
