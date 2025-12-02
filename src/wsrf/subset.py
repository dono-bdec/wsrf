from copy import deepcopy

def subset_forest(model, indices):
    """Return a shallow-copied model containing only trees at `indices`.
    Renormalizes weights to sum to 1.
    """
    new = deepcopy(model)
    new.trees_ = [model.trees_[i] for i in indices]
    total = sum(tr.weight for tr in new.trees_)
    if total == 0:
        w = 1.0 / len(new.trees_)
        for tr in new.trees_: tr.weight = w
    else:
        for tr in new.trees_: tr.weight /= total
    return new
