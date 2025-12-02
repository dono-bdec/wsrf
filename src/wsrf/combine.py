from copy import deepcopy

def combine_forests(*forests):
    """Concatenate trees from multiple fitted WSRFClassifier models.
    Assumes identical feature spaces and class sets; renormalizes weights.
    """
    if len(forests) == 0:
        raise ValueError("Provide at least one forest")
    base = deepcopy(forests[0])
    base.trees_.clear()
    for f in forests:
        base.trees_.extend(f.trees_)
    total = sum(tr.weight for tr in base.trees_)
    if total == 0:
        w = 1.0 / len(base.trees_)
        for tr in base.trees_: tr.weight = w
    else:
        for tr in base.trees_: tr.weight /= total
    return base
