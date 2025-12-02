import numpy as np

def strength(model, X, y):
    """Breiman-style strength: mean margin across samples.
    margin(x,y) = vote_true_class - max_other_vote (using weighted votes).
    """
    X = np.asarray(X); y = np.asarray(y)
    votes = model._vote_matrix(X)
    # normalize votes to sum to 1 for each sample
    sums = votes.sum(axis=1, keepdims=True); sums[sums==0] = 1.0
    probs = votes / sums
    margins = []
    for i, yi in enumerate(y):
        ci = list(model.classes_).index(yi)
        m = probs[i, ci] - np.max(np.delete(probs[i], ci))
        margins.append(m)
    return float(np.mean(margins))
