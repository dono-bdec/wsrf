import numpy as np
from math import floor, log2
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

@dataclass
class _TreeRecord:
    model: object
    features: np.ndarray
    weight: float

class WSRFClassifier:
    """Weighted Subspace Random Forest (Python translation of R wsrf).
    Implements: fit, predict, predict_proba, summary-like repr.
    Tracks: feature_importances_, subspace usage counts, per-tree OOB error.
    """
    def __init__(self, n_estimators=500, random_state=None, nodesize=1, use_weights=True):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.nodesize = nodesize
        self.use_weights = use_weights
        self.trees_ = []                 # list[_TreeRecord]
        self.feature_importances_ = None # initial RF importances (normalized)
        self.classes_ = None
        self.oob_errors_ = None          # per-tree OOB error
        self.subspace_counts_ = None     # times a feature is selected in subspaces

    def _check_is_fitted(self):
        """Check if the model has been fitted."""
        if self.classes_ is None or len(self.trees_) == 0:
            raise AttributeError("This WSRFClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
    
    def _default_mtry(self, M):
        return max(1, floor(log2(M)) + 1)

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X); y = np.asarray(y)
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)

        # initial RF for importances (matching R defaults as closely as possible)
        mtry = self._default_mtry(n_features)
        init_rf = RandomForestClassifier(
            n_estimators=min(50, self.n_estimators),
            max_features=mtry,
            criterion="entropy",
            min_samples_leaf=self.nodesize,
            random_state=self.random_state,
            n_jobs=1,
        ).fit(X, y)
        imp = init_rf.feature_importances_.clip(min=1e-6)
        imp = imp / imp.sum()
        self.feature_importances_ = imp

        # storage
        self.trees_.clear()
        self.oob_errors_ = np.zeros(self.n_estimators, dtype=float)
        self.subspace_counts_ = np.zeros(n_features, dtype=int)

        # build ensemble
        for k in range(self.n_estimators):
            p = imp if self.use_weights else None
            feats = rng.choice(np.arange(n_features), size=mtry, replace=False, p=p)
            self.subspace_counts_[feats] += 1

            idx = rng.randint(0, n_samples, size=n_samples)
            oob = np.ones(n_samples, dtype=bool); oob[idx] = False

            tree = DecisionTreeClassifier(
                criterion="entropy",
                min_samples_leaf=self.nodesize,
                random_state=rng,
            ).fit(X[idx][:, feats], y[idx])

            if oob.sum() > 0:
                err = np.mean(tree.predict(X[oob][:, feats]) != y[oob])
            else:
                err = 1.0

            self.oob_errors_[k] = err
            weight = max(0.0, 1.0 - err)
            self.trees_.append(_TreeRecord(tree, feats, weight))

        # normalize weights
        total_w = sum(tr.weight for tr in self.trees_)
        if total_w == 0:
            avg = 1.0 / len(self.trees_)
            for tr in self.trees_: tr.weight = avg
        else:
            for tr in self.trees_: tr.weight /= total_w

        return self

    def predict(self, X):
        self._check_is_fitted()
        X = np.asarray(X)
        n_features = len(self.feature_importances_)
        if X.shape[1] != n_features:
            raise ValueError(f"X has {X.shape[1]} features, but WSRFClassifier is expecting {n_features} features as input.")
        votes = self._vote_matrix(X)
        return self.classes_[np.argmax(votes, axis=1)]

    def predict_proba(self, X):
        self._check_is_fitted()
        X = np.asarray(X)
        n_features = len(self.feature_importances_)
        if X.shape[1] != n_features:
            raise ValueError(f"X has {X.shape[1]} features, but WSRFClassifier is expecting {n_features} features as input.")
        # Weighted sum of per-tree probabilities
        probs = np.zeros((X.shape[0], len(self.classes_)))
        for tr in self.trees_:
            p = tr.model.predict_proba(X[:, tr.features])
            probs += tr.weight * p
        # Normalize rows for safety
        sums = probs.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        return probs / sums

    # helper: weighted vote counts (labels, not probabilities)
    def _vote_matrix(self, X):
        self._check_is_fitted()
        votes = np.zeros((X.shape[0], len(self.classes_)))
        for tr in self.trees_:
            pred = tr.model.predict(X[:, tr.features])
            for ci, cls in enumerate(self.classes_):
                votes[:, ci] += tr.weight * (pred == cls)
        return votes

    def __repr__(self):
        n_feat = len(self.feature_importances_) if self.feature_importances_ is not None else "NA"
        return (f"WSRFClassifier(n_estimators={self.n_estimators}, nodesize={self.nodesize}, "
                f"use_weights={self.use_weights}, n_features={n_feat})")
