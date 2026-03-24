import math
import numpy as np
from src.iforest_math import anomaly_score
from src.itree import IsolationTree


class IsolationForestCustom:
    def __init__(self, n_trees=100, sample_size=256, random_state=42):
        if n_trees <= 0:
            raise ValueError("n_trees must be positive.")
        
        if sample_size <= 1:
            raise ValueError("sample_size must be greater than 1.")

        self.n_trees = n_trees
        self.sample_size = sample_size
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.height_limit = math.ceil(math.log2(sample_size))
        self.effective_sample_size = sample_size
        self.trees = []


    def fit(self, X):
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        n_samples = X.shape[0]
        if n_samples < 2:
            raise ValueError("Need at least 2 samples to fit Isolation Forest.")

        actual_sample_size = min(self.sample_size, n_samples)
        self.effective_sample_size = actual_sample_size
        self.height_limit = math.ceil(math.log2(max(actual_sample_size, 2)))
        self.trees = []
        for _ in range(self.n_trees):
            indices = self.rng.choice(n_samples, size=actual_sample_size, replace=False)
            sample = X[indices]
            tree_seed = int(self.rng.integers(0, 1_000_000_000))
            tree_rng = np.random.default_rng(tree_seed)
            tree = IsolationTree(self.height_limit, tree_rng).fit(sample)
            self.trees.append(tree)

        return self


    def _mean_path_length(self, X):
        if not self.trees:
            raise RuntimeError("Model is not fitted yet.")

        path_lengths = np.zeros(X.shape[0], dtype=float)
        for tree in self.trees:
            tree_paths = np.array([tree.path_length(row) for row in X], dtype=float)
            path_lengths += tree_paths

        return path_lengths / len(self.trees)


    def score_samples(self, X):
        mean_path_lengths = self._mean_path_length(X)
        return np.array([anomaly_score(path_length, self.effective_sample_size) for path_length in mean_path_lengths], dtype=float)


    def predict(self, X, threshold=0.60):
        scores = self.score_samples(X)
        return (scores >= threshold).astype(int)