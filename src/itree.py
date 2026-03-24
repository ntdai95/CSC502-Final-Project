from dataclasses import dataclass
import numpy as np
from src.iforest_math import c_factor


@dataclass
class ExternalNode:
    size: int


@dataclass
class InternalNode:
    split_att: int
    split_value: float
    left: object
    right: object


class IsolationTree:
    def __init__(self, height_limit, rng):
        self.height_limit = height_limit
        self.rng = rng
        self.root = None


    def fit(self, X):
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        
        self.root = self._fit(X, current_height=0)
        return self


    def _fit(self, X, current_height):
        n_samples, _ = X.shape
        if current_height >= self.height_limit or n_samples <= 1:
            return ExternalNode(n_samples)

        col_mins = X.min(axis=0)
        col_maxs = X.max(axis=0)
        valid_features = np.where(col_maxs > col_mins)[0]
        if len(valid_features) == 0:
            return ExternalNode(n_samples)

        split_att = int(self.rng.choice(valid_features))
        min_val = float(col_mins[split_att])
        max_val = float(col_maxs[split_att])
        split_value = float(self.rng.uniform(min_val, max_val))
        left_mask = X[:, split_att] < split_value
        right_mask = ~left_mask
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return ExternalNode(n_samples)

        left = self._fit(X[left_mask], current_height + 1)
        right = self._fit(X[right_mask], current_height + 1)
        return InternalNode(split_att, split_value, left, right)
    

    def path_length(self, x):
        if self.root is None:
            raise RuntimeError("Tree is not fitted yet.")
        
        return self._path_length(x, self.root, 0)


    def _path_length(self, x, node, current_height):
        if isinstance(node, ExternalNode):
            if node.size <= 1:
                return float(current_height)
            
            return float(current_height) + c_factor(node.size)

        if x[node.split_att] < node.split_value:
            return self._path_length(x, node.left, current_height + 1)
        
        return self._path_length(x, node.right, current_height + 1)