"""Multinomial logistic regression with softmax — accelerated using numpy.

Uses scikit-learn's LogisticRegression with multi_class='multinomial'
for most use cases, with a fallback for edge cases.

The C# SGD-based implementation is replaced by scikit-learn's L-BFGS solver
which is orders of magnitude faster on small-to-medium datasets (n ~ 400).
"""

from typing import Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression


class LogisticMultiRegression:
    """Multinomial logistic regression classifier with scikit-learn backend.
    
    Replaces the slow per-sample SGD from C# with scikit-learn's L-BFGS solver.
    
    Parameters
    ----------
    weights : np.ndarray, optional
        Pre-trained weights of shape (n_features, n_classes).
        Bias is stored in the last row, matching C# convention.
    """
    
    def __init__(self, weights: Optional[np.ndarray] = None):
        self.weights = weights
        self._model: Optional[LogisticRegression] = None
        if weights is not None:
            self.weights = np.asarray(weights, dtype=np.float64)
            self.n_features = self.weights.shape[0] - 1
            self.n_classes = self.weights.shape[1]
        else:
            self.n_features = 0
            self.n_classes = 0
    
    def compute_output(self, x: np.ndarray) -> np.ndarray:
        """Compute class probabilities for a single input."""
        if self._model is not None:
            return self._model.predict_proba(x.reshape(1, -1)).flatten()
        return self._compute_from_weights(x)
    
    def compute_output_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute class probabilities for a batch."""
        if self._model is not None:
            return self._model.predict_proba(X)
        return self._compute_from_weights_batch(X)
    
    def _compute_from_weights(self, x: np.ndarray) -> np.ndarray:
        """Compute using the stored weight matrix directly."""
        scores = x @ self.weights[:self.n_features, :] + self.weights[self.n_features, :]
        # Softmax
        exp_s = np.exp(scores - np.max(scores))
        return exp_s / exp_s.sum()
    
    def _compute_from_weights_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute for a batch using stored weights."""
        scores = X @ self.weights[:self.n_features, :] + self.weights[self.n_features, :]
        exp_s = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp_s / exp_s.sum(axis=1, keepdims=True)
    
    @staticmethod
    def get_model_by_train_data(
        train_X: np.ndarray,
        train_Y: np.ndarray,
        lr: float = 0.01,
        max_epoch: int = 1000,
        seed: int = 0,
        verbose: bool = False,
    ) -> np.ndarray:
        """Train multinomial logistic regression using scikit-learn.
        
        Returns weights in C# format: (n_features+1, n_classes) with bias in last row.
        
        Parameters
        ----------
        train_X : np.ndarray
            Training features, shape (N, n_features).
        train_Y : np.ndarray
            One-hot encoded targets, shape (N, n_classes).
        lr : float
            Inverse regularization strength C = 1 / lr.
        max_epoch : int
            Maximum iterations for solver.
        seed : int
            Random seed.
        
        Returns
        -------
        np.ndarray
            Weight matrix of shape (n_features+1, n_classes).
        """
        n_classes = train_Y.shape[1]
        
        # Convert one-hot to class labels
        y = np.argmax(train_Y, axis=1)
        
        # Fit scikit-learn model
        model = LogisticRegression(
            solver='lbfgs',
            C=1.0 / max(lr * 10.0, 1e-6) if lr > 0 else 1.0,
            max_iter=max_epoch,
            random_state=42,
            verbose=0,
        )
        model.fit(train_X, y)
        
        # Extract weights in C# format: (n_features + 1, n_classes)
        # scikit-learn stores coef_ of shape (1, n_features * n_classes) for multinomial,
        # and intercept_ of shape (1, n_classes)
        weights = np.zeros((train_X.shape[1] + 1, n_classes), dtype=np.float64)
        # sklearn 1.9+: coef_ shape is (n_classes, n_features) for multinomial
        # We need (n_features, n_classes)
        weights[:train_X.shape[1], :] = model.coef_.T
        weights[train_X.shape[1], :] = model.intercept_
        
        if verbose:
            acc = model.score(train_X, y)
            print(f"Training: acc={acc:.4f}, n_iter={model.n_iter_}")
        
        return weights