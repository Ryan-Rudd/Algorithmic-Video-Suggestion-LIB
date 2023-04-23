from typing import List
import numpy as np


class CollaborativeFiltering:
    def __init__(self):
        self.user_factors = None
        self.item_factors = None

    def fit(self, X: np.ndarray, n_factors: int = 20, n_epochs: int = 10,
            lr: float = 0.01, reg: float = 0.01):
        n_users, n_items = X.shape

        # Initialize user and item factors
        self.user_factors = np.random.normal(size=(n_users, n_factors))
        self.item_factors = np.random.normal(size=(n_items, n_factors))

        # Perform stochastic gradient descent
        for epoch in range(n_epochs):
            for i in range(n_users):
                for j in range(n_items):
                    if X[i, j] > 0:
                        error = X[i, j] - self.predict(i, j)
                        self.user_factors[i, :] += lr * \
                            (error * self.item_factors[j, :] - reg * self.user_factors[i, :])
                        self.item_factors[j, :] += lr * \
                            (error * self.user_factors[i, :] - reg * self.item_factors[j, :])

    def predict(self, user_id: int, item_id: int) -> float:
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model has not been trained yet.")
        return self.user_factors[user_id, :].dot(self.item_factors[item_id, :])
    
    def recommend_items(self, user_id: int, X: np.ndarray, n: int = 10) -> List[int]:
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model has not been trained yet.")
        scores = self.user_factors[user_id, :].dot(self.item_factors.T)
        scores[X[user_id, :] > 0] = -np.inf
        top_items = np.argsort(scores)[::-1][:n]
        return top_items.tolist()
