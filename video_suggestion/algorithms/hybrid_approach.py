from typing import List
import numpy as np
from collaborative_filtering import CollaborativeFiltering
from content_based_filtering import ContentBasedFiltering


class HybridApproach:
    def __init__(self, cf_model: CollaborativeFiltering, cbf_model: ContentBasedFiltering, alpha: float = 0.5):
        self.cf_model = cf_model
        self.cbf_model = cbf_model
        self.alpha = alpha

    def fit(self, X: np.ndarray, n_factors: int = 20, n_epochs: int = 10,
            cf_lr: float = 0.01, cf_reg: float = 0.01):
        # Fit collaborative filtering model
        self.cf_model.fit(X, n_factors=n_factors, n_epochs=n_epochs, lr=cf_lr, reg=cf_reg)

        # Fit content-based filtering model
        self.cbf_model.fit(X)

    def predict(self, user_id: int, X: np.ndarray, n: int = 10) -> List[int]:
        # Use collaborative filtering to generate candidate recommendations
        cf_recommendations = self.cf_model.recommend_items(user_id, X, n=n)

        # Use content-based filtering to rerank candidate recommendations
        cbf_recommendations = self.cbf_model.predict(user_id, X, n=n)

        # Combine recommendations using a weighted sum of ranks
        cf_ranks = {item_id: rank for rank, item_id in enumerate(cf_recommendations)}
        cbf_ranks = {item_id: rank for rank, item_id in enumerate(cbf_recommendations)}
        item_ranks = {item_id: self.alpha * cf_ranks.get(item_id, n) + (1 - self.alpha) * cbf_ranks.get(item_id, n) for item_id in set(cf_recommendations + cbf_recommendations)}
        ranked_items = sorted(item_ranks, key=item_ranks.get)[:n]

        return ranked_items
