from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedFiltering:
    def __init__(self):
        self.vectorizer = None
        self.feature_matrix = None

    def fit(self, X: np.ndarray):
        texts = X[:, 1]
        self.vectorizer = TfidfVectorizer()
        self.feature_matrix = self.vectorizer.fit_transform(texts)

    def predict(self, user_id: int, X: np.ndarray, n: int = 10) -> List[int]:
        if self.vectorizer is None or self.feature_matrix is None:
            raise ValueError("Model has not been trained yet.")
        user_text = X[user_id, 1]
        user_features = self.vectorizer.transform([user_text])
        similarities = cosine_similarity(user_features, self.feature_matrix)[0]
        indices = np.argsort(similarities)[::-1][:n]
        return indices.tolist()
