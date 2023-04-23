import numpy as np
from video_suggestion.algorithms.collaborative_filtering import CollaborativeFiltering


def test_collaborative_filtering():
    # Create a test matrix
    X = np.array([[5, 3, 0, 1],
                  [4, 0, 0, 1],
                  [1, 1, 0, 5],
                  [1, 0, 0, 4],
                  [0, 1, 5, 4]])

    # Create a collaborative filtering model
    model = CollaborativeFiltering()

    # Fit the model to the test matrix
    model.fit(X)

    # Test predictions on user-item pairs
    assert np.allclose(model.predict(0, 2), 3.15, atol=1e-2)
    assert np.allclose(model.predict(1, 1), 3.75, atol=1e-2)
    assert np.allclose(model.predict(3, 2), 2.44, atol=1e-2)

    # Test recommendation generation for a user
    assert model.recommend_items(1, X) == [2, 3, 0]
    assert model.recommend_items(4, X) == [0, 3, 2]
