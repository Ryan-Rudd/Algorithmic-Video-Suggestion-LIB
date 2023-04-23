
## Usage

To use the Video Suggestion Library in your Python code, import the relevant modules and classes. Here's an example that demonstrates how to generate recommendations for a user using the hybrid approach:

```python
import numpy as np
import pandas as pd
from video_suggestion.algorithms.collaborative_filtering import CollaborativeFiltering
from video_suggestion.algorithms.content_based_filtering import ContentBasedFiltering
from video_suggestion.algorithms.hybrid_approach import HybridApproach
from video_suggestion.utils.data_processing import load_data, preprocess_data

# Load and preprocess data
raw_data = load_data('data/raw_data/video_data.csv')
processed_data = preprocess_data(raw_data)

# Create user-item matrix
users = processed_data['user_id'].unique()
videos = processed_data['video_id'].unique()
user_to_idx = {user: i for i, user in enumerate(users)}
video_to_idx = {video: i for i, video in enumerate(videos)}
X = np.zeros((len(users), len(videos)))
for _, row in processed_data.iterrows():
    user_idx = user_to_idx[row['user_id']]
    video_idx = video_to_idx[row['video_id']]
    X[user_idx, video_idx] = row['rating']

# Initialize collaborative filtering and content-based filtering models
cf_model = CollaborativeFiltering()
cbf_model = ContentBasedFiltering()

# Initialize hybrid approach model
hybrid_model = HybridApproach(cf_model, cbf_model, alpha=0.7)

# Fit models
hybrid_model.fit(X, n_factors=50, n_epochs=20, cf_lr=0.001, cf_reg=0.01)

# Generate recommendations for a user
user_id = 1
user_idx = user_to_idx[user_id]
n_recommendations = 10
recommendations = hybrid_model.predict(user_idx, X, n=n_recommendations)

# Print recommendations
recommended_videos = [videos[i] for i in recommendations]
recommended_titles = raw_data[raw_data['video_id'].isin(recommended_videos)]['title'].unique()
print(f'Recommended videos for user {user_id}:')
for i, title in enumerate(recommended_titles):
    print(f'{i+1}. {title}') 
```