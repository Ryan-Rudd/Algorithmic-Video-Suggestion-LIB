import pandas as pd
from video_suggestion.utils.data_processing import load_data, preprocess_data


def test_load_data():
    # Test loading of a CSV file
    filepath = 'data/raw_data/test_data.csv'
    df = load_data(filepath)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (5, 3)

def test_preprocess_data():
    # Test preprocessing of a DataFrame
    df = pd.DataFrame({'title': ['Introduction to Python', 'Data Science Fundamentals', 'Linear Algebra for Machine Learning', 'Advanced Python Programming', 'Data Visualization with Matplotlib'],
                       'description': ['Learn the basics of Python programming.', 'Discover the key concepts and techniques used in data science.', 'Get a firm grasp on the fundamentals of linear algebra.', 'Explore advanced topics in Python programming.', 'Create beautiful visualizations with Matplotlib.'],
                       'views': [1000, 2000, 3000, 4000, 5000]})
    processed_df = preprocess_data(df)
    assert processed_df.shape == (5, 4)
    assert 'title_description' in processed_df.columns
    assert processed_df['title_description'].iloc[0] == 'introduction to python learn the basics of python programming.'
