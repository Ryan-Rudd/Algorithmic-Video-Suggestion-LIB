import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df['title_description'] = df['title'] + ' ' + df['description']
    df['title_description'] = df['title_description'].apply(lambda x: x.lower())
    return df
