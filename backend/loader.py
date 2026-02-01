import pandas as pd

def load_csv(file):
    """
    Load uploaded CSV into Pandas DataFrame
    """
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        raise ValueError(f"Error loading CSV: {e}")
