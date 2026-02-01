import pandas as pd
def build_null_duplicate_table(df):
    return pd.DataFrame(
        [
            df.isnull().sum().values,
            [df[col].duplicated().sum() for col in df.columns]
        ],
        index=["Null Count", "Duplicate Count"],
        columns=df.columns
    )