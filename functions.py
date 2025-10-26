from matplotlib import pyplot as plt

from main import airports_dataframe, numeric_cols, labels_ukr, categorical_cols
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px


# Замінює порожні значення на середні по числовим, та випадкові з категорійних
def fill_missing_with_mean_or_mode(df: pd.DataFrame) -> pd.DataFrame:
    df_filled = df.copy()
    for col in df.columns:
        missing_count = df_filled[col].isna().sum()
        if missing_count > 0:
            if pd.api.types.is_numeric_dtype(df_filled[col]):
                mean_value = df_filled[col].mean()
                df_filled[col] = df_filled[col].fillna(mean_value)
            else:
                value_counts = df_filled[col].value_counts(normalize=True, dropna=True)

                if not value_counts.empty:
                    random_values = np.random.choice(
                        value_counts.index,
                        size=missing_count,
                        p=value_counts.values
                    )
                    mask = df_filled[col].isna()
                    df_filled.loc[mask, col] = random_values
                else:
                    df_filled[col] = df_filled[col].fillna('Unknown')
    return df_filled


# Видаляє рядки у яких є порожні клітинки
def drop_rows_with_missing(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df_clean = df.dropna()
    after = len(df_clean)
    return df_clean