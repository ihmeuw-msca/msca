"""
Peng's original metrics.
"""

import pandas as pd


# compute mean average as the reference prediction
def get_weighted_mean(
    data: pd.DataFrame, val: str, weights: str, by: list[str], name: str
) -> pd.DataFrame:
    data = data[by + [val, weights]].copy()
    data[name] = data[val] * data[weights]
    data = data.groupby(by)[[name, weights]].sum().reset_index()
    data[name] = data[name] / data[weights]
    return data.drop(columns=weights)
