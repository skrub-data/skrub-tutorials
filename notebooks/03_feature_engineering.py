# %% 
# In this notebook, we will go over the various transformers provided by skrub
# for feature engineering. 
# As a reminder, feature engineering is the process of transforming raw data
# into features that better represent the underlying problem to the predictive
# models, resulting in improved model performance.

# %%
# First, let's import the necessary libraries and load a sample dataset.
import pandas as pd
from sklearn.datasets import fetch_openml

ames_housing = fetch_openml(data_id=43926)
# %%
data = ames_housing.data
target = ames_housing.target
# %%
from skrub import TableReport

# %%
TableReport(data)
# %%
from skrub._dataframe import _common as sbd 
def has_outliers(column):
    if not sbd.is_numeric(column):
        return False
    q1 = sbd.quantile(column, 0.25) 
    q3 = sbd.quantile(column, 0.75)
    IQR = q3 - q1
    lower_bound = q1 - 1.5 * IQR
    upper_bound = q3 + 1.5 * IQR
    outliers = (column < lower_bound) | (column > upper_bound)
    return any(outliers)

#%%
import skrub.selectors as s


# %%
import skrub
# %%
select = skrub.SelectCols(s.filter(has_outliers))
# %%
