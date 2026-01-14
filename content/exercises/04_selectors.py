# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Exercise: using selectors together with `ApplyToCols`
# Consider this example dataframe:

# %%
import pandas as pd

df = pd.DataFrame(
    {
        "metric_1": [10.5, 20.3, 30.1, 40.2],
        "metric_2": [5.1, 15.6, None, 35.8],
        "metric_3": [1.1, 3.3, 2.6, .8],
        "num_id": [101, 102, 103, 104],
        "str_id": ["A101", "A102", "A103", "A104"],
        "description": ["apple", None, "cherry", "date"],
        "name": ["Alice", "Bob", "Charlie", "David"],
    }
)
df

# %% [markdown]
# Using the skrub selectors and `ApplyToCols`:
#
# - Apply the `StandardScaler` to numeric columns, except `"num_id"`. 
# - Apply a `OneHotEncoder` with `sparse_output=False` on all string columns except
# `"str_id"`. 

# %%
import skrub.selectors as s
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from skrub import ApplyToCols
from sklearn.pipeline import make_pipeline

# Write your solution here
# 
# 
# 
# 
# 
# 
# 
# 
# 

# %%
import skrub.selectors as s
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from skrub import ApplyToCols
from sklearn.pipeline import make_pipeline

scaler = ApplyToCols(StandardScaler(), cols=s.numeric() - "num_id")
one_hot = ApplyToCols(OneHotEncoder(sparse_output=False), cols=s.string() - "str_id")

transformer = make_pipeline(scaler, one_hot)

transformer.fit_transform(df)

# %% [markdown]
# Given the same dataframe and using selectors, drop only string columns that contain
# nulls. 

# %%
from skrub import DropCols

# Write your solution here
# 
# 
# 
# 
# 
# 
# 

# %%
from skrub import DropCols

DropCols(cols=s.has_nulls() & s.string()).fit_transform(df)

# %% [markdown]
# Now write a custom function that selects columns where all values are lower than
# `10.0`. 

# %%
from skrub import SelectCols

# Write your solution here
# 
# 
# 
# 
# 
# 
# 

# %%
from skrub import SelectCols

def lower_than(col):
    return all(col < 10.0)

SelectCols(cols=s.numeric() & s.filter(lower_than)).fit_transform(df)

# %%
