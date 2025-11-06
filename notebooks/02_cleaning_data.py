# %%
# In this notebook, we will show how we can quickly pre-process and sanitize 
# data using skrub's `Cleaner`, and compare it to traditional methods using pandas.
#
# The `Cleaner` is intended to be a first step in preparing tabular data for 
# analysis or modeling, and can handle a variety of common data cleaning tasks
# automatically. It is designed to work out-of-the-box with minimal configuration,
# although it is also possible to customize its behavior if needed.

#
# ## Exercise
# Given the following dataframe, use skrub's `Cleaner` to clean the data so that:
# - Constant columns are removed
# - All columns with more than 50% missing values are removed
#
# Additionally, use `deduplicate` to reduce the number of unique values in the
# `variety` column.

import pandas as pd
from skrub import Cleaner, TableReport
from sklearn.datasets import fetch_openml

# Load the `wine_reviews` dataset from OpenML
wine = fetch_openml(data_id=42074)
# %%
TableReport(wine.data)
# %%
cleaner = Cleaner(drop_null_fraction=0.5)
transformed = cleaner.fit_transform(wine.data)
# %%
TableReport(transformed)
# %%
from skrub import deduplicate
# %%
dedup = deduplicate(transformed["variety"])
# %%
transformed["variety_dedup"] = transformed["variety"].map(dedup.to_dict())
# %%
TableReport(transformed[["variety", "variety_dedup"]])
# %%
