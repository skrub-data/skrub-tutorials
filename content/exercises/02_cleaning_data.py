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
# # Exercise: clean a dataframe using the `Cleaner` 
# Load the given dataframe. 

# %%
import pandas as pd
df = pd.read_csv("../data/cleaner_data.csv")

# %% [markdown]
# Use the `TableReport` to answer the following questions: 
#
# - Are there constant columns? 
# - Are there datetime columns? If so, were they parsed correctly? 
# - What is the dtype of the numerical features? 

# %%
from skrub import TableReport
TableReport(df)

# %% [markdown]
# Then, use the `Cleaner` to sanitize the data so that:
# - Constant columns are removed
# - Datetimes are parsed properly (hint: use `"%d-%b-%Y"` as the datetime format)
# - All columns with more than 50% missing values are removed
# - Numerical features are converted to `float32`

# %%
from skrub import Cleaner

# Write your answer here
# 
# 
# 
# 
# 
# 
# 
# 

# %%
# solution
from skrub import Cleaner

cleaner = Cleaner(
    drop_if_constant=True,
    drop_null_fraction=0.5,
    numeric_dtype="float32",
    datetime_format="%d-%b-%Y",
)

# Apply the cleaner
df_cleaned = cleaner.fit_transform(df)

# Display the cleaned dataframe
TableReport(df_cleaned)

# %% [markdown]
# We can inspect which columns were dropped and what transformations were applied:

# %%
print(f"Original shape: {df.shape}")
print(f"Cleaned shape: {df_cleaned.shape}")
print(
    f"\nColumns dropped: {[col for col in df.columns if col not in cleaner.all_outputs_]}"
)
