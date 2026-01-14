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
# # Exercise: exploring a new table
# For this exercise, we will use the `employee_salaries` dataframe to answer some 
# questions. 
#
# Run the following code to import the dataframe:

# %%
import pandas as pd
data = pd.read_csv("../data/employee_salaries/data.csv")

# %% [markdown]
# Now use the skrub `TableReport` and answer the following questions: 

# %%
from skrub import TableReport
TableReport(data)

# %% [markdown]
# ## Questions
# - What's the size of the dataframe? (columns and rows)
# - How many columns have object/numerical/datetime
# - Are there columns with a large number of missing values?
# - Are there columns that have a high cardinality (>40 unique values)?
# - Were datetime columns parsed correctly?
# - Which columns have outliers?
# - Which columns have an imbalanced distribution?
# - Which columns are strongly correlated with each other?
#
# ```{.python}
# # PLACEHOLDER
# #
# #
# #
# #
# #
# #
# #
# #
# #
# ```
#
# ## Answers
# - What's the size of the dataframe? (columns and rows)
#     - 9228 rows × 8 columns
# - How many columns have object/numerical/datetime
#     - No datetime columns, one integer column (`year_first_hired`), all other columns
#     are objects. 
# - Are there columns with a large number of missing values?
#     - No, only the `gender` column contains a small fraction (0.2%) of missing
#     values.
# - Are there columns that have a high cardinality?
#     - Yes, `division`, `employee_position_title`, `date_first_hired` have a 
#     cardinality larger than 40. 
# - Were datetime columns parsed correctly?
#     - No, the `date_first_hired` column has dtype Object. 
# - Which columns have outliers?
#     - No columns seem to include outliers. 
# - Which columns have an imbalanced distribution?
#     - `assignment_category` has an unbalanced distribution. 
# - Which columns are strongly correlated with each other?
#     - `department` and `department_name` have a Cramer's V of 1, so they are 
#     very strongly correlated. 
