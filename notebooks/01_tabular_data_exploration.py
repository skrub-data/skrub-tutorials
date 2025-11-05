#%% 
# In this notebook, we will show how we use the skrub `TableReport` to explore
# tabular data. We will use the Adult Census dataset as our example table. 

# First, let's import the necessary libraries and load the dataset.
import pandas as pd
from skrub import TableReport
from sklearn.datasets import fetch_openml
# Load the Adult Census dataset from OpenML
adult = fetch_openml(name='adult', version=2, as_frame=True)
# %%
data = adult.data
target = adult.target
# %%
# Now, let's create a TableReport to explore the dataset.
TableReport(data)
# %%
# ### Default view of the TableReport
# The `TableReport` gives us a comprehensive overview of the dataset. The default
# view shows all the columns in the dataset, and allows to select and copy the content
# of the cells shown in the preview. 
#
# The `TableReport` is intended to show a preview of the data, so it does not 
# contain all the rows in the dataset, rather it shows only the first and last
# few rows by default. It is possible to change the number of rows shown in the preview
# by using the `n_rows` parameter when creating the `TableReport`: half of `n_rows`
# will be taken from the start of the dataset, and half from the end.
#
# ### The "Stats" tab
# The "Stats" tab provides a variety of descriptive statistics for each column in the dataset.
# This includes:
# - The column name
# - The detected data type of the column
# - Whether the column is sorted or not 
# - The number of null values in the column, as well as the percentage
# - The number of unique values in the column
# 
# For numerical columns, additional statistics are provided:
# - Mean
# - Standard deviation
# - Minimum and maximum values
# - Median

# %%