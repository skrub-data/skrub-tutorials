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
