# %% [markdown]
# # Exercise: implementing a `TableVectorizer` from its components
# Replicate the behavior of a `TableVectorizer` using `ApplyToCols`, the skrub
# selectors, and the given transformers.

# %%
%pip install skrub
# %%
from skrub import Cleaner, ApplyToCols, StringEncoder, DatetimeEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
import skrub.selectors as s

# %% [markdown]
# Notes on the implementation:
#
# - In the first step, the TableVectorizer cleans the data to parse datetimes and other
# dtypes.
#   - The `Cleaner` should parse strings as numbers and then convert them to `float32`:
#   Use the `cast_to_str` and `cast_to_float32` parameters set to `True`.
# - Numeric features are left untouched, i.e., they use a Passthrough transformer.
# - String and categorical feature are split into high and low cardinality features.
# - For this exercise, set the the cardinality `threshold` to 4.
# - High cardinality features are transformed with a `StringEncoder`. In this exercise,
# set `n_components` to 2.
# - Low cardinality features are transformed with a `OneHotEncoder` with 
# `sparse_output=False` and `drop="if_binary"`.
# - Remember  `cardinality_below` is one of the skrub selectors.
# - Datetimes are transformed by a default `DatetimeEncoder`.
# - Everything should be wrapped in a scikit-learn `Pipeline`.
# - Remember that the order of the operations matters! (hint: categorical encoders
# convert the data to numeric). 
#
# Use the following dataframe to test the result.

# %%
import pandas as pd

data = {
    "int": [15, 56, 63, 12, 44],
    "float": [5.2, 2.4, 6.2, 10.45, 9.0],
    "str1": ["public", "private", "private", "private", "public"],
    "str2": ["officer", "manager", "lawyer", "chef", "teacher"],
    "bool": [True, False, True, False, True],
    "datetime-col": [
        "2020-02-03T12:30:05",
        "2021-03-15T00:37:15",
        "2022-02-13T17:03:25",
        "2023-05-22T08:45:55",
    ]
    + [None],
}
df = pd.DataFrame(data)
df

# %% [markdown]
# Use the following `PassThrough` transformer where needed.

# %%
from skrub._single_column_transformer import SingleColumnTransformer


class PassThrough(SingleColumnTransformer):
    def fit_transform(self, column, y=None):
        return column

    def transform(self, column):
        return column


# %% [markdown]
# You can test the correctness of your solution by comparing it with the equivalent
# `TableVectorizer`.
# 
# The order and name of the columns in the output may differ, but the values should 
# be the same.

# %%
from skrub import TableVectorizer

tv = TableVectorizer(
    high_cardinality=StringEncoder(n_components=2), cardinality_threshold=4
)
tv.fit_transform(df)

# %%
# Write your code here
#
#
#
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
# Solution
cleaner = ApplyToCols(Cleaner(cast_to_str=True, cast_to_float32=True))
high_cardinality = ApplyToCols(
    StringEncoder(n_components=2), cols=~s.cardinality_below(4) & (s.string())
)
low_cardinality = ApplyToCols(
    OneHotEncoder(sparse_output=False, drop="if_binary"),
    cols=s.cardinality_below(4) & s.string(),
)
numeric = ApplyToCols(PassThrough(), cols=s.numeric())
datetime = ApplyToCols(DatetimeEncoder(), cols=s.any_date())

my_table_vectorizer = make_pipeline(
    cleaner, numeric, high_cardinality, low_cardinality, datetime
)

my_table_vectorizer.fit_transform(df)

# %%
