# %% [markdown]
# # Exercise
# Use one of the methods explained so far (Cleaner/ApplyToCols) to convert the provided
# dataframe to datetime dtype, then extract the following features:
# - All parts of the datetime
# - The number of seconds from epoch
# - The day in the week
# - The day of the year
#
# **Hint**: use the format `"%d %B %Y"` for the datetime.
#

# %%
%pip install skrub

# %%
import pandas as pd

data = {
    "admission_dates": [
        "03 January 2023",
        "15 February 2023",
        "27 March 2023",
        "10 April 2023",
    ],
    "patient_ids": [101, 102, 103, 104],
    "age": [25, 34, 45, 52],
    "outcome": ["Recovered", "Under Treatment", "Recovered", "Deceased"],
}
df = pd.DataFrame(data)
print(df)

# %%
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
#
#
#
#
#

# %%
# Solution with ApplyToCols and ToDatetime
from skrub import ApplyToCols, ToDatetime, DatetimeEncoder
from sklearn.pipeline import make_pipeline
import skrub.selectors as s

to_datetime_encoder = ApplyToCols(ToDatetime(format="%d %B %Y"), cols="admission_dates")

datetime_encoder = ApplyToCols(
    DatetimeEncoder(add_total_seconds=True, add_weekday=True, add_day_of_year=True),
    cols=s.any_date(),
)

encoder = make_pipeline(to_datetime_encoder, datetime_encoder)
encoder.fit_transform(df)

# %%
# Solution with Cleaner
from skrub import Cleaner
from sklearn.pipeline import make_pipeline
import skrub.selectors as s

datetime_encoder = ApplyToCols(
    DatetimeEncoder(add_total_seconds=True, add_weekday=True, add_day_of_year=True),
    cols=s.any_date(),
)

encoder = make_pipeline(Cleaner(datetime_format="%d %B %Y"), datetime_encoder)
encoder.fit_transform(df)

# %% [markdown]
# Modify the script so that the `DatetimeEncoder` adds periodic encoding with sine
# and cosine (aka circular encoding):

# %%
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
#
#
#
#
#

# %%
# Solution
from skrub import Cleaner
from sklearn.pipeline import make_pipeline
import skrub.selectors as s

datetime_encoder = ApplyToCols(
    DatetimeEncoder(
        periodic_encoding="circular",
        add_total_seconds=True,
        add_weekday=True,
        add_day_of_year=True,
    ),
    cols=s.any_date(),
)

encoder = make_pipeline(Cleaner(datetime_format="%d %B %Y"), datetime_encoder)
encoder.fit_transform(df)

# %% [markdown]
# # Exercise
# Build a custom `SingleColumnTransformer` that unpacks the combined string column
# in the provided dataframe into separate columns for `str_id`, `num_id`, and
# `datetime`. The `datetime` column should be converted to datetime dtype. Then,
# use this transformer in a pipeline to extract datetime features as shown in
# the previous exercises.
#
# The transformer should reject columns that are not of string type or that cannot 
# be unpacked properly.
# IDs are in the format `STR-NUM-DATETIME`, where `STR` is a string identifier, 
# `NUM` is a numeric identifier, and `DATETIME` is a Unix timestamp.
#
# Hint: you can use the following snippet to extract the components from the string column:
# ```python
# split_data = X.str.split("-", expand=True)
# res = pd.DataFrame(
#     {
#         "str_id": split_data[0],
#         "num_id": split_data[1].astype("int64"),
#         "datetime": pd.to_datetime(split_data[2].astype("int64"), unit="s"),
#     }
# )
# ```

# %%
from skrub.core import SingleColumnTransformer, RejectColumn
import pandas as pd
from skrub import ApplyToCols
df_id = pd.DataFrame(
    {
        "id": [
            "BQG-1001-1577836800",
            "TYW-1002-1577923200",
            "JAY-1003-1578009600",
        ]
    }
)
# %%
# Write your solution here
#
#
#
#
#
#
#

# %%
# Solution
class Unpacker(SingleColumnTransformer):
    """Unpacker for pandas DataFrames."""

    def fit_transform(self, X, y=None):
        """Unpack combined string column into separate columns."""
        if X.dtype != object:
            raise RejectColumn("UnpackerPandas only works on string columns.")
        try:
            split_data = X.str.split("-", expand=True)
            res = pd.DataFrame(
                {
                    "str_id": split_data[0],
                    "num_id": split_data[1].astype("int64"),
                    "datetime": pd.to_datetime(split_data[2].astype("int64"), unit="s"),
                }
            )
            return res
        except Exception as exc:
            raise RejectColumn("UnpackerPandas failed to unpack the column.") from exc


ApplyToCols(Unpacker(), allow_reject=True).fit_transform(df_id)

# %% [markdown]
# Now use this `Unpacker` in a pipeline to extract datetime features as shown in
# the previous exercises. You can use the default `DatetimeEncoder` settings for
# this part.

# %%
# Write your solution here
#
#
#
#
#
#
#

# %%
from sklearn.pipeline import make_pipeline
from skrub import DatetimeEncoder

pipeline = make_pipeline(
    ApplyToCols(Unpacker(), allow_reject=True),
    ApplyToCols(DatetimeEncoder(), allow_reject=True),
)
pipeline.fit_transform(df_id)
# %%
