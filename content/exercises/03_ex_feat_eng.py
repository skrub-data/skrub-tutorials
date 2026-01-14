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

# %% [markdown]
# Now modify the script above to add spline features (`periodic_encoding="spline"`). 
#

# %%
# Solution
from skrub import Cleaner
from sklearn.pipeline import make_pipeline
import skrub.selectors as s

datetime_encoder = ApplyToCols(
    DatetimeEncoder(
        periodic_encoding="spline",
        add_total_seconds=True,
        add_weekday=True,
        add_day_of_year=True,
    ),
    cols=s.any_date(),
)

encoder = make_pipeline(Cleaner(datetime_format="%d %B %Y"), datetime_encoder)
encoder.fit_transform(df)

# %%
