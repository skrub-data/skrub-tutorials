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

# %%
# ## Loading and exploring the dataset
# For this example, we will use the Ames Housing dataset. This dataset includes
# a variety of features about houses in Ames, Iowa, along with their sale prices.
# In particular, the dataset includes both numerical and categorical features,
# making it a good candidate for demonstrating feature engineering techniques.
ames_housing = fetch_openml(data_id=43926)
# %%
data = ames_housing.data
target = ames_housing.target
# %%
# We use the skrub TableReport to get an overview of the dataset and the target.
# We increase the limit on the number of plot columns to 100 to ensure we see all
# features: by default, skrub does not plot columns and does not measure the column
# associations if there are more than 30 columns in the dataset.
from skrub import TableReport

TableReport(data, max_plot_columns=100)

# %%
# Thanks to the "Distributions" tab, we can see that many of the columns in the
# dataset are categorical and imbalanced, which is something that should be
# addressed during feature engineering.
#
# We can use the "Stats" tab to get a summary of the dataset.
# By filtering columns by Categorical type, we can see that there are many categorical,
# and that the categorical columns do not have too many unique values. This means that we can use
# encoding techniques such as One-Hot Encoding or Target Encoding without worrying about
# having too many dimensions.

# %%
# We can now look at the target variable.
TableReport(target)

# %%
# We can see that the target variable is continuous, which means we are dealing with a regression problem.

# %%
# ## Feature Engineering with skrub
# skrub provides a variety of transformers for feature engineering.
# We will start by using the `TableVectorizer` to get a first baseline model,
# and then we will show how we can leverage the skrub column selectors to
# apply specific transformations to specific columns.

from skrub import TableVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error

# %%
# First, we will split the dataset into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=42
)
# %%
# Next, we will create a pipeline with the `TableVectorizer` and a `RandomForestRegressor`.
pipeline = make_pipeline(
    TableVectorizer(), RandomForestRegressor(n_estimators=100, random_state=42)
)

# %%
# We will now evaluate the pipeline using cross-validation.
cv_scores = cross_val_score(
    pipeline, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
)
rmse_scores = (-cv_scores) ** 0.5
print(f"Cross-validated RMSE scores: {rmse_scores}")
print(f"Mean RMSE: {rmse_scores.mean()}")
# %%
# Let's use a ridge regression model instead of random forest to see how it performs.
from sklearn.linear_model import RidgeCV

pipeline_ridge = make_pipeline(TableVectorizer(), RidgeCV())
cv_scores_ridge = cross_val_score(
    pipeline_ridge, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
)
rmse_scores_ridge = (-cv_scores_ridge) ** 0.5
print(f"Cross-validated RMSE scores (Ridge): {rmse_scores_ridge}")
print(f"Mean RMSE (Ridge): {rmse_scores_ridge.mean()}")
# %%
# ### ApplyToCols and selectors for custom feature engineering
# Now that we have a baseline model, we can start applying specific transformations
# to specific columns using the `ApplyToCols` transformer and skrub's column selectors.
import skrub.selectors as s
from skrub import ApplyToCols, SquashingScaler
from sklearn.preprocessing import OneHotEncoder

scale_numeric = ApplyToCols(SquashingScaler(), s.numeric())
encode_categorical = ApplyToCols(
    OneHotEncoder(handle_unknown="ignore", sparse_output=False), s.categorical()
)

pipeline_custom = make_pipeline(
    scale_numeric,
    encode_categorical,
    RidgeCV()
)
# %%
cv_scores_custom = cross_val_score(
    pipeline_custom, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
)
rmse_scores_custom = (-cv_scores_custom) ** 0.5
print(f"Cross-validated RMSE scores (Custom): {rmse_scores_custom}")
print(f"Mean RMSE (Custom): {rmse_scores_custom.mean()}")
# %%