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
# # Exercise 
# In this exercise we're going to use the `TableVectorizer` and `tabular_pipeline` 
# to replicate the behavior of a traditional scikit-learn pipeline. 
#
# First, let's load the dataset: 

# %%
import pandas as pd

X = pd.read_csv("../data/adult_census/data.csv")
y = pd.read_csv("../data/adult_census/target.csv")

# %% [markdown]
# This is the pipeline that needs to be replicated. 
#
# - It uses `LogisticRegression` as the classifier, i.e., a linear model. 
# - It scales numerical features using a `StandardScaler`.
# - Categorical features are one-hot-encoded.
# - Missing values are imputed using a `SimpleImputer`. 

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector as selector
from sklearn.compose import make_column_transformer

categorical_columns = selector(dtype_include="category")(X)
numerical_columns = selector(dtype_include="number")(X)

ct = make_column_transformer(
      (StandardScaler(),
       numerical_columns),
      (OneHotEncoder(handle_unknown="ignore"),
       categorical_columns))

model_base = make_pipeline(ct, SimpleImputer(), LogisticRegression())
model_base

# %% [markdown]
# Use the `TableVectorizer` and `make_pipeline` to write a pipeline named 
# `model_tv`, which includes all the steps necessary for the `LogisticRegression` 
# to work. 

# %%
from skrub import TableVectorizer
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

# %%
from skrub import TableVectorizer

tv = TableVectorizer()

model_tv = make_pipeline(tv, SimpleImputer(), StandardScaler(), LogisticRegression())
model_tv

# %% [markdown]
# Now use the `tabular_pipeline` to get a new pipeline named `model_tp`. 

# %%
from skrub import tabular_pipeline
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

# %%
from skrub import tabular_pipeline

model_tp = tabular_pipeline(LogisticRegression())
model_tp

# %% [markdown]
# For reference, let's also create a pipeline that uses 
# `HistGradientBoostingClassifier`. This can be done by passing the string 
# "classification" to `tabular_pipeline`. 

# %%
model_hgb = tabular_pipeline("classification")
# model_hgb

# %% [markdown]
# Finally, let's evaluate the different models and see how they perform: 

# %%
from sklearn.model_selection import cross_val_score

results_base = cross_val_score(model_base, X, y)
print(f"Base model: {results_base.mean():.4f}")

results_tv = cross_val_score(model_tv, X, y)
print(f"TableVectorizer: {results_tv.mean():.4f}")

results_tp = cross_val_score(model_tp, X, y)
print(f"Tabular pipeline: {results_tp.mean():.4f}")

results_hgb = cross_val_score(model_hgb, X, y)
print(f"HGB model: {results_hgb.mean():.4f}")

# %% [markdown]
# Unsurprisiingly, the model that uses HGB outperforms the other models, while
# being much slower to train. 
