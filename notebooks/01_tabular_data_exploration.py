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
# ### The "Distributions" tab
# The "Distributions" tab provides visualizations of the distributions of values 
# in each column. This includes histograms for numerical columns and bar plots for categorical columns.
# 
# The "Distributions" tab helps with detecting potential issues in the data, such as:
# - Skewed distributions
# - Outliers
# - Unexpected value frequencies
#
# For example, in this dataset we can see that some columns are heavily 
# skewed, such as "workclass", "race", and "native-country": this is important 
# information to keep track of, because these columns may require special handling
# during data preprocessing or modeling.


# %%
# ### The "Associations" tab
# The "Associations" tab provides insights into the relationships between different
# columns in the dataset.
# It shows Pearson's correlation coefficients for numerical columns, as well as
# Cramér's V for all columns. 
#
# While this is a somewhat rough measure of association, it can help identify potential
# relationships worth exploring further during the analysis, and highlights 
# highly correlated columns: depending on the modeling technique used, these may need to be
# handled specially to avoid issues with multicollinearity.
#
# In this example, we can see that "education-num" and "education" have perfect 
# correlation, which means that one of the two columns can be dropped without losing information.

# %%
# # Exploring the target variable
# Let's take a closer look at the target variable, which indicates whether an individual's
# income exceeds $50K per year. We can create a separate `TableReport` for the target variable
# to explore its distribution: 
TableReport(target)

# %% 
# From the distribution, we can see that the dataset is somewhat imbalanced,
# with a larger proportion of individuals earning less than $50K per year.

# %%
# # Building a simple model to predict income level
# Finally, let's build a simple model to predict whether an individual's income
# exceeds $50K per year based on the other features in the dataset.
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# %%
# For simplicity, we will use a Random Forest classifier without any hyperparameter tuning. 
# We wiill use the skrub `TableVectorizer` to preprocess all the features in the dataset.
# The `TableVectorizer` performs a number of preprocessing steps and encoding
# automatically, allowing to quickly build a model without having to manually
# preprocess the data.
# The `TableReport` will be explained in more detail in a later notebook.
from skrub import TableVectorizer
# Create a TableVectorizer to preprocess the data
vectorizer = TableVectorizer()
# Fit and transform the training data, and transform the test data
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
# %%

# Create and train the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
# Make predictions on the test set
y_pred = clf.predict(X_test)
# Evaluate the model's performance
print(classification_report(y_test, y_pred))
# %%
# From the classification report, we can see that the model performs reasonably well
# on the majority class, but has lower precision and recall for the minority class.
# This is expected and likely due to the class imbalance in the dataset.
# In a real-world scenario, we would want to explore techniques to address this imbalance,
# such as resampling or using different evaluation metrics.