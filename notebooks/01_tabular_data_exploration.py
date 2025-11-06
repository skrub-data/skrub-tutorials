#%% 
# In this notebook, we will show how we use the skrub `TableReport` to explore
# tabular data. We will use the Adult Census dataset as our example table. 

# First, let's import the necessary libraries and load the dataset.
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
#
# Additionally, the "Distributions" tab allows so select columns manually, so that
# they can be added to a script and selected for further analysis or modeling.
#

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
# To do so, we need to import a few additional libraries from scikit-learn.
# We use a Logistic Regression classifier for this example, without hyperparameter tuning.
# We use the `StandardScaler` to standardize the numerical features,
# and we use a pipeline to combine the scaler and the classifier.
# Additionally, we import the `train_test_split` function to split the dataset into training and testing sets,
# and the `classification_report` function to evaluate the model's performance.
# 
# As a reminder, to avoid data leakage, we should be training only on the training set
# and evaluating only on the test set, which is why we use `train_test_split` to create
# these splits.

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# %% 
# ## First attempt: using only numerical features
# As a first attempt, let's build a model using only the numerical features in the dataset.
# We can select the numerical columns using the `TableReport`, by clicking on the 
# drop-down menu next to "Associations" and selecting "Numeric". Then, we can 
# click on "Select all" to add all column names to the form, and then click on
# the Copy button to copy the list of column names to the clipboard.

numerical_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
# Select only the numerical columns for training and testing
X_train_num = X_train[numerical_columns]
X_test_num = X_test[numerical_columns]

# Create the scaler
scaler = StandardScaler()
# Create and train the Logistic Regression classifier
clf_num = LogisticRegression(max_iter=1000)

# Create a pipeline that first scales the data, then fits the classifier
clf_num = make_pipeline(scaler, clf_num)
# Fit the model on the training data
clf_num.fit(X_train_num, y_train)
# Make predictions on the test set
y_pred_num = clf_num.predict(X_test_num)
# Evaluate the model's performance
print(classification_report(y_test, y_pred_num))

# %%
# From the classification report, we can see that the model performs reasonably well
# on the majority class, but has lower precision and recall for the minority class.
# We can likely improve the model's performance by including the categorical features
# in the dataset, which we will do next.

# %%
# ## Second attempt: using all features with skrub's TableVectorizer
# We will use the skrub `TableVectorizer` to preprocess all the features in the dataset.
# The `TableVectorizer` is intended to work out-of-the-box with minimal configuration,
# and can handle both numerical and categorical features automatically, although
# it is also possible to customize its behavior if needed. 
# We will use the default settings for this example, and explore its capabilities in more detail
# in a later notebook.

from skrub import TableVectorizer
# Create a TableVectorizer to preprocess the data
vectorizer = TableVectorizer()
# %% 
# The `TableVectorizer` follows the scikit-learn convention of having `fit` and `transform` methods.
# We will fit the vectorizer on the training data, and then transform both the training
# and testing data.
#

# %%
# We can inspect the transformed data to see how the categorical features
# have been encoded. The `TableVectorizer` uses different encoders for different
# types of categorical features, based on their cardinality.
X_transformed = vectorizer.fit_transform(X_train)
TableReport(X_transformed)
# %%
# Now, we can scale features, then create and train the Logistic Regression classifier using the preprocessed data.

# Create the scaler
scaler = StandardScaler()
# 
# Create and train the Logistic Regression classifier
clf = LogisticRegression(max_iter=1000)

# Once again, we build a pipeliine that combines each step of the preprocessing and modeling process.

pipeline = make_pipeline(vectorizer, scaler, clf)

X_train = pipeline.fit(X_train, y_train)
# Make predictions on the test set
y_pred = pipeline.predict(X_test)
# Evaluate the model's performance
print(classification_report(y_test, y_pred))
# %%
# We can observe that the model's performance has improved for both classes, with
# the precision and recall for the minority class increasing significantly.
# In a real-world scenario, we would want to explore techniques to address the class imbalance,
# such as resampling or using different evaluation metrics; however, this is beyond the scope of this notebook.
# %%
