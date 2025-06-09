from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
import xgboost as xgb
import pandas as pd
import numpy as np
from workalendar.europe import France
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
#st.set_page_config(layout="wide", page_title="Who survived the titanic?")

train_data = pd.read_csv("train.csv")
train_data = st.data_editor(train_data, num_rows = "dynamic")

from sklearn.model_selection import train_test_split

# Remove rows with missing target, separate target from predictors
#train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data.Survived              
train_data.drop(['Survived'], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(train_data, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]

# Select numeric columns
numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

# One-hot encode the data (to shorten the code, we use pandas)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

# Streamlit sliders for hyperparameters
st.sidebar.header("XGBoost Hyperparameters")
Set_learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
Set_max_depth = st.sidebar.slider("Max Depth", 1, 20, 8)
Set_n_estimators = st.sidebar.slider("Number of Estimators", 100, 1000, 500)
Set_min_child_weight = st.sidebar.slider("Min Child Weight", 1, 10, 5)
Set_reg_alpha = st.sidebar.slider("Regularization Alpha", 0.0, 1.0, 0.1, 0.01)
Set_colsample_bytree = st.sidebar.slider("Colsample by Tree", 0.1, 1.0, 0.8)

# Initialize XGBClassifier with hyperparameters from sliders
from xgboost import XGBClassifier

my_model = XGBClassifier(
    learning_rate=Set_learning_rate,
    max_depth=Set_max_depth,
    n_estimators=Set_n_estimators,
    min_child_weight=Set_min_child_weight,
    reg_alpha=Set_reg_alpha,
    colsample_bytree=Set_colsample_bytree
)

# Button to trigger training and prediction
if st.button("Train Model and Predict"): 
    my_model.fit(X_train, y_train)
    predictions = my_model.predict(X_valid)
    output = pd.DataFrame({'PassengerId': X_valid.PassengerId,'Survived': predictions})
    output = st.data_editor(output,num_rows= "dynamic")
