# TRAINING PHASE 2 WITH 2 DATASETS AFTER FILTER 2 + 3
# TRAIN AND TEST PHASE 2 WITH BEST SCORE MAE, MSE, RMSE
# Import library
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split


# Reading 2 dataset from drive after split from DIEMLVCHUANHOA1.csv
# 1. Train data 2007 - 2017
dataset = pd.read_csv('TRAIN20072017.csv')

# 2. Test data 2018 - 2020 to test model after training from #1
dataset_test = pd.read_csv('TEST20182020.csv')

# Display 20 rows from dataset #1 Train data 2007 - 2017
print(dataset.head(20))

# Encode dataset #1 from string with OrdinalEncoder()
oe = OrdinalEncoder()
dataset["F_MASV"] = oe.fit_transform(dataset[["F_MASV"]])
dataset["F_MAMH"] = oe.fit_transform(dataset[["F_MAMH"]])

# Assign property column 'F_MASV','F_MAMH','NHHK' to X from #1
X = dataset.iloc[:, 0:3].values

# Assign label column F_DIEM4 to y from #1
y = dataset.iloc[:, 3].values

print("Property 'F_MASV','F_MAMH','NHHK' from #1 after transform : ")
print(X)
print("\n")
print("Label: ")
print(y)

# Build model with custom configuration
# 3 test 7 train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# Config params: n_estimators = 80, random_state = 110,min_samples_leaf = 30, max_depth = 30
regressor = RandomForestRegressor(
    n_estimators=80, random_state=110, min_samples_leaf=30, max_depth=30)

# Train model
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print('Predict result after train with dataset #1: ')
print(y_pred)

print('Display 20 rows from dataset #2 Test data 2018 - 2020')
print(dataset_test.head(20))

# Encode dataset #2 from string with OrdinalEncoder()
dataset_test["F_MASV"] = oe.fit_transform(dataset_test[["F_MASV"]])
dataset_test["F_MAMH"] = oe.fit_transform(dataset_test[["F_MAMH"]])

# Assign property column 'F_MASV','F_MAMH','NHHK' to data_train from #2
data_train = dataset_test.iloc[:, 0:3].values

# Assign label column F_DIEM4 to data_test from #2
data_test = dataset_test.iloc[:, 3].values

print("Property 'F_MASV','F_MAMH','NHHK' from #2 after transform : : ")
print(data_train)

print("Label: ")
print(data_test)

# Predict set label for dataset #2 with model after training
pred_2 = regressor.predict(data_train)

print('Predict result for #2 with model from #1: ')
print(pred_2)

# Display MAE,MSE, RMSE

print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))

print('Mean Squared Error:', mean_squared_error(y_test, y_pred))

print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
