#!/usr/bin/env python
# coding: utf-8

# imports
import numpy as np
import pandas as pd
from itertools import product
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error
from sklearn.base import clone

# lese inn data
data = pd.read_csv('data/huspris.csv')


# velge ut features
numeric_features = ['LotFrontage', 'LotArea', 'OverallQual', 'YrSold']
categorical_features = ['Street', 'HouseStyle', 'BsmtQual', 'GarageCond']

# del data i X og y
X = data.loc[:, numeric_features + categorical_features]
y = data.SalePrice

# dele data i trenings, validerings og testdata
X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=42)

# imputers
imputers = {
    'median': SimpleImputer(strategy='median'), 
    'knn': KNNImputer(n_neighbors=10)
}

# modeller
models = {'lr': LinearRegression(),
          'lasso': Lasso(alpha=1),
          'rf': RandomForestRegressor(), 
          'svm': SVR()}   

# kombiner
pipes = {imputer_key + '_' + regressor_key: Pipeline(
    steps=[('preprocess', ColumnTransformer(transformers=[
        ('num', Pipeline(steps = [('impute', clone(imputer)), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline(steps = [('ohe', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)])),
           ('regress', clone(regressor))])
         for (imputer_key, imputer), (regressor_key, regressor) in product(imputers.items(), models.items())}


# modelutvalg
validation_rmse = pd.DataFrame(
    {key: root_mean_squared_error(y_val, pipe.fit(X_train, y_train).predict(X_val))
     for key, pipe in pipes.items()}.items(),
    columns=['model', 'rmse'])
best_model = pipes[validation_rmse.loc[np.argmin(validation_rmse.rmse), 'model']]

# modell og generaliseringsfeil
print('Test RMSE:', root_mean_squared_error(y_test, best_model.predict(X_test)))
print(best_model)

# lagre model
pickle.dump(best_model, open('model.pkl', 'wb'))
