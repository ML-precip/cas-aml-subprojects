# general imports
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
# ML imports
from utils_ml import *
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import scale
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def main_reg(df_all, df_PCs, yy_train, yy_test):
    """Function to prepare the input data for the models"""
    df_input = pd.merge(df_all, df_PCs)
    i_target= df_input.columns.str.contains('reg_aggre')
    n=df_input.columns[i_target]
    df_input = df_input.rename({n[0]: 'Y'},axis='columns')
    names_col = df_input.columns
    # define label and attributes
    ylabel = 'Y'
    attributes = names_col.drop(['date','Y','MSL'])
    # Split the data
    train_dataset, train_labels, test_dataset, test_labels, train_dates, test_dates = split_data(df_input, yy_train, yy_test, attributes, ylabel)
    fpipeline = prepareData(train_dataset, None)
    X_prep_train = fpipeline.fit_transform(train_dataset)
    X_prep_test = fpipeline.fit_transform(test_dataset)
    
    return(X_prep_train, X_prep_test, train_labels, test_labels, train_dates, test_dates)


def lnmodel(X_prep_train, X_prep_test, train_labels, test_labels):
    
    lr = LinearRegression(n_jobs=16)
    lr.fit(X_prep_train, train_labels)
    mse_train = mean_squared_error(train_labels, lr.predict(X_prep_train))
    mse_test = mean_squared_error(test_labels, lr.predict(X_prep_test))
    print(f'Train MSE = {mse_train}'); print(f'Test MSE = {mse_test}')
    print(f'Train RMSE = {np.sqrt(mse_train)}'); print(f'Test RMSE = {np.sqrt(mse_test)}')

    # get the coefficients
    lr.coef_
    #coeff_df = pd.DataFrame(lr.coef_, attributes, columns=['Coefficient'])
    # makes some predictions
    y_pred = lr.predict(X_prep_test)
    #print('Comparing predictions')
    #plot_prediction_ts(test_dates, y_pred, test_labels)
    
    return(lr, mse_train, mse_test, y_pred)





def rfmodel(X_prep_train, X_prep_test, train_labels, test_labels):
    forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    forest_reg.fit(X_prep_train, train_labels)
    # make predictions
    y_rf_pred = forest_reg.predict(X_prep_test)
    mse_rf_train = mean_squared_error(train_labels, forest_reg.predict(X_prep_train))
    mse_rf_test = mean_squared_error(test_labels, forest_reg.predict(X_prep_test))
    print(f'Train MSE = {mse_rf_train}'); print(f'Test MSE = {mse_rf_test}')
    print(f'Train RMSE = {np.sqrt(mse_rf_train)}'); print(f'Test RMSE = {np.sqrt(mse_rf_test)}')
    
    param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
      ]
    # Create the parameter grid based on the results of random search 
    
    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
    grid_search.fit(X_prep_train, train_labels)
    best_params = grid_search.best_params_
    forest_GCV_reg = RandomForestRegressor(n_jobs=-1).set_params(**best_params)
    forest_GCV_reg.fit(X_prep_train,train_labels)
    y_rf_cv_predict = forest_GCV_reg.predict(X_prep_test)
    mse_rf_cv_train = mean_squared_error(train_labels, forest_GCV_reg.predict(X_prep_train))
    mse_rf_cv_test = mean_squared_error(test_labels, forest_GCV_reg.predict(X_prep_test))
    print(f'Train MSE = {mse_rf_cv_train}'); print(f'Test MSE = {mse_rf_cv_test}')
    print(f'Train RMSE = {np.sqrt(mse_rf_cv_train)}'); print(f'Test RMSE = {np.sqrt(mse_rf_cv_test)}')
    #plot_prediction_ts(test_dates, y_rf_cv_predict, test_labels)
    
    features_importance = forest_GCV_reg.feature_importances_
    
    #sorted_features_importance = sorted(zip(features_importance, attributes), reverse=True)
    
    #plot_importance(features_importance,attributes, IMAGES_PATH)
    
    return(forest_GCV_reg, mse_rf_cv_train, mse_rf_cv_test,y_rf_cv_predict, features_importance)