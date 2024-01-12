import os
import logging
import warnings
import contextlib
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import SimpleExpSmoothing,ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import pmdarima as pm
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from scipy import stats
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import grid_search_forecaster
from sklearn.ensemble import HistGradientBoostingRegressor
import time
from datetime import datetime

# from main import read_data

def create_output_directory(output_dir):
    # Ensure the base directory exists
        if not os.path.exists(output_dir):
            output_dir=os.makedirs(output_dir)
            print(f"Directory '{output_dir}' created.")
        else:
            print(f"Directory '{output_dir}' already exists.")
        return output_dir

def train_test_split(data,percent_split):
    unique_ids = data['unique_id'].unique()
    # Initialize empty train and test DataFrames
    train = pd.DataFrame()
    test= pd.DataFrame()
    # Split data for each unique ID
    for id_value in unique_ids:
        # Select data for the current ID
        id_data = data[data['unique_id'] == id_value]
        len_id_data=int(len(id_data)*(percent_split/100))
        train_id = id_data.iloc[:len(id_data)-len_id_data]
        test_id = id_data.iloc[len(id_data)-len_id_data:]
        # Append the split data to the respective DataFrames
        train = pd.concat([train,train_id])
        test = pd.concat([test,test_id])
    train=train.reset_index(drop=True) 
    test=test.reset_index(drop=True)
    return train,test

def check_intermittency(df,col):
    adi_threshold=1.32
    cov_threshold=0.49
    non_zeroes=(df[col]!=0).sum()
    total_periods=len(df)
    ADI=total_periods/non_zeroes
    #display(ADI)
    label=''
     
    COV= (np.std(df[col])/np.mean(df[col]))**2
    #display(COV)
    
    if ADI < adi_threshold and COV < cov_threshold:
        label='Smooth'
    if ADI >= adi_threshold and COV < cov_threshold:
        label='Intermittent'
    if ADI < adi_threshold and COV >= cov_threshold:
        label='Erratic'
    if ADI >= adi_threshold and COV >= cov_threshold:
        label='Lumpy'
    return label
    
def intermittency_dict_label(check_df,m_id):
    label=check_intermittency(check_df,m_id)
    return label

# needs to be revised later for pipeline sections
def full_train_test(train,test,intermittency_dict):
    test_check,train_check=pd.DataFrame(),pd.DataFrame()
    test_df=test.copy()
    for intermittent_k,intermittent_v in intermittency_dict.items():
        train['Intermittency_Type']=str(intermittent_k)
        test_df['Intermittency_Type']=str(intermittent_k)
        train_check=pd.concat([train_check,train],axis=0)
        test_check=pd.concat([test_check,test_df],axis=0)
    return test_check,train_check,test_df

# calculating prediction interval
def calculate_prediction_interval(predictions,forecast_errors):
    alpha=0.05
    z_alpha = np.abs(stats.norm.ppf(1 - alpha / 2))
    lower_bound=predictions-z_alpha*forecast_errors
    upper_bound=predictions+z_alpha*forecast_errors
    return lower_bound,upper_bound

def get_significant_lags(df, ycol, desired_lags=40):
    n = len(df[ycol])
    # Calculate the maximum allowable number of lags (50% of the sample size)
    max_lags = int(n * 0.5)
    # Specify the number of lags as the minimum between the desired value and the maximum allowable value
    nlags = min(desired_lags, max_lags)
    pacf_values = sm.tsa.pacf(df[ycol], nlags=nlags)
    significance_level = 0.05
    confidence_interval = 1.96 / np.sqrt(n)
    significant_lags = [i for i, pacf_value in enumerate(pacf_values) if abs(pacf_value) > confidence_interval]
    significant_lags_v2 = [i for i in significant_lags if i not in [0, 1]]
    # If no significant lags are found, provide default values (e.g., [4, 7])
    if len(significant_lags_v2) == 0:
        significant_lags_v2 = [4, 7]
    return significant_lags_v2


def SES(df_try,forecast_length,forecast_df):
    alpha=0.05
    model_ses=SimpleExpSmoothing(df_try)
    logging.info("Fitting Simple Exponential Smoothing model...")
    fit_ses=model_ses.fit()
    logging.info("Model fit successfully.")
    
    logging.info(f"Forecasting {forecast_length} steps ahead...")
    forecast_df['SES']=fit_ses.forecast(steps=(forecast_length)).values
    logging.info(f"Forecasting completed for {forecast_length} steps.")
    forecast_error_std = np.std(fit_ses.resid)
    lower_bound,upper_bound=calculate_prediction_interval(forecast_df['SES'],forecast_error_std)
    
    forecast_df['SES_Lower'] = lower_bound
    forecast_df['SES_Upper'] = upper_bound
    logging.info("SES function executed successfully.")
    logging.info("")  # Add an empty line
    return forecast_df

def DES(df_try,forecast_length,forecast_df):
    model_des=ExponentialSmoothing(df_try,trend='add')

    logging.info("Fitting DES model...")
    fit_des=model_des.fit()
    logging.info("Model fit successfully.")

    logging.info(f"Forecasting {forecast_length} steps ahead...")
    forecast_df['DES']=fit_des.forecast(steps=(forecast_length)).values
    logging.info(f"Forecasting completed for {forecast_length} steps.")

    forecast_error_std = np.std(fit_des.resid)
    lower_bound,upper_bound=calculate_prediction_interval(forecast_df['DES'],forecast_error_std)
    
    forecast_df['DES_Lower'] = lower_bound
    forecast_df['DES_Upper'] = upper_bound
    logging.info("DES function executed successfully.")
    logging.info("")  # Add an empty line
    return forecast_df


def TES(df_try,forecast_length,forecast_df,seasonal_period,initial_method):
    model_tes=ExponentialSmoothing(df_try,trend='add',seasonal='add',seasonal_periods=seasonal_period,initialization_method=initial_method)
    
    logging.info("Fitting TES model...")
    fit_tes=model_tes.fit()
    logging.info("Model fit successfully.")

    logging.info(f"Forecasting {forecast_length} steps ahead...")
    forecast_df['TES'] = fit_tes.forecast(steps=(forecast_length)).values
    logging.info(f"Forecasting completed for {forecast_length} steps.")
    forecast_error_std = np.std(fit_tes.resid)
    lower_bound,upper_bound=calculate_prediction_interval(forecast_df['TES'],forecast_error_std)
    
    forecast_df['TES_Lower'] = lower_bound
    forecast_df['TES_Upper'] = upper_bound
    logging.info("TES function executed successfully.")
    logging.info("")  # Add an empty line
    return forecast_df


def ARIMA_check(df_try,forecast_length,forecast_df):
    with contextlib.redirect_stdout(None):
        autoarima_model=pm.auto_arima(df_try,seasonal=False,stepwise=True,trace=True,suppress_warnings=True,error_action="ignore")
        p,d,q=autoarima_model.order
        model_arima=pm.ARIMA(order=(p,d,q),seasonal=False)

        logging.info("Fitting ARIMA model...")
        model_arima.fit(df_try)
        logging.info("Model fit successfully.")

    logging.info(f"Forecasting {forecast_length} steps ahead...")
    arima_forecast_results,conf_int_arima=model_arima.predict(n_periods=(forecast_length),return_conf_int=True)
    logging.info(f"Forecasting completed for {forecast_length} steps.")

    forecast_df['ARIMA']=arima_forecast_results.values
    forecast_df['ARIMA_Lower']=conf_int_arima[:,0]
    forecast_df['ARIMA_Upper']=conf_int_arima[:,1]
    logging.info("ARIMA function executed successfully.")
    logging.info("")  # Add an empty line
    return forecast_df

def SARIMA_check(df_try,forecast_length,forecast_df):
    auto_sarima_model=pm.auto_arima(df_try,seasonal=True,m=12,stepwise=True)
    p,d,q=auto_sarima_model.order
    P,D,Q,S=auto_sarima_model.seasonal_order
    model_sarima=pm.ARIMA(order=(p,d,q),seasonal_order=(P,D,Q,S),initialization='approximate_diffuse') #added intialization due to linalg error
    
    logging.info("Fitting SARIMA model...")
    model_sarima.fit(df_try)
    logging.info("Model fit successfully.")

    logging.info(f"Forecasting {forecast_length} steps ahead...")
    sarima_forecast_results,conf_int_sarima=model_sarima.predict(n_periods=(forecast_length),return_conf_int=True)
    logging.info(f"Forecasting completed for {forecast_length} steps.")

    forecast_df['SARIMA']=sarima_forecast_results.values
    forecast_df['SARIMA_Lower']=conf_int_sarima[:,0]#.conf_int().iloc[:,0]
    forecast_df['SARIMA_Upper']=conf_int_sarima[:,1]
    logging.info("SARIMA function executed successfully.")
    logging.info("")  # Add an empty line
    return forecast_df

def SKFORECAST_XGB(df_try,test_df_try,forecast_length,forecast_df,lags_grid):
    params_dict=dict()
    forecast_horizon=7
    initial_train_size=int(len(df_try)*0.7)
    len_test=len(test_df_try)
    with contextlib.redirect_stdout(None):
        sk_model=ForecasterAutoreg(regressor=xgb.XGBRegressor(verbose=False),lags=12)
        df_try=df_try.reset_index()
        param_grid={'regressor_n_estimators':[50,75,150,200],
            'regressor_max_depth':[1,2],
            'regressor_learning_rate':[0.1]}
        best_forecaster=grid_search_forecaster(forecaster=sk_model,   
            y                  = df_try['Values'],
            param_grid         = param_grid,
            lags_grid          = lags_grid,
            steps              = forecast_length,
            refit              = False,
            metric             = 'mean_absolute_percentage_error',
            initial_train_size = int(len(df_try)*0.7),
            fixed_train_size   = False,
            return_best        = True,
            n_jobs             = 'auto',
            verbose            = False
                                        
            )
    logging.info("Fitting SKFORECAST_XGB model...")
    sk_model.fit(df_try['Values'])
    logging.info("Model fit successfully.")

    logging.info(f"Forecasting {forecast_length} steps ahead...")
    predictions=sk_model.predict(steps=(forecast_length))
    logging.info(f"Forecasting completed for {forecast_length} steps.")

    predictions=pd.DataFrame(predictions)
    test_df_try = test_df_try.reset_index(drop=True)
    predictions = predictions.reset_index(drop=True)
    residuals=test_df_try['Values']-predictions['pred'][:len_test]
    forecast_error_std=np.std(residuals)
    lower_bound,upper_bound=calculate_prediction_interval(predictions['pred'],forecast_error_std)
    
    forecast_df['SKFORECAST_XGB']=predictions['pred']
    forecast_df['SKFORECAST_XGB_Lower']=lower_bound
    forecast_df['SKFORECAST_XGB_Upper']=upper_bound
    
    best_params=best_forecaster.iloc[0]['params']
    best_lags=best_forecaster.iloc[0]['lags']
    params_dict.update({'params':best_params,'lags':best_lags})
    
    logging.info("SKFORECAST_XGB function executed successfully.")
    logging.info("")  # Add an empty line
    return forecast_df,params_dict

def SKFORECAST_LGB(df_try,test_df_try,forecast_length,forecast_df,lags_grid):
    params_dict=dict()
    forecast_horizon=7
    initial_train_size=int(len(df_try)*0.7)
    len_test=len(test_df_try)
    with contextlib.redirect_stdout(None):
        sk_model_lgb=ForecasterAutoreg(regressor=lgb.LGBMRegressor(verbose=-1),lags=5)
        df_try=df_try.reset_index()
        param_grid={'regressor_n_estimators':[30,40,50],
            'regressor_max_depth':[4],
            'regressor_learning_rate':[0.1]}
        best_forecaster=grid_search_forecaster(forecaster=sk_model_lgb,
    
            y                  = df_try['Values'],
            param_grid         = param_grid,
            lags_grid          = lags_grid,
            steps              = forecast_length,
            refit              = False,
            metric             = 'mean_absolute_percentage_error',
            initial_train_size = int(len(df_try)*0.8),
            fixed_train_size   = False,
            return_best        = True,
            n_jobs             = 'auto',
            verbose            = False
                                        
            )
    logging.info("Fitting SKFORECAST_LGB model...")
    sk_model_lgb.fit(df_try['Values'])
    logging.info("Model fit successfully.")

    logging.info(f"Forecasting {forecast_length} steps ahead...")
    predictions=sk_model_lgb.predict(steps=(forecast_length))
    logging.info(f"Forecasting completed for {forecast_length} steps.")

    predictions=pd.DataFrame(predictions)
    test_df_try = test_df_try.reset_index(drop=True)
    predictions = predictions.reset_index(drop=True)
    residuals=test_df_try['Values']-predictions['pred'][:len_test]
    forecast_error_std=np.std(residuals)
    lower_bound,upper_bound=calculate_prediction_interval(predictions['pred'],forecast_error_std)
    forecast_df['SKFORECAST_LGB']=predictions['pred']
    forecast_df['SKFORECAST_LGB_Lower']=lower_bound
    forecast_df['SKFORECAST_LGB_Upper']=upper_bound
    best_params=best_forecaster.iloc[0]['params']
    best_lags=best_forecaster.iloc[0]['lags']
    params_dict.update({'params':best_params,'lags':best_lags})
    
    logging.info("SKFORECAST_LGB function executed successfully.")
    logging.info("")  # Add an empty line
    return forecast_df,params_dict

def SKFORECAST_Catboost(df_try,test_df_try,forecast_length,forecast_df,lags_grid):
    params_dict=dict()
    forecast_horizon=7
    initial_train_size=int(len(df_try)*0.7)
    len_test=len(test_df_try)
    with contextlib.redirect_stdout(None):
        sk_model_cb=ForecasterAutoreg(regressor=cb.CatBoostRegressor(silent=True),lags=5)
        df_try=df_try.reset_index()
        param_grid={'n_estimators':[100,400,500],
            'depth':[4,5],
            'learning_rate':[0.1]}
        best_forecaster=grid_search_forecaster(forecaster=sk_model_cb, 
            y                  = df_try['Values'],
            param_grid         = param_grid,
            lags_grid          = lags_grid,
            steps              = forecast_length,
            refit              = False,
            metric             = 'mean_absolute_percentage_error',
            initial_train_size = int(len(df_try)*0.7),
            fixed_train_size   = False,
            return_best        = True,
            n_jobs             = 'auto',
            verbose            = False
                                        
            )
        
    logging.info("Fitting SKFORECAST_Catboost model...")
    sk_model_cb.fit(df_try['Values'])
    logging.info("Model fit successfully.")

    logging.info(f"Forecasting {forecast_length} steps ahead...")
    predictions=sk_model_cb.predict(steps=(forecast_length))
    logging.info(f"Forecasting completed for {forecast_length} steps.")

    predictions=pd.DataFrame(predictions)
    test_df_try = test_df_try.reset_index(drop=True)
    predictions = predictions.reset_index(drop=True)
    residuals=test_df_try['Values']-predictions['pred'][:len_test]
    forecast_error_std=np.std(residuals)
    lower_bound,upper_bound=calculate_prediction_interval(predictions['pred'],forecast_error_std)
    forecast_df['SKFORECAST_CATBOOST']=predictions.values
    forecast_df['SKFORECAST_CATBOOST_Lower']=lower_bound
    forecast_df['SKFORECAST_CATBOOST_Upper']=upper_bound
    best_params=best_forecaster.iloc[0]['params']
    best_lags=best_forecaster.iloc[0]['lags']
    params_dict.update({'params':best_params,'lags':best_lags})

    logging.info("SKFORECAST_Catboost function executed successfully.")
    logging.info("")  # Add an empty line
    return forecast_df,params_dict

def SKFORECAST_HistGradboost(df_try,test_df_try,forecast_length,forecast_df,lags_grid):
    params_dict=dict()
    forecast_horizon=7
    initial_train_size=int(len(df_try)*0.7)
    len_test=len(test_df_try)
    with contextlib.redirect_stdout(None):
        sk_model_hgb=ForecasterAutoreg(regressor=HistGradientBoostingRegressor(verbose=0),lags=5)
        df_try=df_try.reset_index()
        #sk_model.fit(df_try)
        param_grid={'max_iter':[100,200,400],
            'max_depth':[3,4],
            'learning_rate':[0.1]}
        best_forecaster=grid_search_forecaster(forecaster=sk_model_hgb,   
            y                  = df_try['Values'],
            param_grid         = param_grid,
            lags_grid          = lags_grid,
            steps              = forecast_length,
            refit              = False,
            metric             = 'mean_absolute_percentage_error',
            initial_train_size = int(len(df_try)*0.7),
            fixed_train_size   = False,
            return_best        = True,
            n_jobs             = 'auto',
            verbose            = False
                                        
            )
        
    logging.info("Fitting SKFORECAST_HistGradboost model...")
    sk_model_hgb.fit(df_try['Values'])
    logging.info("Model fit successfully.")
    
    logging.info(f"Forecasting {forecast_length} steps ahead...")
    predictions=sk_model_hgb.predict(steps=(forecast_length))
    logging.info(f"Forecasting completed for {forecast_length} steps.")

    
    predictions=pd.DataFrame(predictions)
    test_df_try = test_df_try.reset_index(drop=True)
    predictions = predictions.reset_index(drop=True)
    residuals=test_df_try['Values']-predictions['pred'][:len_test]
    forecast_error_std=np.std(residuals)
    lower_bound,upper_bound=calculate_prediction_interval(predictions['pred'],forecast_error_std)
    forecast_df['SKFORECAST_HISTGRADBOOST']=predictions['pred']
    forecast_df['SKFORECAST_HISTGRADBOOST_Lower']=lower_bound
    forecast_df['SKFORECAST_HISTGRADBOOST_Upper']=upper_bound
    best_params=best_forecaster.iloc[0]['params']
    best_lags=best_forecaster.iloc[0]['lags']
    params_dict.update({'params':best_params,'lags':best_lags})

    logging.info("SKFORECAST_HistGradboost function executed successfully.")
    logging.info("")  # Add an empty line
    return forecast_df,params_dict

def Hybrid_DES_SKFORECASTXGB(df_try,test_df_try,forecast_length,forecast_df,master_forecast_df):
    model_1=['DES','TES','ARIMA','SARIMA']
    model_2=['SKFORECAST_XGB','SKFORECAST_LGB','SKFORECAST_CATBOOST']#'SKFORECAST_HISTGRADBOOST']
    weight_des=0.5
    weight_xgb=0.5
    len_test=len(test_df_try)
    for m1 in model_1:
        for m2 in model_2:
            forecast_df['HYBRID_'+m1+'_'+m2]=(weight_des*master_forecast_df[m1].values)+(weight_xgb*forecast_df[m2].values)
            predictions=forecast_df['HYBRID_'+m1+'_'+m2]
            predictions=pd.DataFrame(predictions)
            test_df_try = test_df_try.reset_index(drop=True)
            predictions = predictions.reset_index(drop=True)
            residuals=test_df_try['Values']-predictions['HYBRID_'+m1+'_'+m2][:len_test]
            forecast_error_std=np.std(residuals)
            lower_bound,upper_bound=calculate_prediction_interval(predictions['HYBRID_'+m1+'_'+m2],forecast_error_std)
            forecast_df['HYBRID_'+m1+'_'+m2+'_Lower']=lower_bound
            forecast_df['HYBRID_'+m1+'_'+m2+'_Upper']=upper_bound
            logging.info("Hybrid_DES_SKFORECASTXGB function executed successfully.")
            logging.info("")  # Add an empty line
    return forecast_df

def models_pipeline1(df_try,forecast_length,forecast_df,seasonal_period,initial_method):
    st_ses=time.time()
    forecast_df=SES(df_try,forecast_length,forecast_df)
    end_ses=time.time()
    print("SES time",end_ses-st_ses)
    st_des=time.time()
    forecast_df=DES(df_try,forecast_length,forecast_df)
    end_des=time.time()
    print("DES time",end_des-st_des)
    st_tes=time.time()
    forecast_df=TES(df_try,forecast_length,forecast_df,seasonal_period,initial_method)
    end_tes=time.time()
    print("TES time",end_tes-st_tes)
    st_arima=time.time()
    forecast_df=ARIMA_check(df_try,forecast_length,forecast_df)
    end_arima=time.time()
    print("ARIMA time",end_arima-st_arima)
    st_sarima=time.time()
    forecast_df=SARIMA_check(df_try,forecast_length,forecast_df)
    end_sarima=time.time()
    print("SARIMA time",end_sarima-st_sarima)
    return forecast_df

def pipeline1_forecast (train_check,test_check,fin_id, no_months_forecast,seasonal_period,initial_method):
    df_try=train_check[train_check.unique_id==fin_id].copy()
    reg_df_try=df_try.copy()
    df_try['Values']=df_try['Values'].astype(float)
    test_df_try=test_check[test_check.unique_id==fin_id]

    reg_test_df_try=test_df_try.copy()
    forecast_df=pd.DataFrame()
    forecast_length=len(test_df_try)+no_months_forecast
    forecast_df=models_pipeline1(df_try['Values'],forecast_length,forecast_df,seasonal_period,initial_method)
    # print('forecast_df',forecast_df)
    forecast_df['Intermittency_check']=test_df_try['Intermittency_Type'].values[-1]
    forecast_df['unique_id']=fin_id
    forecast_df['Actual'] = np.nan  # Initialize the target column with NaN
    n=test_df_try.shape[0]
    
    forecast_df.loc[:n, 'Actual'] = test_df_try['Values'].reset_index(drop=True)
    
    df=test_df_try[['Date']]
    last_date = test_df_try['Date'].iloc[-1]
    extended_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=no_months_forecast, freq='MS')
    extended_df = pd.DataFrame({'Date': extended_dates})
    result_df = pd.concat([df, extended_df], ignore_index=True)
    forecast_df['Date']=result_df['Date']
    # print('result_df',result_df)
    
    return forecast_df

def models_pipeline2(df_try,test_df_try,forecast_length,forecast_df,masterfrcst1_temp,lags_grid):
    st=time.time()
    forecast_df,xgb_bestparams=SKFORECAST_XGB(df_try,test_df_try,forecast_length,forecast_df,lags_grid)
    end=time.time()
    print("SKFORECAST_XGB time",end-st)
    
    st=time.time()
    forecast_df,lgb_bestparams =SKFORECAST_LGB(df_try,test_df_try,forecast_length,forecast_df,lags_grid)
    end=time.time()
    print("SKFORECAST_LGB time",end-st)
    
    st=time.time()
    forecast_df,catboost_bestparams=SKFORECAST_Catboost(df_try,test_df_try,forecast_length,forecast_df,lags_grid)
    end=time.time()
    print("SKFORECAST_Catboost time",end-st)
    
    st=time.time()
    forecast_df,histgradboost_bestparams=SKFORECAST_HistGradboost(df_try,test_df_try,forecast_length,forecast_df,lags_grid)
    end=time.time()
    print("SKFORECAST_HistGradboost time",end-st)

    st=time.time()
    forecast_df=Hybrid_DES_SKFORECASTXGB(df_try,test_df_try,forecast_length,forecast_df,masterfrcst1_temp)
    end=time.time()
    print("Hybrid_DES_SKFORECASTXGB time",end-st)

    return forecast_df,xgb_bestparams,lgb_bestparams,catboost_bestparams,histgradboost_bestparams


def pipeline2_forecast(train_check,test_check,master_forecast_df,fin_id,params_struct,no_months_forecast):
    df_try=train_check[train_check.unique_id==fin_id]
    reg_df_try=df_try.copy()
    df_try['Values']=df_try['Values'].astype(float)
    test_df_try=test_check[test_check.unique_id==fin_id]
    forecast_df=pd.DataFrame()
    forecast_length=len(test_df_try)+no_months_forecast
    reg_df_try=reg_df_try[['Date','Values']]
    masterfrcst1_temp=master_forecast_df[master_forecast_df.unique_id==fin_id].copy()
    lags_grid=get_significant_lags(df_try,'Values')
    forecast_df,xgb_bestparams,lgb_bestparams,catboost_bestparams,histgradboost_bestparams=models_pipeline2(df_try['Values'],test_df_try,forecast_length,forecast_df,masterfrcst1_temp,lags_grid)
    
    params_struct=params_struct._append({'unique_id':fin_id,'XGB':xgb_bestparams,'LGB':lgb_bestparams,'CATBOOST':catboost_bestparams,'HISTGRADBOOST':histgradboost_bestparams},ignore_index=True)
    forecast_df['Intermittency_check']=test_df_try['Intermittency_Type'].values[-1]
    forecast_df['unique_id']=fin_id
    forecast_df['Actual'] = np.nan  # Initialize the target column with NaN
    n=test_df_try.shape[0]
    
    forecast_df.loc[:n, 'Actual'] = test_df_try['Values'].reset_index(drop=True)
    df=test_df_try[['Date']]
    last_date = test_df_try['Date'].iloc[-1]
    extended_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=no_months_forecast, freq='MS')
    extended_df = pd.DataFrame({'Date': extended_dates})
    result_df = pd.concat([df, extended_df], ignore_index=True)
    forecast_df['Date']=result_df['Date']
    return forecast_df

def metrics_evaluation(forecast_df):
    models=['SES','DES','TES','ARIMA','SARIMA','SKFORECAST_HISTGRADBOOST',
                       'SKFORECAST_CATBOOST','SKFORECAST_LGB','SKFORECAST_XGB','HYBRID_DES_SKFORECAST_XGB','HYBRID_DES_SKFORECAST_LGB','HYBRID_DES_SKFORECAST_CATBOOST',#,'HYBRID_DES_SKFORECAST_HISTGRADBOOST',
'HYBRID_TES_SKFORECAST_XGB','HYBRID_TES_SKFORECAST_LGB','HYBRID_TES_SKFORECAST_CATBOOST',#,'HYBRID_TES_SKFORECAST_HISTGRADBOOST',
'HYBRID_ARIMA_SKFORECAST_XGB','HYBRID_ARIMA_SKFORECAST_LGB','HYBRID_ARIMA_SKFORECAST_CATBOOST',#,'HYBRID_ARIMA_SKFORECAST_HISTGRADBOOST',
'HYBRID_SARIMA_SKFORECAST_XGB','HYBRID_SARIMA_SKFORECAST_LGB','HYBRID_SARIMA_SKFORECAST_CATBOOST']#,'HYBRID_SARIMA_SKFORECAST_HISTGRADBOOST']#,'Auto_ML']
    error_metric_dataframe=pd.DataFrame(columns=['Model','MSE','MAE','RMSE','MAPE'])
    best_mae=float('inf')
    best_mse=float('inf')
    best_rmse=float('inf')
    best_mape=float('inf')
    best_model='XYZ'
    for model in models:
        forecast_df[model+'_Error'] = forecast_df['Actual'] - forecast_df[model]
        forecast_df[model+'_Absolute_Error'] = np.abs(forecast_df[model+'_Error'])
        mae = forecast_df[model+'_Absolute_Error'].mean()
        mse = (forecast_df[model+'_Error'] ** 2).mean()
        rmse = np.sqrt(mse)
        mape = (forecast_df[model+'_Absolute_Error'] / forecast_df['Actual']).mean() * 100
        #if mae < best_mae and mse < best_mse and rmse < best_rmse and mape <best_mape:
        if mape <best_mape:
            # best_mae=mae
            # best_mse=mse
            # best_rmse=rmse
            best_mape=mape
            best_model=model
        new_df=pd.DataFrame({'Model':model,'MSE':mse,'MAE':mae,'RMSE':rmse,'MAPE':mape},index=[0])
        error_metric_dataframe=pd.concat([error_metric_dataframe,new_df],axis=0)
    return error_metric_dataframe,best_mae,best_mse,best_rmse,best_mape,best_model


def train_predict(data,no_months_forecast):
    # creating output dir
    # st=time.time()
    # output_dir = r'C:\Users\Amit Kumar\OneDrive - ORMAE\Desktop\azure_webapp\output_genric'
    # output_dir = create_output_directory(output_dir)
    # end=time.time()
    # print("output_dir_time",end-st)

    st=time.time()
    required_col_name=["unique_id", "Date", "Values"]
    data.columns=required_col_name
    ###if data is given in weekly form starting---
    data['Date']=pd.to_datetime(data['Date'])
    data['month_year']=pd.to_datetime(data['Date'].dt.year.astype(str) + "-"+data['Date'].dt.month.astype(str)+"-01")
    data_1=data.groupby(['unique_id','month_year'])['Values'].sum().reset_index()
    data_1.rename(columns={"month_year":"Date"},inplace=True)
    data_2=data_1.copy()
    data_2['unique_id'] = data_2['unique_id'].replace('14265_1235', '14265_1236')
    data=pd.concat([data_1,data_2])
    data=data.reset_index(drop=True)
    ###ending--- till here for data conversion,do not use till this part if you have monthly data
     
    # splitting train and test data
    percent_split=10
    train,test=train_test_split(data,percent_split)
    print(len(train),len(test))
    end=time.time()
    print("data_preprocessing_time",end-st)

    # creating intermittency_dict
    st=time.time()
    intermittency_dict={}
    for m_id in train.unique_id:
        check_df=train[train.unique_id==m_id]
        check_df=check_df.pivot(columns='unique_id',values='Values')
        label=intermittency_dict_label(check_df,m_id)
        if label in intermittency_dict:
            if m_id not in intermittency_dict[label]:
                intermittency_dict[label].append(m_id)
            else:
                continue
        else:
            intermittency_dict[label] = [m_id]
    end=time.time()
    print("intermittency_dict_label_time",end-st)

    # creating intermittency_list
    intermittency_list=[]
    for label,v in intermittency_dict.items():
        intermittency_dict.update({label:v[:]})
        for item in v[:]:
            intermittency_list.append(item)

    st=time.time()
    test_check,train_check,test_df=full_train_test(train,test,intermittency_dict)
    end=time.time()
    print("full_train_test_time",end-st)


    #converting date into datetime object
    train_check['Date']=pd.to_datetime(train_check['Date'])
    test_check['Date']=pd.to_datetime(test_check['Date'])

    # for running pipeline 1
    st=time.time()
    seasonal_period=12
    initial_method='heuristic'
    # user_input = input("months for you want to forecast: ")
    # no_months_forecast=int(user_input)
    train_check['Values']=train_check['Values'].astype(float)
    model_implemented=['SES','DES','TES']

    master_forecast_df=pd.DataFrame()
    for fin_id in list(train_check['unique_id'].unique()):
        forecast_df=pipeline1_forecast (train_check,test_check,fin_id,no_months_forecast,seasonal_period,initial_method)
        master_forecast_df=pd.concat([master_forecast_df,forecast_df],axis=0)
    master_forecast_df.to_csv(os.path.join(output_dir,'forecast_results'+'.csv'),index=False)
    end=time.time()
    print("pipeline 1 running time",end-st)


    # for running pipeline 2
    st=time.time()
    master_forecast_df_2=pd.DataFrame()
    params_struct=pd.DataFrame(columns=['unique_id','XGB','LGB','CATBOOST','HISTGRADBOOST'])
    start_time=time.time()
    # csv_filename='forecast_results2.csv'

    for fin_id in list(train_check['unique_id'].unique()):
        forecast_df=pipeline2_forecast(train_check,test_check,master_forecast_df,fin_id,params_struct,no_months_forecast)
        master_forecast_df_2=pd.concat([master_forecast_df_2,forecast_df],axis=0)
    master_forecast_df_2.to_csv(os.path.join(output_dir,'forecast_results2'+'.csv'),index=False)  
    end=time.time()
    print("pipeline 2 running time",end-st)
    

    # combining pipeline_1 and pipeline_2 results
    master_forecast_df_fin=pd.concat([master_forecast_df,master_forecast_df_2],axis=1)
    
    # to remove the duplicate values from the dataframe
    master_forecast_df_fin = master_forecast_df_fin.loc[:, ~master_forecast_df_fin.columns.duplicated()]

    
    # for calculating the metric evaluation and best models
    final_error_metric_df=pd.DataFrame(columns=['unique_id','Intermittency_Type','Best_Model_Evaluated','MAPE'])
    for fin_id in list(train_check['unique_id'].unique()):
        forecast_df=master_forecast_df_fin[master_forecast_df_fin['unique_id']==fin_id]
        # print("forecast_df",forecast_df)
        test_len_unique_id=len(test)//(test['unique_id'].nunique())
        error_metric_dataframe,best_mae,best_mse,best_rmse,best_mape,best_model=metrics_evaluation(forecast_df.iloc[:test_len_unique_id])
        new_df=pd.DataFrame({'unique_id':fin_id,'Intermittency_Type':str(forecast_df['Intermittency_check'][0]),'Best_Model_Evaluated':best_model,'MAPE':best_mape},index=[0])
        final_error_metric_df=pd.concat([final_error_metric_df,new_df],axis=0)

    final_error_metric_df.to_csv(os.path.join(output_dir,'MAPE_values'+'.csv'),index=False)


    final_ormaefit_output=pd.DataFrame()
    for fin_id in train_check.unique_id.unique():
        mid_frame=pd.DataFrame()
        forecast_df=master_forecast_df_fin[master_forecast_df_fin['unique_id']==fin_id]
        best_model=final_error_metric_df[final_error_metric_df.unique_id==fin_id]['Best_Model_Evaluated'].values[0]
        mid_frame['Forecast']=forecast_df[best_model]
        mid_frame['Forecast_Upper']=forecast_df[best_model+'_Upper']
        mid_frame['Forecast_Lower']=forecast_df[best_model+'_Lower']
        mid_frame['Actual']=forecast_df['Actual']
        mid_frame['unique_id']=fin_id
        mid_frame['Intermittency_Type']=forecast_df['Intermittency_check']
        mid_frame['Best Model']=best_model
        mid_frame['Date']=forecast_df['Date']
        final_ormaefit_output=pd.concat([final_ormaefit_output,mid_frame],axis=0)
        final_ormaefit_output=final_ormaefit_output[['unique_id','Intermittency_Type','Best Model','Date','Forecast','Actual','Forecast_Upper','Forecast_Lower']]
    
    forcast_result = final_ormaefit_output[final_ormaefit_output['Actual'].isnull()]
    forcast_result = forcast_result[['Date','unique_id','Intermittency_Type','Best Model','Forecast','Forecast_Lower','Forecast_Upper']]
    return forcast_result

# print(train_predict())










