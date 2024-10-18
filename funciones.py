import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

def agrega_caracteristicas(df_imputado):
    df_features = df_imputado.copy()
    
    df_features['RATIO_HOSPITALIZADOS'] = (
        df_imputado['HOSPITALIZADO']
    .divide(df_imputado['CANTIDAD'])
    .fillna(0)
    )
    
    df_features['RATIO_AISLAMIENTO'] = (
    df_imputado['AISLAMIENTO_DOMICILIARIO']
    .divide(df_imputado['CANTIDAD'])
    .fillna(0)
    )
    
    df_features['SEVERIDAD'] = (
    (df_imputado['UCI'] + df_imputado['FALLECIDO'])
    .divide(df_imputado['CANTIDAD'])
    .fillna(0)
    )
    return df_features

def prepara_el_dataset(df, var_dependiente, vars_independientes , test_size = 0.2):
    # vars_independientes es una lista
    X = df[vars_independientes]
    y = df[var_dependiente]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=32,
    )
    
    return X_train, X_test, y_train, y_test

def validacion_cruzada(X_train, y_train, modelo, cv=5, scaling=False, **kwargs):
        
    if scaling:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)

    if kwargs:
        modelo.set_params(**kwargs)

    modelo.fit(X_train, y_train)

    y_train_pred = modelo.predict(X_train)

    training_mse = round(float(mean_squared_error(y_train, y_train_pred)), 3)

    training_r2 = round(float(r2_score(y_train, y_train_pred)), 3)

    mse_scores = cross_val_score(
        modelo, 
        X_train, 
        y_train, 
        cv=cv, 
        scoring='neg_mean_squared_error'
    )

    r2_scores = cross_val_score(
        modelo, 
        X_train, 
        y_train, 
        cv=cv, 
        scoring='r2'
    )

    mse_scores = -mse_scores

    average_mse = round(float(np.mean(mse_scores)),3)
    std_mse = round(float(np.std(mse_scores)),3)

    average_r2 = round(float(np.mean(r2_scores)), 3)
    std_r2 = round(float(np.std(r2_scores)), 3)

    resultados = {
    'MSE train': training_mse,
    'MSE avg CV': average_mse,
    'Std CV MSE': std_mse,
    'R2 train': training_r2,
    'R2 avg CV': average_r2,
    }
    return resultados

def regresor(X_train, X_test, y_train, y_test, modelo, scaling=False, **kwargs):
    
    if scaling:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    if kwargs:
        modelo.set_params(**kwargs)

    modelo.fit(X_train, y_train)
    
    y_pred = modelo.predict(X_test)
    
    mse = round(float(mean_squared_error(y_test, y_pred)), 3)
    r2 = round(r2_score(y_test, y_pred), 3)
    rmse = round(float(np.sqrt(mse)), 3)
    mae = round(float(mean_absolute_error(y_test, y_pred)), 3)
    
    results = {
        'Modelo': modelo.__class__.__name__,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': abs(r2)
    }
    return results