import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from statsmodels.tsa.ar_model import AutoReg
import numpy as np

def logistic_regression_baseline(game_train: pd.DataFrame, game_test: pd.DataFrame):
    """
    Train a logistic regression model to predict whether the offensive team gains positive yardage.
    """

    # Define binary target: positive yardage (1) vs. non-positive yardage (0)
    y_train = (game_train['team0_yards'] > 0).astype(int)
    y_test = (game_test['team0_yards'] > 0).astype(int)

    # Simple features: play time, offensive team
    X_train = game_train[['play_time', 'posteam']]
    X_test = game_test[['play_time', 'posteam']]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    print(f"Logistic Regression Accuracy: {accuracy:.2%}")
    return model, accuracy

def time_series_baseline(game_train: pd.DataFrame, game_test: pd.DataFrame, lags=5):
    """
    Elementary time series model (autoregressive) predicting yards gained on the next play.
    """
    
    # Fit AutoReg model to yardage
    train_series = game_train['team0_yards'].values

    if len(train_series) < lags:
        raise ValueError("Not enough data points for the specified number of lags.")

    ar_model = AutoReg(train_series, lags=lags).fit()

    # Forecast for length of test data
    preds = ar_model.predict(start=len(train_series), end=len(train_series) + len(game_test) - 1)

    # Evaluate with Mean Squared Error
    mse = np.mean((preds - game_test['team0_yards'].values)**2)

    print(f"Time Series (AR) MSE: {mse:.4f}")
    return ar_model, mse
