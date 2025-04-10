import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import kagglehub


def get_game_level_data(game_train: pd.DataFrame, game_test: pd.DataFrame):
    """
    Returns a single feature (first-half yardage difference) and a single label (which team 
    has more yardage in the second half).
    Team 0 is the 'home' side in create_team0_yardage, so if team0_yards is positive, team0 is outgaining team1.
    
    Returns:
        X_value (float): yardage difference in the first half (Team 0's total - Team 1's total).
        y_label (int): 1 if Team 0 outgains Team 1 in the second half, else 0.
    """
    # First half yardage difference
    # 'team0_yards' is positive if team0 gains yardage, negative if team1 does.
    first_half_total = game_train['team0_yards'].sum()
    
    # Second half yardage difference
    second_half_total = game_test['team0_yards'].sum()
    
    X_value = first_half_total  # yardage difference from the first half
    y_label = 1 if second_half_total > 0 else 0
    return X_value, y_label

def logistic_regression_game_level(feature_list, label_list):
    """
    Fits a logistic regression using the first-half yardage difference as predictor
    and second-half outgain as the binary target.
    """
    # Convert lists to arrays
    X = np.array(feature_list).reshape(-1,1)  # shape (n_games, 1)
    y = np.array(label_list)                  # shape (n_games,)

    model = LogisticRegression()
    model.fit(X, y)

    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    print(f'Logistic Regression (game-level) Accuracy: {acc:.2%}')
    
    return model

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

def time_series_baseline(game_train, game_test, lags=5):
    """
    A time-series baseline that:
    1) Fits an AutoReg model to the first-half yardage for Team 0.
    2) Forecasts the same number of plays in the second half.
    3) Sums predicted yardage and compares it to the actual total.
    4) Additionally computes an MSE over the second half plays.

    Returns a dictionary with:
        - 'predicted_total': float, predicted total yardage in second half
        - 'actual_total': float, actual total yardage in second half
        - 'sign_correct': bool, whether sign of predicted total matches the actual total
        - 'mse': float, mean squared error across second-half plays
    """
    train_series = game_train["team0_yards"].values
    test_series = game_test["team0_yards"].values

    # Edge case: Not enough points to fit an AR model with 'lags'
    if len(train_series) < lags:
        raise ValueError("Not enough data points for the specified number of lags.")

    # Fit an AutoReg model to the first-half yardage
    model = AutoReg(train_series, lags=lags).fit()

    # Forecast for the same number of second-half plays
    num_test_plays = len(test_series)
    preds = model.predict(start=len(train_series),
                          end=len(train_series) + num_test_plays - 1)

    # Compute predicted and actual totals
    predicted_total = np.sum(preds)
    actual_total = np.sum(test_series)

    # Compare signs
    sign_correct = ((predicted_total > 0 and actual_total > 0) or
                    (predicted_total <= 0 and actual_total <= 0))

    # Compute MSE
    mse = np.mean((preds - test_series)**2)

    return {
        "predicted_total": predicted_total,
        "actual_total": actual_total,
        "sign_correct": sign_correct,
        "mse": mse
    }

def simple_baseline(game_train, game_test):
    """
    Train a model that assumes that the 1st and second half are similar
    to predict whether the offensive team gains positive yardage.
    """
    assumed_total_yards = 2*np.sum(game_train)

    true_total_yards = np.sum(game_train)+np.sum(game_test)

    if (assumed_total_yards > 0 and true_total_yards > 0) or (assumed_total_yards <= 0 and true_total_yards <= 0):
        return 1
    return 0