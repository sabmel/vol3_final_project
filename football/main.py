from data_loader import DataLoader
from baseline import logistic_regression_baseline, time_series_baseline

# Load data
path = "/home/sabmel/.cache/kagglehub/datasets/maxhorowitz/nflplaybyplay2009to2016/versions/6"  
loader = DataLoader(path)
season_2016 = loader.seasons[-1] 
season_2016.clean()

# Select one game 
game = season_2016.games[30] 

# Split into first half (train) and second half (test)
game_train, game_test = game.train_test_split(offense=None)

# Run Logistic Regression Baseline
log_model, log_acc = logistic_regression_baseline(game_train, game_test)

# Run Time Series Baseline
ts_model, ts_mse = time_series_baseline(game_train, game_test, lags=5)
