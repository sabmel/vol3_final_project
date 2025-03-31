import kagglehub
from football.data_loader import DataLoader
from matplotlib import pyplot as plt
from pyhhmm.heterogeneous import HeterogeneousHMM
import numpy as np
import pandas as pd

# NOTE: you need to install pyhhmm from the GitHub repo, not from pip
# ! python -m pip install "git+https://github.com/fmorenopino/HeterogeneousHMM"

class HeterogeneousModel():
    def __init__(self, n_components: int):
        """Set up the DataLoader and the HeterogeneousHMM model."""
        # Set up the DataLoader
        path = kagglehub.dataset_download("maxhorowitz/nflplaybyplay2009to2016")
        self.dl = DataLoader(path)
        
        # Indicate we haven't gotten the train/test sets yet
        self.train_cols = None
        self.test_cols = None
        
        # Define the HeterogeneousHMM with number of components equaling the number of "momenta" states
        self.model = HeterogeneousHMM(
            n_states=n_components,  # experiment with this
            n_g_emissions=1,        # number of continuous observations
            n_d_emissions=1,        # number of discrete observations
            n_d_features=[2],       # number of possible values for each discrete variable
            covariance_type="diagonal",
            verbose=True
        )

        
    def get_game_yards_posessions(self, season_id=0, game_id=0):
        """Get the yardage gains and possession ID from the first half (train set)
        and the second half (test set) of the given game"""
        train, test = self.dl[season_id][game_id].train_test_split()
        self.train_cols = train[["Yards.Gained", "posteam"]].to_numpy().reshape(-1, 2)
        self.test_cols = test[["Yards.Gained", "posteam"]].to_numpy().reshape(-1, 2)

    def fit(self):
        """Fit the sequence of yardage gains to the GaussianHMM model"""
        if not self.train_cols:
            print("No game is stored. Running get_game_yards() with default parameters...")
            self.get_game_yards_posessions()

        # Train on the observations
        self.model, log_likelihood = self.model.train([self.train_cols], n_init=10, n_iter=10)
        
    def forecast(self):
        """Forecast the second half observations, and plot them"""
        
        # Forecast out the second half
        # N = length of test set (TODO: placeholder if we want to change this)
        N = len(self.test_cols)        
        samples = self.model.sample(n_samples=N)
        self.forecast = samples[0]
    
    def plot_forecast_yards(self):
        # Plot the forecasted values with the actual values
        plt.scatter(np.arange(len(self.forecast)), self.forecast[:,0], alpha=0.8, label="Forecast")
        plt.scatter(np.arange(len(self.test_cols)), self.test_cols[:,0], alpha=0.8, label="Actual")
        
        # Show the plot
        plt.xlabel("Play Number")
        plt.ylabel("Yards Gained")
        plt.legend()
        plt.show()
        
    def plot_forecast_posessions(self):
        # Plot the forecasted possession values with the actual values
        plt.scatter(np.arange(len(self.forecast)), self.forecast[:,1], alpha=0.8, label="Forecast")
        plt.scatter(np.arange(len(self.test_cols)), self.test_cols[:,1], alpha=0.8, label="Actual")
        
        # Show the plot
        plt.xlabel("Play Number")
        plt.ylabel("Team Possessing")
        plt.legend()
        plt.show()        
        
if __name__ == "__main__":
    print("Generating model...")
    hm = HeterogeneousModel(n_components=7)
    print("Fitting model...")
    hm.fit()
    print("Forecasting...")
    hm.forecast()
    print("Plotting results...")
    hm.plot_forecast_yards()
    hm.plot_forecast_posessions()
