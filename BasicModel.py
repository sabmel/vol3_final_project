import kagglehub
from football.data_loader import DataLoader
from matplotlib import pyplot as plt
from hmmlearn.hmm import GaussianHMM
import numpy as np
import pandas as pd

class BasicModel():
    def __init__(self, n_components: int):
        """Set up the DataLoader and the GaussianHMM model."""
        # Set up the DataLoader
        path = kagglehub.dataset_download("maxhorowitz/nflplaybyplay2009to2016")
        self.dl = DataLoader(path)
        
        # Indicate we haven't gotten the train/test sets yet
        self.train_yards = None
        self.test_yards = None  
        
        # Define the GaussianHMM with number of components equaling the number of "momenta" states
        self.model = GaussianHMM(n_components=n_components, n_iter=1000)

        
    def get_game_yards(self, season_id=0, game_id=0):
        """Get the yardage gains from the first half (train set)
        and the second half (test set) of the given game"""
        train, test = self.dl[season_id][game_id].train_test_split()
        self.train_yards = train["team0_yards"].to_numpy().reshape(-1, 1)
        self.test_yards = test["team0_yards"].to_numpy().reshape(-1, 1)

    def fit(self):
        """Fit the sequence of yardage gains to the GaussianHMM model"""
        if not self.train_yards:
            print("No game is stored. Running get_game_yards() with default parameters...")
            self.get_game_yards()

        # Train on the observations
        self.model.fit(self.train_yards)
        
    def forecast(self):
        """Forecast the second half observations, and plot them"""
        
        # Predict the hidden states from the observed yard gains
        preds = self.model.predict(self.train_yards)

        # Set the hidden state according to the hidden state at the end of the first half
        startprob = np.zeros(self.model.n_components)
        startprob[preds[-1]] = 1.0
        self.model.startprob_ = startprob
        
        # Forecast out the second half
        # N = length of test set (TODO: placeholder if we want to change this)
        N = len(self.test_yards)
        self.forecast, _ = self.model.sample(N)
    
    def plot_forecast(self):
        # Plot the forecasted values with the actual values
        plt.scatter(np.arange(len(self.forecast)), self.forecast, alpha=0.8, label="Forecast")
        plt.scatter(np.arange(len(self.test_yards)), self.test_yards, alpha=0.8, label="Actual")
        
        # Show the plot
        plt.xlabel("Play Number")
        plt.ylabel("Yards Gained")
        plt.legend()
        plt.show()

    def score(self, y):
        """Predict the second half yardage, calculate the
         yardage differential, and then see if we correctly
         predict the team with more yards in the second half

        y: the true yardage gain list from the 2nd half data
        """

        true_gain = np.sum(y)

        self.forecast()
        predicted = self.forecast
        predicted_gain = np.sum(predicted)

        if (predicted_gain>0 & true_gain>0) | (predicted_gain<=0 & true_gain<=0):
            return 1
        return 0
        
if __name__ == "__main__":
    bm = BasicModel(n_components=7)
    bm.fit()
    bm.forecast()
    bm.plot_forecast()