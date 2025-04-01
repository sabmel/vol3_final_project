import kagglehub
from football.data_loader import DataLoader
from matplotlib import pyplot as plt
from pyhhmm.heterogeneous import HeterogeneousHMM
import numpy as np
import pandas as pd
from multiprocessing import Pool
import multiprocessing
import warnings

# NOTE: you need to install pyhhmm from the GitHub repo, not from pip
# ! python -m pip install "git+https://github.com/fmorenopino/HeterogeneousHMM"

class HeterogeneousModel():
    def __init__(self, n_components: int, dl=None):
        """Set up the DataLoader and the HeterogeneousHMM model."""
        
        # Set up the DataLoader
        if (dl == None):
            path = kagglehub.dataset_download("maxhorowitz/nflplaybyplay2009to2016")
            self.dl = DataLoader(path)
        else:
            self.dl = dl
        
        # Indicate we haven't gotten the train/test sets yet
        self.train_cols = None
        self.test_cols = None
        self.prediction = None
        
        # Define the HeterogeneousHMM with number of components equaling the number of "momenta" states
        self.model = HeterogeneousHMM(
            n_states=n_components,  # experiment with this
            n_g_emissions=1,        # number of continuous observations
            n_d_emissions=1,        # number of discrete observations
            n_d_features=[2],       # number of possible values for each discrete variable
            covariance_type="diagonal",
            verbose=False
        )

        
    def get_game_yards_possessions(self, season_id=0, game_id=0):
        """Get the yardage gains and possession ID from the first half (train set)
        and the second half (test set) of the given game"""
        train, test = self.dl[season_id][game_id].train_test_split()
        self.train_cols = train[["Yards.Gained", "posteam"]].to_numpy().reshape(-1, 2)
        self.test_cols = test[["Yards.Gained", "posteam"]].to_numpy().reshape(-1, 2)

    def fit(self):
        """Fit the sequence of yardage gains to the GaussianHMM model"""
        if self.train_cols is None:
            print("No game is stored. Running get_game_yards() with default parameters...")
            self.get_game_yards_possessions()

        # Train on the observations
        self.model, log_likelihood = self.model.train([self.train_cols], n_init=1, n_iter=100)
        
        # Add smoothing
        # eps = 1e-6
        # self.model.A += eps
        # self.model.A /= self.A.sum(axis=1, keepdims=True)
        
    def forecast(self):
        """Forecast the second half observations, and plot them"""
        
        # Forecast out the second half
        # N = length of test set (TODO: placeholder if we want to change this)
        N = len(self.test_cols)        
        samples = self.model.sample(n_samples=N)
        self.prediction = samples[0]
    
    def plot_forecast_yards(self):
        # Plot the forecasted values with the actual values
        plt.scatter(np.arange(len(self.prediction)), self.prediction[:,0], alpha=0.8, label="Forecast")
        plt.scatter(np.arange(len(self.test_cols)), self.test_cols[:,0], alpha=0.8, label="Actual")
        
        # Show the plot
        plt.xlabel("Play Number")
        plt.ylabel("Yards Gained")
        plt.legend()
        plt.show()
        
    def plot_forecast_posessions(self):
        # Plot the forecasted possession values with the actual values
        plt.scatter(np.arange(len(self.prediction)), self.prediction[:,1], alpha=0.8, label="Forecast")
        plt.scatter(np.arange(len(self.test_cols)), self.test_cols[:,1], alpha=0.8, label="Actual")
        
        # Show the plot
        plt.xlabel("Play Number")
        plt.ylabel("Team Possessing")
        plt.legend()
        plt.show()  
        
    def score(self, y):
        """Predict the second half yardage, calculate the
         yardage differential, and then see if we correctly
         predict the team with more yards in the second half

        y: the true yardage gain list from the 2nd half data
        """

        h1_net_yards = np.sum(self.train_cols[:,0][self.train_cols[:,1] == 0]) \
                        - np.sum(self.train_cols[:,0][self.train_cols[:,1] == 1])
                        
        true_gain = np.sum(y[:,0][y[:,1] == 0]) - np.sum(y[:,0][y[:,1] == 1])

        predicted_gain = np.sum(self.prediction[:,0][self.prediction[:,1] == 0]) \
                        - np.sum(self.prediction[:,0][self.prediction[:,1] == 1])

        if (predicted_gain + h1_net_yards > 0 and true_gain + h1_net_yards > 0) \
        or (predicted_gain + h1_net_yards <= 0 and true_gain + h1_net_yards <= 0):
            return 1
        
        return 0

def run_model(tup):
    """Run model for season i, game j"""
    i, j = tup
    model_correct = 0
    for n in range(10):
        hm = HeterogeneousModel(n_components=7, dl=loader)
        # print(i,j)
        hm.get_game_yards_possessions(i, j)
        hm.fit()
        hm.forecast()
        model_correct += hm.score(hm.test_cols)
        # hm.plot_forecast_yards()
        # hm.plot_forecast_posessions()

    return model_correct / 10


if __name__ == "__main__":
    multiprocessing.set_start_method("fork")
    path = kagglehub.dataset_download("maxhorowitz/nflplaybyplay2009to2016")
    correct = 0
    n = 500
    num_seasons = 8

    #There are more than 200 games, but lets start with this
    loader = DataLoader(path)
    correct_percents = []

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        for i in range(num_seasons):
            with Pool() as pool:
                for corr in pool.imap_unordered(run_model, zip([i]*len(loader[i]), range(len(loader[i])))):
                    correct_percents.append(corr)
                    print("PERC CORRECT:", corr)
    
    print(sum(correct_percents)/len(correct_percents))
