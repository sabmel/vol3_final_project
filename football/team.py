import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM

class Team():
    """A class which can be used for simulating games. It is only used for a
    single game. The Team() class keeps track of downs, yards to make, and 
    total yards gained in one game.
    """

    def __init__(self, train: pd.DataFrame, n_components: int = 3):
        """
        Parameters:
            - plays (df.DataFrame): Plays already made, from the first half of the game.
            - home_team (bool) : Boolean indicating whether the team is the home team1
            - n_components (int) : number of hidden state components in the GMMHMM
        """
        # Store parameters
        self.n_components = n_components

        # Save previous play data
        self.previous_plays = train

        # Calculate yardage from first half
        self.first_half_yrds = self.previous_plays['Yards.Gained'].sum()

        self.train_GMMHMM()

        # Important attributes for playing the game

    def train_GMMHMM(self):
        """Train a GMMHMM to predict the time spent on each play and yardage gained.
        """
        # Training data 
        obs = self.previous_plays[["Yards.Gained", "play_time"]]

        # Create and train several models, keep only the best
        best_model = None
        best_score = -np.inf
        for seed in range(15):
            model = GaussianHMM(n_components=self.n_components, covariance_type="full", random_state=seed, n_iter=1000)
            model.fit(obs)
            if model.score(obs) > best_score:
                best_model = model
        self.model = best_model

        # Store the last hidden state
        preds = self.model.predict(obs)
        startprob = np.zeros(self.n_components)
        startprob[preds[-1]] = 1.0
        self.last_hidden_state = startprob # TODO: what does this look like? An ndarray?




    def play_possession(self):
        """Play through a possession. The team only loses the ball if they run out of
        downs.
        
        Returns:
            - yards gained (int): total yards gained from the play, rounded to an integer
            - time spent (float): seconds spent in possession of the ball
        """
        yards_gained = 0
        time_spent = 0

        i = 0
        # Play until the team runs out of downs
        while self.down < 5:
            i += 1
            # pull play yardage and play time from the GMMHMM
            self.model.startprob_ = self.last_hidden_state
            z, x = self.model.sample(1)
            # split observation
            yards, time = z.T[0], z.T[1]
            yards, time = yards[0], time[0]
            yards = yards.round()
            # update possession information
            yards_gained += yards
            time_spent += time
            self.down += 1
            self.yards_to_make -= yards
            # update last hidden state
            startprob = np.zeros(self.n_components)
            startprob[x] = 1.0
            self.last_hidden_state = startprob
            # reset down if 10 yards gained
            if self.yards_to_make <= 0:
                # reset down count and yards to make
                self.down = 1
                self.yards_to_make = 10
            # if we have covered over 70 yards, consider it a turnover
            if yards_gained >= 70:
                break
        

        return yards_gained, time_spent

    def play_drive(self):
        """Play a single drive. 
        
        Returns:
            - yards gained (int): total yards gained from the play, rounded to an integer
            - time spent (float): seconds spent in possession of the ball
        """
        # pull play yardage and play time from the GMMHMM
        #self.model.startprob_ = self.last_hidden_state
        z, x = self.model.sample(1)

        # split observation
        yards, time = z.T[0], z.T[1]
        yards, time = yards[0], time[0]
        yards = yards.round()

        # update last hidden state
        #startprob = np.zeros(self.n_components)
        #startprob[x] = 1.0
        #self.last_hidden_state = startprob

        return yards, time

