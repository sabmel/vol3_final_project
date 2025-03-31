import pandas as pd
import numpy as np
from .team import Team

class Game():
    def __init__(self, df: pd.DataFrame):
        self.data = df
        self.cleaned = False

    def calculate_time_per_play(self) -> pd.DataFrame:
        """Create a new column which is the time each play took.
        Kicks will have NANs in the new play_time column, which should make them easy to remove
        """
        self.data['play_time'] = -self.data['TimeSecs'].diff()

        return self.data


    def drop_unnecessary_rows(self) -> pd.DataFrame:
        """The end of each quarter is its own row. Same with timeouts
        and the end of the game. Other values are mostly NANs.
        This removes all of those unhelpful rows and reindexes

        NOTE: plays must be indexed starting with their first play
        TODO: Might be able to just drop rows with missing posteam
        """
        # find indices
        # entries without a team in posession typically are timeouts or ends of quarters
        self.data.dropna(subset=['posteam', 'play_time'], inplace=True)
        # reset index
        self.data = self.data.reset_index(drop=True)

        return self.data

    def encode_teams(self) -> pd.DataFrame:
        """Change all team names to just 0s or 1s. This won't be retraceable if you are
        looking for a self.data with a specific team playing.
        """
        teams = self.data['posteam'].unique()
        if len(teams) != 2:
            print(teams)
            raise ValueError("Dataset has not been properly cleaned. There are more than 2 values in posteam.")

        home_team = self.data["HomeTeam"][0]
        team_map = {team:0 if team == home_team else 1 for team in teams} # this marks the home team as team 0
        self.data['posteam'] = self.data['posteam'].map(team_map)
        self.data['DefensiveTeam'] = self.data['DefensiveTeam'].map(team_map)

        return self.data

    def create_team0_yardage(self) -> pd.DataFrame:
        """Create a new column which is the yards gained in the play by team zero. 
        It is negative if team 1 is in posession and gains yards.
        """
        self.data['team0_yards'] = np.where(self.data['posteam'] == 0, self.data['Yards.Gained'], -self.data['Yards.Gained'])

        return self.data

    def no_overtime(self) -> pd.DataFrame:
        """Discard all data on overtime periods. 
        Will result in another Dataframe
        """
        self.data = self.data[self.data["qtr"]!=5]
        return self.data
    
    def __repr__(self):
        return self.data.to_string()
    
    def clean(self):
        # If we accidentally clean again, it deletes all of 'posteam' and 'Defensiveteam'
        if self.cleaned:
            return

        self.calculate_time_per_play()
        self.drop_unnecessary_rows()
        self.encode_teams()
        self.create_team0_yardage()
        self.no_overtime()
        self.cleaned = True
    
    def train_test_split(self, home_team=None):
        """Create a train-test split where the training data comes from the first half
        and the test data comes from the second half.
        Save the train/test sets as attributes and also return them.
        
        If home_team=None (default parameter), all rows are included in the train/test splits.
        If home_team=True, only rows corresponding to home possessions are included.
        If home_team=False, only rows corresponding to away possessions are included.
        """
        
        # Clean the datasets (if not already cleaned)
        if not self.cleaned:
            self.clean()
        
        # Detect whether we are filtering based on possession:
        # No filtering
        if home_team is None:
            # Select train data from the first half and test data from the second
            self.train = self.data[self.data["qtr"].isin([1, 2])]
            self.test = self.data[self.data["qtr"].isin([3,4])]
            
            return self.train, self.test            
        
        # Filtering for home team on offense
        if home_team == True:
            posteam = 0
            
        # Filtering for away team on defense
        elif home_team == False:
            posteam = 1

        # Select train data from the first half and test data from the second
        self.train = self.data[(self.data["qtr"].isin([1, 2])) & (self.data["posteam"] == posteam)]
        self.test = self.data[(self.data["qtr"].isin([3,4])) & (self.data["posteam"] == posteam)]
        
        return self.train, self.test
        
    def play_(self, hometeam: Team, awayteam: Team, start_team: int) -> tuple[list]:
        """Helper function
        
        Returns:
            - home_list (list): list of yards gained by home team
            during each of their possessions
            - away_list (list): list of yards gained by away team
            during each of their possession
        """
        time_left  = 1800 # seconds in 2nd half

        # keep yards gained by each team in separate lists to start with
        home_list = []
        away_list = []

        if start_team == 0:
            while time_left > 0:
                home_yards, time_spent = hometeam.play_possession()
                time_left -= time_spent
                home_list.append(home_yards)
                if time_left < 0: break # Away team has no time to play
                away_yards, time_spent = awayteam.play_possession()
                time_left -= time_spent
                away_list.append(away_yards)
        elif start_team == 1:
            while time_left > 0:
                away_yards, time_spent = awayteam.play_possession()
                time_left -= time_spent
                away_list.append(away_yards)
                if time_left < 0: break # Away team has no time to play
                home_yards, time_spent = hometeam.play_possession()
                time_left -= time_spent
                home_list.append(home_yards)
        else:
            print(f"start_team was set to {start_team}. It needs to be 0 or 1.")

        return home_list, away_list

    def predict_2nd_half(self, n_hidden_states) -> pd.DataFrame:
        """Use data from the first half of a game to predict the 2nd half. This
        method defers to the Team class to make predictions. The Team class uses
        GMMHMM 

        Parameters:
            - n_hidden_states (int) : The number of hidden states used in the HMM

        Returns:
            - possession (pd.DataFrame) : Pandas dataframe array with home team yards 
                gained during possessions in the first column and away team yards 
                gained during possessions
                in the 2nd column
        """

        # Clean the datasets (if not already cleaned)
        if not self.cleaned:
            self.clean()
        
        # Get plays from first half
        hometrain, hometest = self.train_test_split(home_team=True) # TODO: do something with the test set?
        hometrain, hometest = hometrain.copy(), hometest.copy()
        awaytrain, awaytest = self.train_test_split(home_team=False)
        awaytrain, awaytest = awaytrain.copy(), awaytest.copy()

        # Initialize teams
        hometeam = Team(hometrain, n_components=n_hidden_states)
        awayteam = Team(awaytrain, n_components=n_hidden_states)

        # Choose first team to play in the 2nd half
        #   Assume the opposite team starts 2nd half
        start_team = 1 - self.data.iloc[0]["posteam"]


        # Play 2nd half
        home_list, away_list = self.play_(hometeam, awayteam, start_team)

        # Put yards together into dataframe with 2 columns of equal length
        def append_zero_to_shorter_(list1, list2):
            if len(list1) < len(list2): list1.append(0)
            elif len(list2) < len(list1): list2.append(0)
            return list1, list2
        home_list, away_list = append_zero_to_shorter_(home_list, away_list)

        print(home_list)
        print()
        print(away_list)
        print("Length of list: ", len(home_list), len(away_list))
        print("possession potential shape: ", np.array([home_list, away_list]).shape)
        print("Length of list: ", len(home_list))
        print(np.array([home_list, away_list]))
        possessions = pd.DataFrame(np.array([home_list, away_list]).T)

        return possessions
