import pandas as pd
import numpy as np

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
        
        team_map = {team:i for i, team in enumerate(teams)}
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
        self.calculate_time_per_play()
        self.drop_unnecessary_rows()
        self.encode_teams()
        self.create_team0_yardage()
        self.no_overtime()
        self.cleaned = True
    
    def train_test_split(self, offense=None):
        """Create a train-test split where the training data comes from the first half
        and the test data comes from the second half.
        Save the train/test sets as attributes and also return them.
        
        If offense=None (default parameter), all rows are included in the train/test splits.
        If offense=True, only rows corresponding to team0 possessions are included.
        If offense=False, only rows corresponding to team1 possessions are included.
        """
        
        # Clean the datasets (if not already cleaned)
        if not self.cleaned:
            self.clean()
        
        # Detect whether we are filtering based on possession:
        # No filtering
        if offense is None:
            # Select train data from the first half and test data from the second
            self.train = self.data[self.data["qtr"].isin([1, 2])]
            self.test = self.data[self.data["qtr"].isin([3,4])]
            
            return self.train, self.test            
        
        # Filtering for team0 on offense
        if offense == True:
            posteam = 0
            
        # Filtering for team0 on defense
        elif offense == False:
            posteam = 1

        # Select train data from the first half and test data from the second
        self.train = self.data[(self.data["qtr"].isin([1, 2])) & (self.data["posteam"] == posteam)]
        self.test = self.data[(self.data["qtr"].isin([3,4])) & (self.data["posteam"] == posteam)]
        
        return self.train, self.test
