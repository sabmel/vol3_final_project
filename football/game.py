import pandas as pd
import numpy as np

class Game():
    def __init__(self, df: pd.DataFrame):
        self.data = df

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
    
    def clean(self):
        self.calculate_time_per_play()
        self.drop_unnecessary_rows()
        self.encode_teams()
        self.create_team0_yardage()
        self.no_overtime()
        
