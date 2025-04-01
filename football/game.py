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
        self.data.dropna(subset=['posteam', 'play_time','FirstDown'], inplace=True)
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
        if self.cleaned:
            print("game data is already clean")
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

    
    def play_half(self, startteam: Team, otherteam: Team, start_is_home: bool) -> pd.DataFrame:
        """Helper function
        
        Returns:
            - plays (pd.Dataframe) : Dataframe with each play and an index of seconds left in the half
        """
        time_left  = 1800 # seconds in 2nd half

        # keep yards gained by each team in separate lists to start with
        play_times = []
        start_team_plays = []
        other_team_plays = []
        remaining_times = []
        pos_team = []

        def play_possession_(team: Team):
            '''Play one possession with a team

            Returns: list of yards, list of times
            '''
            nonlocal time_left
            yards = []
            times = []
            # Keep track of downs and yards left
            down = 1        # you have 4 of these before you "turn over" the ball
            yards_to_make = 10 # need to make 10 yards to reset to down 1 (first down)
            yards_made = 0 # we stop a possession after 70 yards gained (probably crossed the field)
            while down <= 4:
                # record remaining time
                remaining_times.append(time_left)

                # play one drive and record
                yard_gain, time_spent = team.play_drive()
                yards.append(yard_gain)
                times.append(time_spent)

                # update game variables
                down += 1
                yards_to_make -= yard_gain
                yards_made += yard_gain
                # update time
                time_left -= time_spent

                # reset down and yards_to_make if necessary
                if yards_to_make <= 0:
                    down = 1
                    yards_to_make = 10

                # Conditions to end possession
                if time_left <= 0:
                    return yards, times # end possession for the end of the game
                elif yards_made >= 70:
                    return yards, times # end possession for (probably) making a touchdown
            
            return yards, times
        
        # Play the 2nd half of the game
        while time_left > 0: 
            # let starting team play
            yards, time = play_possession_(startteam)
            # update lists
            play_times.extend(time)
            start_team_plays.extend(yards)
            other_team_plays.extend([0]*len(yards))
            pos_team.extend([0]*len(yards)) # assume home team is starting

            # check for game being over
            if time_left <= 0: break 

            # have other team play
            yards, time = play_possession_(otherteam)
            # update lists
            play_times.extend(time)
            other_team_plays.extend(yards)
            start_team_plays.extend([0]*len(yards))
            pos_team.extend([1]*len(yards)) # assume away team goes 2nd

        # Create the DataFrame
        plays = pd.DataFrame({
            'play_time'     : play_times,
            'time_remaining': remaining_times,
            })
        
        # rename according to if away team started
        if start_is_home:
            plays['posteam'] = np.array(pos_team) # we assumed that the home team started
            plays['team0_yards'] = np.array(start_team_plays) - np.array(other_team_plays)
        else:
            plays['posteam'] = 1 - np.array(pos_team)
            plays['team0_yards'] = np.array(other_team_plays) - np.array(start_team_plays)
        plays['Yards.Gained'] = np.array(start_team_plays) + np.array(other_team_plays)

        return plays

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
        if start_team == 0:
            prediction = self.play_half(hometeam, awayteam, True)
        else:
            prediction = self.play_half(awayteam, hometeam, False)

        return prediction
