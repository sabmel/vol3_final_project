import pandas as pd
from game import Game

class Season():
    def __init__(self, df: pd.DataFrame):
        """The main component of a season is its games, which we keep as a list
        """
        self.all_data = df
        self.games = []

        self.separate_games()

    def separate_games(self) -> list[Game]:
        """Separate dataframe into separate games via the game ID. Place into a 
        list of games. Indices are reindexed so plays are numbered, starting with 0
        """
        games = [Game(self.all_data[self.all_data['GameID'] == value].reset_index(drop=True)) for value in self.all_data['GameID'].unique()]
        
        self.games = games
        return games

    def __getitem__(self, key):
        return self.games[key]
    
    def clean(self):
        for game in self.games:
            game.clean()
    
