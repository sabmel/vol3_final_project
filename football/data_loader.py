import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from .season import Season

class DataLoader():
    def __init__(self, path: str):
        """Keep dataframes as class atributes
        
        Parameters:
            - path: ends with the directory the dataset is in
        """
        df = pd.read_csv(path + "/NFL Play by Play 2009-2016 (v3).csv", low_memory=False)

        self.all_data = df
        self.seasons = []

        self.separate_years()

    def separate_years(self) -> list[Season]:
        """Separates the games by which season in which they occured. Outputs a list of seasons."""
        seasons = [Season(self.all_data[self.all_data['Season'] == value].reset_index(drop=True)) for value in self.all_data['Season'].unique()]
        
        self.seasons = seasons
        return seasons
    
    def __getitem__(self, key):
        return self.seasons[key]
    
    def __repr__(self):
        return str([year for year in self.all_data['Season'].unique()])
    
    def __len__(self):
        return len(self.seasons)
    
    def clean(self):
        for season in self.seasons:
            season.clean()

        
