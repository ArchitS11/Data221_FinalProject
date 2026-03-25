# Data 221 Final Project Code
# Preprocessing (combine + split + scale)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data(train_path='train.csv', test_path='test.csv'):
    # Load the kaggle datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
