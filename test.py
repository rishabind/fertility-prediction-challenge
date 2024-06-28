#Load the background data

import pandas as pd

df=pd.read_csv('C:/Users/Tyagi/Desktop/PREfer/5e8ab08c-b634-4948-8b47-8792d36d753f/training_data/PreFer_train_data.csv', low_memory=False)

#Load outcome data
outcomes_df=pd.read_csv("C:/Users/Tyagi/Desktop/PREfer/5e8ab08c-b634-4948-8b47-8792d36d753f/training_data/PreFer_train_outcome.csv", low_memory=False)

### Run clean df function
cleaned_df = clean_df(df) 

### Train and save model
train_save_model(cleaned_df ,outcomes_df)