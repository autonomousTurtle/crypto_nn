import pandas as pd 
from collections import deque
import random
import numpy as np 
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, ModelCheckpoint 
import time
from sklearn import preprocessing

####################### HYPERPARAMTERS #######################

SEQ_LEN = 60 #how many minutes to look at per sequence - each datapoint is 1 min
FUTURE_PERIOD_PREDICT = 3 #how many minutes in the future to predict
RATIO_TO_PREDICT = "LTC-USD" #which crypto to look at
EPOCHS = 10 #number of passes through the data
BATCH_SIZE = 64 #how many batches of data to run through the network
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"

######################### FUNCTIONS #######################

def classify(current, future):
    if float(future) > float(current): #if the future price is higher, we should buy
        return 1
    else:  #else, we should not buy
        return 0


def preprocess_df(df):
    #use pandas to preprocess the data before sending into the rnn
    df = df.drop("future", 1) #don't need the future value anymore

    for col in df.columns: #go through all the columns 
        if col != "target": #if it is not the target value, normalize (target is already a 1 or 0)
            df[col] = df[col].pct_change # pct_changes normalzies the different currencies to a %
            df.dropna(inplace = True) #throw na if data is missing
            df[col] = preprocessing.scale(df[col].values) #scales all the values between 0 and 1, .values ignores the time parameter

    df.dropna(inplace=True) #clean up again in case we missed one


    sequential_data = [] #sequential_data is a list that will contain the sequences
    prev_days = deque(maxlen = SEQ_LEN) #deque keeps a list with the length SEQ_LEN and kicks the last value out when a new one comes in 

    for i in df.values: #iterate over the values
        prev_days.append([n for n in i[:-1]]) #grab each of the values not including the last value which is the target
        print(prev_days)


main_df = pd.DataFrame() #begin with an empty dataframe

ratios = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"]
for ratio in ratios:
    print(ratio)
    dataset = f'datasets/{ratio}.csv' #path to the dataset file
    df = pd.read_csv(dataset, names=['time', 'low', 'high', 'open', 'close', 'volume']) # read the csv file

    #print(df.head())
    df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True) # rename close and volume to include ratio type to keep track of later
    
    df.set_index("time", inplace = True) #set time as index - we can join datasets by time 
    df = df[[f"{ratio}_close", f"{ratio}_volume"]] #only load in the close and volume columns of the data

    if len(main_df)==0: #if there is nothing in main
        main_df = df #load in the df data
    else: #else, join the dataframes
        main_df = main_df.join(df) 

main_df.dropna(inplace=True) #make sure the data is good
#print(main_df.head()) #check how we did

main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT) # the future is the close value shifted by the period we want to predict
main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future'])) # pass in the current colse data and the future data
    
main_df.dropna(inplace=True)

#split away some data for training and validation
times = sorted(main_df.index.values) #grab the index (time) in sorted order and save as times
last_5pct  = sorted(main_df.index.values)[-int(0.05*len(times))] #take last 5% of timestamps to sort through later

validation_main_df = main_df[(main_df.index >= last_5pct)] #validation data is data in last 5%
main_df = main_df[(main_df.index <= last_5pct)] #training data is data in first 95%


