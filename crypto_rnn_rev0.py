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
NAME = f"{RATIO_TO_PREDICT}-RATIO-{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}" #to save in tensorboard 


######################### FUNCTIONS #######################

# classify the data we are passing to see if the future value is better or worse than the current value
# returns 1 if good, 0 if bad
def classify(current, future):
    if float(future) > float(current): #if the future price is higher, we should buy
        return 1
    else:  #else, we should not buy
        return 0


# process the data frame that is sent in - normalize all the data, create sequences with an associated target value for X min in the future
# returns x and y
def preprocess_df(df): 
    #use pandas to preprocess the data before sending into the rnn
    df = df.drop("future", 1) #don't need the future value anymore

    for col in df.columns: #go through all the columns 
        if col != "target": #if it is not the target value, normalize (target is already a 1 or 0)
            df[col] = df[col].pct_change() # pct_changes normalzies the different currencies to a %
            df.dropna(inplace = True) #throw na if data is missing
            df[col] = preprocessing.scale(df[col].values) #scales all the values between 0 and 1, .values ignores the time parameter
    
    df.dropna(inplace=True) #clean up again in case we missed one


    sequential_data = [] #sequential_data is a list that will contain the sequences
    prev_days = deque(maxlen = SEQ_LEN) #deque keeps a list with the length SEQ_LEN and kicks the last value out when a new one comes in 
 
    for i in df.values: #iterate over the values
        prev_days.append([n for n in i[:-1]]) #grab each of the values not including the last value which is the target
        if len(prev_days) == SEQ_LEN: #make sure we have 60 sequences
            sequential_data.append([np.array(prev_days), i[-1]]) #append the values and the associated target

    random.shuffle(sequential_data) #shuffle to randomize
    #want the same number of buys and sells so we don't skew the nerual net one way or another during training
    buys = []
    sells = []

    for seq, target in sequential_data: #iterate over the sequetial data
        if target == 0: 
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq,target])

    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells)) #find the minimum amount of buys or sells 

    buys=buys[:lower] #keep buys up until the shortest number
    sells=sells[:lower] #keep sells up unitl the shortest number

    #print("buys:", len(buys))
    #print("sells:", len(sells))

    sequential_data = buys+sells
    random.shuffle(sequential_data) # shuffle again so it isn't buys and sells back to back

    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq) #X is the sequences
        y.append(target) #y is the target

    return np.array(X), y


################## THE MAIN PROGRAM ############################3

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

main_df.fillna(method="ffill", inplace=True)
main_df.dropna(inplace=True) #make sure the data is good
#print(main_df.head()) #check how we did

main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT) # the future is the close value shifted by the period we want to predict
main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future'])) # pass in the current colse data and the future data
    
main_df.dropna(inplace=True)

#split away some data for training and validation
times = sorted(main_df.index.values) #grab the index (time) in sorted order and save as times
last_5pct  = sorted(main_df.index.values)[-int(0.05*len(times))] #take last 5% of timestamps to sort through later

validation_main_df = main_df[(main_df.index >= last_5pct)] #validation data is data in last 5%
main_df = main_df[(main_df.index < last_5pct)] #training data is data in first 95%


train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)

#print some stats
print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")

################### NEURAL NETWORK MODEL ########################3

model = Sequential()

model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=(train_x.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

#output layer, binary choice
model.add(Dense(2, activation="softmax"))

opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)

model.compile(loss='sparse_categorical_crossentropy', optimizer = opt, metrics=['accuracy'])

#callbacks
tensorboard = TensorBoard(log_dir=f'logs/{NAME}') #tensorboard
filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}" #checkpoints
#heckpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only = True, mode='max'))

#re
train_x = np.asarray(train_x) 
train_y = np.asarray(train_y)
validation_x = np.asarray(validation_x)
validation_y = np.asarray(validation_y)

history = model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS,validation_data=(validation_x, validation_y), callbacks=[tensorboard])#, checkpoint])

#score the model
score=model.evaluate(validation_x, validation_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy', score[1])
#save model
model.save(" models/{}".format(NAME))