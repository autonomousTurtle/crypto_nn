# crypto_nn
crypto currency recurrent nerual network

to run tensorboard from terminal, navigate to the director where the code is run from: 

`$ tensorboard --logdir=logs/`


### Input to nerual network: 
60 minutes of price data and attempts to predict if the price 3 mintues in the future will be higher or lower than the current time. 


### rnn structure: 

LSTM Layer with 128 nodes

Dropout 20%

LSTM Layer with 128 nodes

Droput 20% 

LSTM Layer with 128 nodes

Dense layer, 32 nodes, relu

Dropout 20%

Dense layer, 2 nodes, softmax (output layer)

Using the Adam optimizer
