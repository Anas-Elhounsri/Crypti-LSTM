import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import LSTM, Dense
import math

data_op = pd.read_csv("BTC-USD.csv", usecols=[1])

#normalizing the data to values between -1 and 1 since LSTM uses 
#activation functions tanh
data_op = data_op.values.reshape(-1, 1)
data_op = data_op / np.max(data_op)

#sliptting the data to 80% training and 20% testing
train_size = int(len(data_op) * 0.8)
test_size = len(data_op) - train_size
train_data, test_data = data_op[0:train_size,:], data_op[train_size:len(data_op),:]

############################################################
#create input and output function data for our LSTM model
def lstm_dataset(data_op, look_back=1):
    x,y =[],[]

    for i in range(len(data_op)-look_back):
        #x contains sequences of length look_back from the input data_op which will work as input data for LSTM
        x.append(data_op[i:(i+look_back),0])

        #y contains the next value after each sequence in x which will work as output data for LSTM
        y.append(data_op[i + look_back, 0])

    return np.array(x), np.array(y)

# [1,2,3,4,5,6,7,8,9,10]
# x=[[1,2,3],[2,3,4]]
# y = [4,5]

look_back = 30
train_x, train_y = lstm_dataset(train_data,look_back)#80% of the data
test_x, test_y = lstm_dataset(test_data,look_back)#the remaining 20% of the data

#reshaping the input data for LSTM model as 
#[samples, features, look back]
train_x = train_x.reshape((train_x.shape[0], 1, look_back))
test_x = test_x.reshape((test_x.shape[0], 1, look_back))

############################################################
#defining our LSTM model

#creates a linear stack of layers 
model = Sequential()

#adds and LSTM layer to the model with 64 LSTM units in the layer
#the input_shape parameter specifies the shape of the input data, 
#which in this case is one feature(Opening prices) and look_back which is 10 days 
model.add(LSTM(64, input_shape=(1, look_back)))

#adds a Dense layer, the 1 parameter specifies the output size of the layer
model.add(Dense(1))#needs some research

#compiles the model while specifying the loss function and optimizer
#adam is a common optimizer used in deep learning
model.compile(loss='mean_squared_error', optimizer='adam')

#prints a summary of the model architecture
model.summary()
#by defaulytn the activation function used by the LSTM layer in Keras is Tanh

############################################################
#training and evaluating the LSTM model
#fiiting and training the model on the training data
model.fit(train_x, train_y, validation_data=(test_x, test_y),verbose=2, epochs=100)

#Evaluate the LSTM model
#double check the evaluation process
train_score = model.evaluate(train_x, train_y, verbose=0)
test_score = model.evaluate(test_x, test_y, verbose=0)
print('Train Score: {:.2f} MSE, ({:.2f} RMSE)'.format(train_score, np.sqrt(train_score)))
print('Test Score: {:.2f} MSE, ({:.2f} RMSE)'.format(test_score, np.sqrt(test_score)))

############################################################
#Finally predition and normilizing the data
train_predictions = model.predict(train_x)
test_predictions = model.predict(test_x)

# Invert the normalization of the data
train_predictions = train_predictions * np.max(data_op)
train_Y = train_y * np.max(data_op)
test_predictions = test_predictions * np.max(data_op)
test_Y = test_y * np.max(data_op)

#Plotting the prediction 
plt.plot(train_y, label = "train_y")
plt.plot(train_predictions, label="train_predictions")
plt.plot(test_y, label = "test_y")
plt.plot(test_predictions, label = "test_predictions")
plt.legend(title = "Legend")
plt.show()


