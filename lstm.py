import numpy as np
import boto3
import csv
import io
import json
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

s3_client = boto3.client('s3')
src_bucket_name = 's3-bucket-name'
dst_bucket_name = 's3-bucket-name'
# lists the objects in the s3 bucket '(the csv files)
response = s3_client.list_objects(Bucket=src_bucket_name, Prefix='object-name/')

#This will loop through each object for processing and predicting data
for content in response['Contents']:
    # reads the file
    response = s3_client.get_object(Bucket=src_bucket_name, Key=content['Key'])
    data = response['Body'].read()
    #converts the file into readable format
    data_file = io.BytesIO(data)
    data_op = pd.read_csv(data_file, usecols = [1])

    # normalizing the data to values between 0 and 1 since LSTM uses 
    # activation functions tanh which sensitive to certian magnitude
    data_op = data_op.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range = (0,1))
    data_op = scaler.fit_transform(data_op)
    # plt.plot(data_op)
    # plt.show()
    #data_op = (data_op - min(data_op))/(max(data_op)-min(data_op))

    #sliptting the data to 80% training and 20% testing
    train_size = int(len(data_op) * 0.8)
    test_size = len(data_op) - train_size
    train_data, test_data = data_op[0:train_size,:], data_op[train_size:len(data_op),:]

###########################################################
    # create input and output function data for our LSTM model
    def lstm_dataset(data_op, look_back = 1):
        x,y =[],[]

        for i in range(len(data_op)-look_back):
            #x contains sequences of length look_back from the input data_op which will work as input data for LSTM
            x.append(data_op[i:(i+look_back),0])

            #y contains the next value after each sequence in x which will work as output data for LSTM
            y.append(data_op[i + look_back, 0])

        return np.array(x), np.array(y)

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
    model.add(LSTM(64, input_shape = (1, look_back)))

    #adds a Dense layer, the 1 parameter specifies the output size of the layer
    model.add(Dense(1))#needs some research

    #compiles the model while specifying the loss function and optimizer
    #adam is a common optimizer used in deep learning
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')

    #prints a summary of the model architecture
    model.summary()

############################################################
    #training and evaluating the LSTM model
    #fitting and training the model on the training data
    model.fit(train_x, train_y, validation_data = (test_x, test_y), verbose = 2, epochs = 100)

    train_predictions = model.predict(train_x)
    test_predictions = model.predict(test_x)

    #Evaluate the LSTM model
    #(Self Note)double check the evaluation process
    train_score = model.evaluate(train_x, train_y, verbose = 0)
    test_score = model.evaluate(test_x, test_y, verbose = 0)  
    print('Train Score: {:.2f} MSE, ({:.2f} RMSE)'.format(train_score, np.sqrt(train_score)))
    print('Test Score: {:.2f} MSE, ({:.2f} RMSE)'.format(test_score, np.sqrt(test_score)))

    last_thirty_days = data_op[-30:]
    predicted_days = []
    count = 0

    for i in range(30):

        #reshapes the data accordingly for the LSTM model e.g[[['a'],['b'],['c']]]
        last_thirty_days = last_thirty_days.reshape(1,1,30)
        #predicts the next 30 days
        predictions = model.predict(last_thirty_days)
        predicted_days.append(predictions)

        #adds the predicted day at the end the list
        last_thirty_days = np.append(last_thirty_days, predictions)
        #shifts the list to get rid of the oldest day and keep it 30 days for the LSTM to predict with the new data
        last_thirty_days = last_thirty_days[1:]

        #printing a counter to make sure it loops 30 times
        print("this is count: " + str(count))
        # print(last_thirty_days)
        count = count + 1

    #converts the predicted days to a numpy array
    predicted_days = np.array(predicted_days)
    #reverts the values of the predicted days to original values while reshaping the array into a 2D array
    predicted_days = scaler.inverse_transform(predicted_days.reshape(-1,1))

    json_arr = predicted_days[:,0].tolist()
    ct = datetime.datetime.utcnow()
    
    #This slicing operation gets the name of the coins from content['Key'] array
    coin_name = content['Key'].split('/')[1].split('.')[0]
    dict_data = {
        "coin_name" : coin_name,
        "timestamp": ct.isoformat(),
        "prediction_price_list": json_arr,
        "mse": f'{train_score}',
        "rmse": f'{np.sqrt(train_score)}'
    }

    #creates a json file
    with open(f'{coin_name}.json', "w") as f:
        json.dump(dict_data, f)

    filename = f'{coin_name}.json'
    object_key =  filename

    #uploads the file into an different s3 bucket
    s3_client.upload_file(object_key, dst_bucket_name, filename)

    print("Works fine!!")
    print(response)


######################################################

#Finally predition and inverting the data
# train_predictions = model.predict(train_x)
# test_predictions = model.predict(test_x)

# #Evaluate the LSTM model
# #(Self Note)double check the evaluation process
# train_score = model.evaluate(train_x, train_y, verbose = 0)
# test_score = model.evaluate(test_x, test_y, verbose = 0)
# print('Train Score: {:.2f} MSE, ({:.2f} RMSE)'.format(train_score, np.sqrt(train_score)))
# print('Test Score: {:.2f} MSE, ({:.2f} RMSE)'.format(test_score, np.sqrt(test_score)))

# # Invert the normalization of the data
# train_predictions = scaler.inverse_transform(train_predictions)
# train_y = scaler.inverse_transform([train_y])
# test_predictions = scaler.inverse_transform(test_predictions)
# test_y = scaler.inverse_transform([test_y])

######################################################

# Plotting the prediction 
# plt.plot(train_predictions, label='Train Predictions')
# plt.plot(train_y.flatten(), label='Train Y')
# plt.plot(test_predictions, label='Test Predictions')
# plt.plot(test_y.flatten(), label='Test Y')
# plt.legend()
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.title('Predicted vs Actual Values')
# plt.show()

######################################################




