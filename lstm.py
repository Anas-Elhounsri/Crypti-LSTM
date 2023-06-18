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

look_back = 30
s3_client = boto3.client('s3')
src_bucket_name = "<Insert-bucket-name>"
dst_bucket_name = "<Insert-bucket-name>"
# lists the objects in the s3 bucket '(the csv files)
response = s3_client.list_objects(Bucket=src_bucket_name, Prefix= "<Insert-Prefix-name-if-any>" )

#This will loop through each object for processing and predicting data
for content in response['Contents']:
    # reads the file
    response = s3_client.get_object(Bucket=src_bucket_name, Key=content['Key'])
    data = response['Body'].read()
    #converts the file into readable format
    data_file = io.BytesIO(data)
    data_op = pd.read_csv(data_file, usecols = [1])

    # normalizing the data to values between 0 and 1 since LSTM uses 
    # activation functions tanh which sensitive to certain magnitude
    data_op = data_op.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range = (0,1))
    data_op = scaler.fit_transform(data_op)

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
    #which in this case is one feature(Opening prices) and look_back which is 30 days 
    model.add(LSTM(64, input_shape = (1, look_back), return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))

    #adds a Dense layer, the 1 parameter specifies the output size of the layer

    model.add(Dense(30))
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
    print(f"This is train predictions:\n{train_predictions}")
    print(f"Length: {len(train_predictions)}")
    print(f"Length of train prediction arrays: {len(train_predictions[0])}")
    test_predictions = model.predict(test_x)

    #Evaluate the LSTM model
    train_score = model.evaluate(train_x, train_y, verbose = 0)
    test_score = model.evaluate(test_x, test_y, verbose = 0)  
    print('Train Score: {:.2f} MSE, ({:.2f} RMSE)'.format(train_score, np.sqrt(train_score)))
    print('Test Score: {:.2f} MSE, ({:.2f} RMSE)'.format(test_score, np.sqrt(test_score)))

    #Invert the normalization of the data
    train_predictions = scaler.inverse_transform(train_predictions)
    train_y = scaler.inverse_transform([train_y])
    test_predictions = scaler.inverse_transform(test_predictions)
    test_y = scaler.inverse_transform([test_y])
    
    last_thirty_days = data_op[-look_back:]
    predicted_days = []
    count = 0

    last_thirty_days = last_thirty_days.reshape(1,1,look_back)
    
    #predicts the next 30 days
    predictions = model.predict(last_thirty_days)
    predicted_days.append(predictions)
    predicted_days = np.array(predicted_days)
    predicted_days = scaler.inverse_transform(predicted_days.reshape(-1,1))
    
    json_arr = predicted_days[:,0].tolist()
    ct = datetime.datetime.utcnow()

    # This slicing operation gets the name of the coins from content['Key'] array
    coin_name = content['Key'].split('/')[1].split('.')[0]
    dict_data = {
         "coin_name" : coin_name,
         "timestamp": ct.isoformat(),
         "prediction_price_list": json_arr,
         "mse": f'{train_score}',
         "rmse": f'{np.sqrt(train_score)}'
     }
    
    # creates a json file
    with open(f'{coin_name}.json', "w") as f:
        json.dump(dict_data, f)

    filename = f'{coin_name}.json'
    object_key =  filename

    # uploads the file into an different s3 bucket
    s3_client.upload_file(object_key, dst_bucket_name, filename)

    print(predicted_days)
    print("Works fine!!")


   



