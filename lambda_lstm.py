import json
from keras.models import load_model
import numpy as np
import csv
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

data_op = pd.read_csv("BTC-USD.csv", usecols = [1])

def lambda_handler(event, context):

    lstm_model = load_model("crypti-lstm.h5")
    data_op = data_op.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range = (0,1))
    data_op = scaler.fit_transform(data_op)
    
    last_thirty_days = data_op[-30:]
    predicted_days = []
    count = 0

    last_thirty_days = data_op[-30:]
    predicted_days = []
    count = 0

    for i in range(30):

        #reshapes the data accordingly for the LSTM model e.g[[['a'],['b'],['c']]]
        last_thirty_days = last_thirty_days.reshape(1,1,30)
        #predicts the next 30 days
        predictions = lstm_model.predict(last_thirty_days)
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

    csv_file = "output.csv" 
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Prices"])
        for row in predicted_days:
            writer.writerow(row)

    return {
        'statusCode': 200,
        'body': json.dumps(predicted_days)
    }


