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
src_bucket_name = 'crypti-hist'
dst_bucket_name = 'crypti-predict'
# selects the S3 bucket
response = s3_client.list_objects(Bucket=src_bucket_name, Prefix='coin-market-data/')

for content in response['Contents']:
    # reads the file
    coin_name = content['Key'].split('/')[1].split('.')[0]
    print(coin_name)
    dict_data = {"coin_name" : coin_name,}
    print(dict_data)
