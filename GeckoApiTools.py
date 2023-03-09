from pycoingecko import CoinGeckoAPI
import  boto3
import csv

cg = CoinGeckoAPI()
s3 = boto3.client('s3')#defines the client to be s3

def getCoinsList():
    return cg.getCoinsList()

def getCoinData(coin_name, fiat_currency="usd", days="max"):
    coin_market_data = cg.get_coin_market_chart_by_id(
        coin_name,
        fiat_currency,
        days,
    )

    prices = coin_market_data["prices"]
    market_caps = coin_market_data["market_caps"]
    volumes = coin_market_data["total_volumes"]

    return prices, market_caps, volumes

#this is the lamda function that receives trigger events 
#(now its manual i will automate the process with API GateWay HTTP request)
#The function's argument is provided by AWS automatically whenever its triggered

def lambda_handler(event, context):
    bitcoin_data = getCoinData("bitcoin")
    prices = bitcoin_data[0]
    market_caps = bitcoin_data[1]
    volumes = bitcoin_data[2]

    #creates an iterator that combines prices, market_caps and volumes intp tuples (I read its better and easier open for suggestions)
    #resulting in an object that makes it easier to iterate over 
    data = zip(prices, market_caps, volumes)

    #Creates a CSV file and fills it with data with for loop
    csv_string =""
    for i in data:
        #string concatination ik fancy stuff ;)
        csv_string += f"{i[0]}, { i[1]}, {i[2]} \n"

    #this saves the CSV data into the s3
    bucket_name = "crypti-food"
    object_key = "coin-data/bitcoin.csv"
    s3.put_object(Bucket=bucket_name, Key=object_key, Body=csv_string)

    return {
        'statuCode': 200,
        'body' : 'Data saved to s3' 
    }