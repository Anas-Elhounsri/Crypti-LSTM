from pycoingecko import CoinGeckoAPI
import  boto3
import csv
from datetime import datetime

cg = CoinGeckoAPI()

def getCoinsList():
    return cg.getCoinsList()

#this is the lamda function that receives trigger events 
#(now its manual i will automate the process with API GateWay HTTP request)
#The function's argument is provided by AWS automatically whenever its triggered

def lambda_handler(event, context):
    bitcoin_data = getCoinData("bitcoin")

    prices = bitcoin_data[0]
    new_prices = []
    market_caps = bitcoin_data[1]
    new_market_caps = []
    volumes = bitcoin_data[2]
    new_volumes = []
    timeStamps = []

    for price in prices:
        date = datetime.fromtimestamp(price[0]/1000)
        timeStamps.append(date.strftime("%D %T"))
        new_prices.append(price[1])

    for market_cap in market_caps:
        new_market_caps.append(market_cap[1])

    for volume in volumes:
        new_volumes.append(volume[1])

    rows = zip(timeStamps, new_prices, new_market_caps, new_volumes)
    fields = [ "TimeStamp", "Price", "Market Cap", "Volume"]

    csv_string =""

    for i in fields:
        csv_string += i + ","
    for i in rows:
        #ik fancy stuff ;)
        csv_string += f"{i[0]}, { i[1]}, {i[2]} , {i[3]}\n"

    #this saves the CSV data into the s3
    #defines the client to be s3
    s3 = boto3.client('s3')
    s3.put_object(
        Bucket='crypti-data',
        Key='coin-market-data/{}.csv'.format("bitcoin"),
        Body=csv_string
        )

    return {
        "statusCode": 200,
        "body" : "Data saved to s3"
    }

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