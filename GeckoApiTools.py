from pycoingecko import CoinGeckoAPI
import  boto3
import csv
from datetime import datetime

cg = CoinGeckoAPI()

#this is the lamda function that receives trigger events 
#The function's argument is provided by AWS automatically whenever its triggered by Amazon Event Bridge     
def lambda_handler(event, context):

    #opens the txt file that has the list of coins
    with open("coins.txt" , "r") as f:
        content = f.readlines()

    #loops through the list of coins to call the API 10 times for each coin
    for coins in content:
        #strips out any space 
        n_coins = coins.rstrip()
        #calls the function to get necessary data
        coin_data = getCoinData(n_coins)
        prices = coin_data[0]
        new_prices = []
        market_caps = coin_data[1]
        new_market_caps = []
        volumes = coin_data[2]
        new_volumes = []
        timeStamps = []

        #creates a list of prices with their timestamps
        for price in prices:
            date = datetime.fromtimestamp(price[0]/1000)
            timeStamps.append(date.strftime("%D %T"))
            new_prices.append(price[1])

        #creates a list of market caps
        for market_cap in market_caps:
            new_market_caps.append(market_cap[1])

        #creates a list of volumes
        for volume in volumes:
            new_volumes.append(volume[1])

        #zips the three previously mentioned array into a dictionary
        rows = zip(timeStamps, new_prices, new_market_caps, new_volumes)
        fields = [ "TimeStamp", "Price", "Market Cap", "Volume"]

        csv_string =""

        #writes the headers in the csv file
        for i in fields:
            csv_string += i + ","
        for i in rows:
            csv_string += f"{i[0]}, { i[1]}, {i[2]} , {i[3]}\n"

        #this saves the CSV data into the s3
        #defines the client to be s3
        s3 = boto3.client('s3')
        s3.put_object(
            Bucket='s3-bucket-name',
            Key='object-name/{}.csv'.format(n_coins),
            Body=csv_string
            )

    #confirms whether the lambda function worked or not
    return {
        "statusCode": 200,
        "body" : "Data saved to s3"
    }

#function that requests the coingecko API the necessary data
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