from pycoingecko import CoinGeckoAPI
cg = CoinGeckoAPI()

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