import logging
import warnings
import os
import time

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from datetime import datetime
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import alpaca_trade_api as alp
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.objective_functions import L2_reg
from pypfopt.discrete_allocation import DiscreteAllocation
from Functions import *


logging.basicConfig(
    filename="Logs/MVP/backtest_log.log",
    level="DEBUG",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


with open("Data/paper_api_keys.txt") as api_file:
    api_keys = api_file.read().replace("\n", "").split(",")
    alpaca_api = {a.split("=")[0]: a.split("=")[1] for a in api_keys}

api = alp.REST(
    key_id=alpaca_api["APCA_API_KEY_ID"],
    secret_key=alpaca_api["APCA_API_SECRET_KEY"],
    base_url=alpaca_api["APCA_API_BASE_URL"],
)


########## PARAMETERS ##############

START_DATE = "2017-01-01"
END_DATE = "2017-03-31"

COV_HISTORY = 20_000  # in minutes -> 60 * 8 * 90

STARTING_CASH = 100_000
MAX_CONCURRENT_SECURITIES = 20
FUND_UNIVERSE = IdentifyStocksOfInterest(method="SP")
N_NEW_PER_DAY = 5
REFRESH_PERIOD = 7

########## END PARAMETERS #############

logging.info("Starting Script: ")
logging.info(datetime.now())

# Get Alpaca APIs

### Get portfolio
###
# portfolio = pd.DataFrame([{'Symbol': p.symbol, 'Quantity': float(p.qty), 'Value': float(p.market_value)} for p in api.list_positions()])

portfolio = pd.DataFrame(columns=["Symbol", "Quantity", "Value"])
portfolio["Symbol"] = portfolio["Symbol"].astype(str)
portfolio["Quantity"] = portfolio["Quantity"].astype(float)
portfolio["Value"] = portfolio["Value"].astype(float)


### Define One Day of trading behavior, should take in a date, and output the actions that will be taken


def TradingDay(current_day, portfolio, buying_power, api):
    portfolio_stocks = portfolio["Symbol"].tolist()
    if len(portfolio_stocks) > MAX_CONCURRENT_SECURITIES:
        soi = portfolio_stocks
    else:
        new_stocks = np.random.choice(
            FUND_UNIVERSE, N_NEW_PER_DAY, replace=False
        ).tolist()
        soi = list(set(portfolio_stocks + new_stocks))

    logging.debug("Pulling Data")
    stock_prices = GetHistoricalData(
        soi,
        api,
        end_date=current_day,
        n_data_points=COV_HISTORY,
        time_interval="minute",
    )

    current_prices = {
        stock_prices.columns[i]: stock_prices.iloc[-1, i]
        for i in range(stock_prices.shape[1])
    }

    logging.debug("Calculating Covariance")
    cov_matrix = CovarianceShrinkage(stock_prices).ledoit_wolf()

    logging.debug("Calculating Efficient Frontier")
    ef = EfficientFrontier(None, cov_matrix, weight_bounds=(0, 1))

    logging.debug("Minimizing Volatility")
    ef.min_volatility()
    cleaned_weights = ef.clean_weights()

    ### Reconcile Current Portfolio with Optimal Portfolio
    portfolio_value = buying_power + np.sum(portfolio["Value"])
    da = DiscreteAllocation(cleaned_weights, pd.Series(current_prices), portfolio_value)
    new_allocation, _ = da.lp_portfolio()
    for stock in [
        stock
        for stock in portfolio["Symbol"].tolist()
        if stock not in new_allocation.keys()
    ]:
        new_allocation[stock] = 0
    logging.info("Target Allocation")
    logging.info(new_allocation)
    orders = dict([])
    for symbol in new_allocation.keys():
        target_quantity = new_allocation[symbol]

        if symbol in list(portfolio["Symbol"]):
            current_quantity = portfolio[portfolio["Symbol"] == symbol]["Quantity"][0]
        else:
            current_quantity = 0

        if target_quantity > current_quantity:
            side = "buy"
            quantity = target_quantity - current_quantity
        elif target_quantity < current_quantity:
            side = "sell"
            quantity = current_quantity - target_quantity
        else:
            continue

        orders[symbol] = {"Side": side, "Quantity": quantity}

    return orders, current_prices


def Execution(day_of_order, orders, portfolio, buying_power):
    if not isinstance(day_of_order, pd.Timestamp):
        day_of_order = pd.Timestamp(day_of_order, tz="America/New_York")

    ###First, ensure your dates are valid
    calendar = api.get_calendar(
        start=day_of_order, end=day_of_order + relativedelta(days=5)
    )

    assert calendar[0].date == day_of_order

    ### Now determine which day these trades will be placed
    day_of_execution = calendar[1].date

    prices_to_pull = list(set(list(orders.keys()) + list(portfolio["Symbol"])))
    next_day_prices = dict(GetDayQuotes(prices_to_pull, api, day_of_execution, "open"))
    orders = {
        symbol: orders[symbol]
        for symbol in orders.keys()
        if symbol in next_day_prices.keys()
    }

    sell_symbols = [
        symbol for symbol in orders.keys() if orders[symbol]["Side"] == "sell"
    ]
    buy_symbols = [
        symbol for symbol in orders.keys() if orders[symbol]["Side"] == "buy"
    ]

    for symbol in sell_symbols:
        quantity = orders[symbol]["Quantity"]
        price = next_day_prices[symbol]
        cost = quantity * price

        logging.debug(
            "Selling {} shares of {} at {} per share for a total sale of {}.".format(
                quantity, symbol, np.round(price, 5), np.round(cost, 5)
            )
        )
        buying_power += cost
        portfolio.loc[(portfolio["Symbol"] == symbol), "Quantity"] -= quantity
        logging.debug("New buying power: " + str(buying_power))

    for symbol in buy_symbols:
        quantity = orders[symbol]["Quantity"]
        price = next_day_prices[symbol]
        cost = quantity * price
        logging.debug(
            "Trying to buy {} shares of {} at {} per share for a total cost of {}".format(
                quantity, symbol, np.round(price, 5), np.round(cost, 5)
            )
        )

        while cost > buying_power:
            logging.debug(
                "Cannot buy, total cost is {} and we only have {}".format(
                    cost, buying_power
                )
            )
            quantity -= 1
            cost = quantity * price
        if quantity == 0:
            logging.debug("not enough cash to buy a single share")
            continue
        else:
            buying_power -= cost
            if symbol in list(portfolio["Symbol"]):
                portfolio.loc[portfolio["Symbol"] == symbol, "Quantity"] += quantity
            else:
                portfolio = portfolio.append(
                    pd.DataFrame(
                        [
                            {
                                "Symbol": symbol,
                                "Quantity": quantity,
                                "Value": quantity * price,
                            }
                        ]
                    )
                )
            logging.debug(
                "Successfully bought {} shares of {} at {} per share for a total cost of {}".format(
                    quantity, symbol, np.round(price, 5), np.round(cost, 5)
                )
            )
            logging.debug("New buying power: " + str(buying_power))
        portfolio.loc[portfolio["Symbol"] == symbol, "Value"] = (
            price * portfolio.loc[portfolio["Symbol"] == symbol, "Quantity"]
        )

    portfolio = portfolio[portfolio["Quantity"] != 0]

    for symbol in next_day_prices.keys():
        portfolio.loc[portfolio["Symbol"] == symbol, "Value"] = (
            next_day_prices[symbol]
            * portfolio.loc[portfolio["Symbol"] == symbol, "Quantity"]
        )

    return next_day_prices, portfolio, buying_power


####################
####################
#################### BEGIN BACKTESTING
####################
####################

day = pd.to_datetime(START_DATE)
end_day = pd.to_datetime(END_DATE)

order_history = []
price_history = []
portfolio_history = []
value_history = []
buying_power = STARTING_CASH

while day < end_day:
    logging.info("-----------------------------------------------------------")
    logging.info("-----------------------------------------------------------")
    day_start = datetime.now()

    day_ts = pd.Timestamp(day, tz="America/New_York")
    calendar = api.get_calendar(
        start=day_ts.isoformat(), end=(day_ts + relativedelta(days=1)).isoformat()
    )

    if calendar[0].date != day:
        logging.info(
            "Skipping {}, the market is closed.".format(day.strftime("%Y-%m-%d"))
        )

        day = day + relativedelta(days=1)
        continue

    logging.info("Working on {}".format(day.strftime("%Y-%m-%d")))

    orders, prices = TradingDay(day, portfolio, buying_power, api)

    logging.debug("----Closing prices for the day:" + str(day))
    logging.debug(prices)

    logging.info("----Orders for next day open:")
    logging.info(orders)

    if len(orders) == 0:
        logging.info("No orders for the day")
        day = day + relativedelta(days=1)
        continue

    next_day_prices, portfolio, buying_power = Execution(
        day, orders, portfolio, buying_power
    )

    logging.debug("----Prices at Open of Next Day:")
    logging.debug(next_day_prices)
    logging.info("-----Portfolio:")
    logging.info(portfolio)
    logging.info("Buying Power {}".format(buying_power))
    portfolio_value = buying_power + np.sum(portfolio["Value"])
    logging.info("Portfolio Value {}".format(buying_power + np.sum(portfolio["Value"])))

    order_history.append(orders)
    price_history.append(prices)
    portfolio_history.append(portfolio)
    value_history.append(portfolio_value)

    day = day + relativedelta(days=REFRESH_PERIOD)
    day_end = datetime.now()

    # API LIMIT is 200 per minute
    if (day_end - day_start).total_seconds() < 30:
        time.sleep(60)

logging.info("Final buying power: {}".format(buying_power))
logging.info("Final asset value: {}".format(np.sum(portfolio["Value"])))
logging.info(
    "Final portfolio value: {}".format(buying_power + np.sum(portfolio["Value"]))
)
logging.info(
    "Final rate of return: {}".format(
        (buying_power + np.sum(portfolio["Value"])) / STARTING_CASH
    )
)

logging.info(portfolio)

logging.info("-----------------------------------------------------------")
logging.info("----------Baseline Return")
logging.info("-----------------------------------------------------------")

baseline = GetLongReturns(
    IdentifyStocksOfInterest(method="sp"), api, START_DATE, END_DATE
)

logging.info(baseline)

logging.info(datetime.now())


import pickle as pk

with open("./Logs/MVP/order_history", "wb") as f:
    pk.dump(order_history, f)

with open("./Logs/MVP/price_history", "wb") as f:
    pk.dump(price_history, f)

with open("./Logs/MVP/portfolio_history", "wb") as f:
    pk.dump(portfolio_history, f)
