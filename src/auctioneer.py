import matplotlib.pyplot as plt
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.timeframe import TimeFrame
from alpaca.data.requests import CryptoBarsRequest
import pandas as pd
from datetime import datetime
import tensorflow as tf
import numpy as np

MODELS_PATH = '../data/models/'

class Portfolio:

    def __init__(self, balance, data, holdings = {}):
        self.balance = balance
        self.data = data
        self.holdings = holdings
        self.transaction_log = []

    def value(self, date = None):
        #only supports one security for now
        if date is not None:
            price = self.data.loc[date]['close']
        else:
            price = self.data.iloc[-1]['close']
        
        aum = 0
        for security, amount in self.holdings.items():
            aum += price * amount
        
        out = {
            'Cash': self.balance,
            'AUM': aum,
            'Total': self.balance + aum
        }

        return out

    def execute(self, date, order_dict):
        if (order_dict is None) | (order_dict == {}):
            return 
        #just goes in order of order
        for security, order in order_dict.items():
            price = self.data.loc[date]['open']


            if order['action'] == 'buy':

                if order['amount'] == 'max':
                    amount = self.balance // price
                else:
                    amount = order['amount']# // price
                
                if amount > 0:
                    if amount * price > self.balance:
                        amount = self.balance // price
                    self.holdings[security] = self.holdings.get(security, 0) + amount
                    self.balance -= amount * price
                    self.transaction_log += [{'date': date, 'action': 'buy', 'amount': amount, 'value': amount*price}]

            elif order['action'] == 'sell':

                if order['amount'] == 'max':
                    amount = self.holdings.get(security, 0)
                else:
                    #TODO: not right
                    amount = order['amount']# // price

                if self.holdings.get(security, 0) == 0:
                    continue
                if amount > self.holdings[security]:
                    amount = self.holdings[security]
                
                self.holdings[security] -= amount
                self.balance += amount * price
                self.transaction_log += [{'date': date, 'action': 'sell', 'amount': amount, 'value': amount*price}]

    def transaction_log_summary(self):
        buys = []
        sells = []
        for order in self.transaction_log:
            if order['action'] == 'buy':
                buys += [order['value']]
            if order['action'] == 'sell':
                sells += [order['value']]
        
        # print(f"""
        # {len(buys)+len(sells)} total transactions.
        # {len(buys)} buy orders, totalling {sum(buys)}.
        # {len(sells)} sell orders, totalling {sum(sells)}.
        # """)
        return {'buys': (len(buys), sum(buys)), 'sells': (len(sells), sum(sells))}

    def plot_transactions(self):
        transaction_log = self.transaction_log

        for t in transaction_log:
            if t['action'] == 'buy':
                color = 'green'
            else:
                color='red'
            plt.axvline(t['date'], color=color, alpha = .05)
            
        plt.plot(self.data['close'])
        plt.show()

def pull_data(security, start_date, end_date):
    client = CryptoHistoricalDataClient()
    request_params = CryptoBarsRequest(
        symbol_or_symbols=[security],
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date,
        limit = 10_000
    )

    bars_df = client.get_crypto_bars(request_params).df
    data = bars_df.droplevel('symbol').copy()

    return data




def pull_crypto_prices(symbols, start_date, end_date = None, timeframe = 'minute', client = None, column='close'):

    if client is None:
        client = CryptoHistoricalDataClient()
    start_date = pd.to_datetime(start_date)
    if end_date is None:
        end_date = pd.to_datetime(datetime.now())
    else:
        end_date = pd.to_datetime(end_date)
    
    if timeframe.lower() == 'minute':
        timeframe = TimeFrame.Minute
    elif timeframe.lower() == 'day':
        timeframe = TimeFrame.Day
    elif timeframe.lower() == 'month':
        timeframe = TimeFrame.Month
    else:
        raise ValueError(f"Unrecognized timeframe: {timeframe}")
    
    request_params = CryptoBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=timeframe,
        start=start_date,
        end=end_date,
        # limit = 10_000
    )

    bars_df = client.get_crypto_bars(request_params).df

    prices = bars_df[column]
    if isinstance(symbols, str) | len(symbols) == 1:
        print('dropping symbol index')
        prices = bars_df.droplevel('symbol')

    #TODO: cannot compare tz-naive and tz-aware timestamps
    # pstart = prices.index.get_level_values('timestamp').min()
    # pend = prices.index.get_level_values('timestamp').max()
    # if (prices.shape[0] < 10_000) | (pstart > start_date) | (pend < end_date):
    #     print(f"""
    #     Requested Start: {start_date}
    #     Returned Start: {pstart}
    #     Requested End: {end_date}
    #     Returned End: {pend}
    #     """)

    return prices

def get_crypto_symbols():
    symbols = ['BTC/USD','ETH/USD','DOGE/USD','SHIB/USD','MATIC/USD','ALGO/USD','AVAX/USD','LINK/USD','SOL/USD']
    return symbols

def pivot_price_data(price_data):
    """Pivots price data from tall to wide format, 
    filters data to only after first timestamp where all symbols have non-NA
    """
    wide_data = price_data.set_index('timestamp')
    wide_data = wide_data.pivot(columns='symbol', values = 'close')
    try:
        first_valid_index = wide_data.index[wide_data.isna().mean(axis = 1)==0][0]
    except IndexError as e:
        raise IndexError(f"No record where all symbols have valid data")
    wide_data = wide_data.loc[first_valid_index:]
    return wide_data

def window_data(df, HISTORY_STEPS, TARGET_STEPS, train_test_split = .9):
    def split_series(series, n_past, n_future):
        X, Y = list(), list()

        for window_start in range(len(series)):
            past_end = window_start + n_past
            future_end = past_end + n_future
            if future_end > len(series):
                break
            past = series[window_start:past_end, :]
            future = series[past_end:future_end, :]
            X.append(past)
            Y.append(future)
        return np.array(X), np.array(Y)
    
    df = df.sort_index().copy()
    n_features = df.shape[1]
    scalers = {}
    if train_test_split > 0:
        n_train_samples = int(df.shape[0] * train_test_split)
        train, test = df[:n_train_samples], df[n_train_samples:]
        
        for col in train.columns:
            pass
            # scaler = MinMaxScaler(feature_range=(-1,1))
            # norm = scaler.fit_transform(train[col].values.reshape(-1,1))
            # norm = np.reshape(norm, len(norm))
            # scalers[col] = scaler
            # train[col] = norm

        for col in train.columns:
            pass
            # scaler = scalers[col]
            # norm = scaler.transform(test[col].values.reshape(-1,1))
            # norm = np.reshape(norm, len(norm))
            # test[col] = norm
        
        X_train, Y_train = split_series(train.values, HISTORY_STEPS, TARGET_STEPS)
        X_test, Y_test = split_series(test.values, HISTORY_STEPS, TARGET_STEPS)
        return X_train, Y_train, X_test, Y_test, scalers
    else: 
        for col in df.columns:
            pass
            # scaler = MinMaxScaler(feature_range=(-1,1))
            # norm = scaler.fit_transform(df[col].values.reshape(-1,1))
            # norm = np.reshape(norm, len(norm))
            # scalers[col] = scaler
            # df[col] = norm
        X_df, Y_df = split_series(df.values, HISTORY_STEPS, TARGET_STEPS)
        return X_df, Y_df, scalers

def encoder_model(history_steps, target_steps, n_features):
    enc_inputs = tf.keras.layers.Input(shape = (history_steps, n_features))
    enc_out1 = tf.keras.layers.LSTM(16, return_sequences = True, return_state = True)(enc_inputs)
    enc_states1 = enc_out1[1:]

    enc_out2 = tf.keras.layers.LSTM(16, return_state = True)(enc_out1[0])
    enc_states2 = enc_out2[1:]

    dec_inputs = tf.keras.layers.RepeatVector(target_steps)(enc_out2[0])

    dec_l1 = tf.keras.layers.LSTM(16, return_sequences = True)(dec_inputs, initial_state = enc_states1)
    dec_l2 = tf.keras.layers.LSTM(16, return_sequences = True)(dec_l1, initial_state = enc_states2)

    dec_out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(dec_l2)

    model = tf.keras.models.Model(enc_inputs, dec_out)

    return model

def train_encoder_model(df, HISTORY_STEPS, TARGET_STEPS, MAX_EPOCHS, BATCH_SIZE, LEARNING_RATE):
    
    X_train, Y_train, X_test, Y_test, scalers = GenerateWindowData(df, HISTORY_STEPS, TARGET_STEPS)
    n_features = X_train.shape[2]
    
    model = encoder_model(HISTORY_STEPS, TARGET_STEPS, n_features)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        LEARNING_RATE, 
        decay_steps = int(X_train.shape[0] / BATCH_SIZE) * 2, 
        decay_rate = .96
    )
    
    model.compile(
        optimizer = tf.keras.optimizers.Adam(), 
        loss = tf.keras.losses.Huber(),
        # learning_rate = lr_schedule
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(patience = 5)
    model_checkpoints = tf.keras.callbacks.ModelCheckpoint(MODELS_PATH+'checkpoints/', save_best_only = True, save_weights_only = True)
    date = df.index.max().strftime('%Y%m%d')
    #[os.remove(os.path.join('Logs/Tensorboard', f)) for f in os.listdir('Logs/Tensorboard')]
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir = MODELS_PATH + 'Tensorboard/' + date)
    my_callbacks = [early_stopping, model_checkpoints]

    model.fit(
        X_train, Y_train, 
        epochs = MAX_EPOCHS, 
        validation_data = (X_test, Y_test), 
        batch_size = BATCH_SIZE, 
        callbacks = my_callbacks,
        verbose = 0
        )

    model.load_weights(MODELS_PATH+'checkpoints/')
    model.save(MODELS_PATH+'TrainedModel')

    return model, scalers

def predict_forward(data, model, history = None):
    #TODO: verify you dont need to scale by history steps
    #and it might be worth 
    if history is not None:
        inference_data = data[-history:]

    # for col in inference_data.columns:
    #     scaler = scalers[col]
    #     norm = scaler.transform(inference_data[col].values.reshape(-1,1))
    #     norm = np.reshape(norm, len(norm))
    #     inference_data[col] = norm

    inference_data = np.array(inference_data).reshape((1, inference_data.shape[0], -1))

    predictions = model.predict(inference_data).squeeze() 
    
    # for i, col in enumerate(data.columns):
    #     scaler = scalers[col]
    #     predictions[:,i] = scaler.inverse_transform(predictions[:,i].reshape(-1,1)).reshape(-1)

    return predictions

def long_return(data, starting_balance):
    starting_average = data.bfill().iloc[0].mean()
    ending_average = data.ffill().iloc[-1].mean()
    return (ending_average / starting_average) * starting_balance
    # long_portfolio = Portfolio(STARTING_BALANCE, data, {})
    # long_portfolio.execute(data.index[0], {SECURITY: {'action': 'buy', 'amount': 'max'}})
    # return long_portfolio.value()['Total']