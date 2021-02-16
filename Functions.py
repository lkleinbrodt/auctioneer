from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import tensorflow as tf
from dateutil.relativedelta import relativedelta
import alpaca_trade_api as alp
from sklearn.preprocessing import MinMaxScaler
import os
import pandas_market_calendars as mcal
import logging

def IdentifyStocksOfInterest(all = False):
    #symbols = pd.read_csv('symbols.csv')['Symbol'].tolist()
    with open('./Data/energy_tickers.txt', 'r') as f:
        symbols = f.read().split('\n')
    
    if all:
        return symbols
    else:
        return np.random.choice(symbols, 10, replace = False).tolist()

def GetHistoricalData(symbols, api, end_date = datetime.now(), n_data_points = 200_000):
    symbols_to_pull = np.unique(symbols)
    
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)

    data = pd.DataFrame()

    if n_data_points < 2000:
        time_step = 1
    elif n_data_points < 10_000:
        time_step = 10
    else:
        time_step = 50

    while data.shape[0] < n_data_points:
        start = end_date - relativedelta(days=time_step)
        data_list = []
        for sym in symbols_to_pull:
            quotes = api.polygon.historic_agg_v2(symbol = sym, multiplier = 1, timespan = 'minute', 
                                                 _from = start, to = end_date, limit = 50_000).df
            quotes = quotes[['close']]
            quotes.rename(columns = {'close': sym}, inplace = True)
            data_list.append(quotes)
        batch_df = pd.concat(data_list, axis = 1)
  
        data = pd.concat([data, batch_df], axis = 0)
        end_date = start
    
    data.sort_index(inplace = True)
    data = data.fillna(method = 'ffill')
    data = data.fillna(method = 'bfill')

    data = data.tail(n_data_points)

    bad_cols = data.columns[data.isna().sum() > 0]
    if len(bad_cols) > 0:
        data = data.drop(bad_cols, axis = 1)
        logging.debug('Skipping {}, had missing values for the period.'.format(list(bad_cols)))

    return data

def GetDayQuotes(symbols, api, date, open_or_close = 'open'):
    
    if isinstance(date, str):
        date = pd.to_datetime(date)
    if date is None:
        date = datetime.now()
        
    all_quotes = []
    for sym in symbols:
        quotes = api.polygon.historic_agg_v2(symbol = sym, multiplier = 1, timespan = 'day', _from = date, to = date).df
        quotes = quotes[[open_or_close]]
        quotes.rename(columns={open_or_close: sym}, inplace = True)
        all_quotes.append(quotes)
    data = pd.concat(all_quotes, axis = 1).head(1)
    bad_cols = data.columns[data.isna().sum() > 0]
    if len(bad_cols) > 0:
        data = data.drop(bad_cols, axis = 1)
        logging.debug('Skipping {}, had missing values for the period.'.format(list(bad_cols)))
    return dict(data.iloc[0])

#%%

def GetLongReturns(symbols_to_consider, api, start, end):

    starting_prices = GetDayQuotes(symbols_to_consider, api, start)
    ending_prices = GetDayQuotes(symbols_to_consider, api, end)

    valid_symbols = [sym for sym in symbols_to_consider if sym in starting_prices.keys() and ending_prices.keys()]
    invalid_symbols = [sym for sym in symbols_to_consider if sym not in valid_symbols]

    if len(invalid_symbols) > 0:
        logging.debug('Skipping {}, had missing values for the period.'.format(list(bad_cols)))

    returns = pd.DataFrame({'Start': starting_prices, 'End': ending_prices})
    returns['Return'] = returns['End'] / returns['Start']
    returns['ReturnRank'] = returns['Return'].rank(ascending=False)

    top_fund_return = returns[returns['ReturnRank']==1]['Return'][0]
    top_fund = returns[returns['ReturnRank']==1].index[0]
    top_five_return = np.mean(returns[returns['ReturnRank']<6]['Return'])
    top_five = list(returns[returns['ReturnRank']<6].index)
    weighted_return = np.mean(returns['Return'])

    logging.info('Top Fund: ' + top_fund)
    logging.info('Top Fund Return: {}'.format(top_fund_return))
    logging.info('Top 5 Funds: {}'.format(top_five))
    logging.info('Top 5 Fund Return: {}'.format(top_five_return))
    logging.info('Overall weighted Return: {}'.format(weighted_return))

    return returns

def CreateEncoderModel(history_steps, target_steps, n_features):
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

def GenerateWindowData(df, HISTORY_STEPS, TARGET_STEPS, train_test_split = .9):
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
        n_train_samples = np.int(df.shape[0] * train_test_split)
        train, test = df[:n_train_samples], df[n_train_samples:]
        
        for col in train.columns:
            scaler = MinMaxScaler(feature_range=(-1,1))
            norm = scaler.fit_transform(train[col].values.reshape(-1,1))
            norm = np.reshape(norm, len(norm))
            scalers[col] = scaler
            train[col] = norm

        for col in train.columns:
            scaler = scalers[col]
            norm = scaler.transform(test[col].values.reshape(-1,1))
            norm = np.reshape(norm, len(norm))
            test[col] = norm
        
        X_train, Y_train = split_series(train.values, HISTORY_STEPS, TARGET_STEPS)
        X_test, Y_test = split_series(test.values, HISTORY_STEPS, TARGET_STEPS)
        return X_train, Y_train, X_test, Y_test, scalers
    else: 
        for col in df.columns:
            scaler = MinMaxScaler(feature_range=(-1,1))
            norm = scaler.fit_transform(df[col].values.reshape(-1,1))
            norm = np.reshape(norm, len(norm))
            scalers[col] = scaler
            df[col] = norm
        X_df, Y_df = split_series(df.values, HISTORY_STEPS, TARGET_STEPS)
        return X_df, Y_df, scalers

def TrainEncoderModel(df, HISTORY_STEPS, TARGET_STEPS, MAX_EPOCHS, BATCH_SIZE, LEARNING_RATE):
    
    X_train, Y_train, X_test, Y_test, scalers = GenerateWindowData(df, HISTORY_STEPS, TARGET_STEPS)
    n_features = X_train.shape[2]
    
    model = CreateEncoderModel(HISTORY_STEPS, TARGET_STEPS, n_features)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        LEARNING_RATE, 
        decay_steps = int(X_train.shape[0] / BATCH_SIZE) * 2, 
        decay_rate = .96
    )

    model.compile(
        optimizer = tf.keras.optimizers.Adam(), 
        loss = tf.keras.losses.Huber(),
        learning_rate = lr_schedule
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(patience = 5)
    model_checkpoints = tf.keras.callbacks.ModelCheckpoint('models/checkpoints/', save_best_only = True, save_weights_only = True)
    log_callback = tf.keras.callbacks.CSVLogger('Logs/backtest_log.log', append = True)
    my_callbacks = [early_stopping, model_checkpoints, log_callback]

    model.fit(
        X_train, Y_train, 
        epochs = MAX_EPOCHS, 
        validation_data = (X_test, Y_test), 
        batch_size = BATCH_SIZE, 
        callbacks = my_callbacks,
        verbose = 1
        )

    model.load_weights('models/checkpoints/')
    model.save_weights('models/TrainedModel')

    return model, scalers

def CheckHoliday(date):
    nyse = mcal.get_calendar('NYSE')
    while date.isoweekday() > 5 or date in nyse.holidays().holidays:
        date += timedelta(days = 1)
    return date

############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED
############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED
############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED
############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED

class WindowGenerator():
    def __init__(self, input_width, label_width, shift, data, label_columns=None):
        
        #Index the labels (and all columns)
        self.label_columns = label_columns
        if label_columns is not None:
           self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i,name in enumerate(data.columns)}
        
        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
    
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis = -1)
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels
    
    def plot(self, plot_col, model=None, max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(3, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)
            
            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index
            
            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                          marker='X', edgecolors='k', label='Predictions',
                          c='#ff7f0e', s=64)
            if n == 0:
                plt.legend()
        plt.xlabel('Time [d]')
    
    def make_dataset(self, data, batch_size = 32):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
           # shuffle=shuffle, shuffle now down in fit call
            batch_size=batch_size
        )
        ds = ds.map(self.split_window)
        
        return ds


class FeedBack(tf.keras.Model):
    def __init__(self, units, out_steps, num_features):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.num_features = num_features
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_features)
    
    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x , *state = self.lstm_rnn(inputs)
        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state
        
    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the lstm state
        prediction, state = self.warmup(inputs)

        # Insert the first prediction
        predictions.append(prediction)

        # Run the rest of the prediction steps
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state, training=training)
            x, state = self.lstm_cell(x, states = state, training = training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions
    
def CreateModel(data):
    input_width = 30
    label_width = 1
    
    model = FeedBack(16, label_width, data.shape[1])
    model.compile(loss = tf.losses.MeanSquaredError(), optimizer = tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
    
    return model
    
def TrainModel(model, target_symbol, input_data, epochs):
    input_width = 30
    label_width = 1
    # perhaps just normalize? no differencing?
    diff_data = input_data.diff()
    diff_data.iloc[0,:] = 0

    train_df = diff_data[:-label_width*3]
    val_df = diff_data[-(input_width+label_width*3):]

    window = WindowGenerator(input_width, label_width, label_width, train_df, label_columns=[target_symbol])

    train_ds = window.make_dataset(train_df, batch_size=32)
    val_ds = window.make_dataset(val_df, batch_size = label_width)

    early_stop = tf.keras.callbacks.EarlyStopping(patience = 10)

    checkpoints = tf.keras.callbacks.ModelCheckpoint('./models/checkpoint', save_best_only=True, save_weights_only=True)
    history = model.fit(train_ds, validation_data=val_ds, shuffle=True, validation_steps=1, callbacks = [early_stop, checkpoints], epochs = epochs, verbose = 0)

    model.load_weights('./models/checkpoint')
    return model
    

def Predict7DayHigh(model, target_symbol, input_data):

    pred_data = np.reshape(np.array(diff_data[-input_width:]), newshape = (1, input_width, -1))
    next_7_days = model.predict(tf.constant(pred_data)).reshape(label_width, -1)
    symbol_seven_day_high = np.max(np.cumsum(next_7_days[:,diff_data.columns == target_symbol]))

    return (target_symbol, symbol_seven_day_high)

class WindowGenerator():
    def __init__(self, input_width, label_width, shift, data, label_columns=None):
        
        #Index the labels (and all columns)
        self.label_columns = label_columns
        if label_columns is not None:
           self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i,name in enumerate(data.columns)}
        
        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
    
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis = -1)
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels
    
    def plot(self, plot_col, model=None, max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(3, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)
            
            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index
            
            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                          marker='X', edgecolors='k', label='Predictions',
                          c='#ff7f0e', s=64)
            if n == 0:
                plt.legend()
        plt.xlabel('Time [d]')
    
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            #shuffle=shuffle, shuffle now done at training
            batch_size=32
        )
        ds = ds.map(self.split_window)
        
        return ds


def ProcessData(input_data, columns_of_interest):
    df = input_data.dropna()
    ori_df = df.copy()
    dates = pd.to_datetime(df.pop('begins_at'))
    
    #keep only the close price columns
    #df = df[[col for col in df.columns if 'close_price' in col]]
    df = df[columns_of_interest]
    
    #Difference data and remove introduced NA rows
    df = df.diff()
    df = df[1:]
    ori_df = ori_df[1:]
    dates = dates[1:]
    
    ori_df.set_index(dates, inplace = True)
    df.set_index(dates, inplace = True)
    
    return df, ori_df

def TrainSplit(input_data, n_months_val, n_months_test):
    n_months = len(np.unique(input_data.index.strftime('%m/%Y')))
    val_end = np.max(input_data.index) - relativedelta(months = n_months_test)
    train_end = val_end - relativedelta(months = n_months_val)
    
    train_df = input_data[input_data.index <= train_end]
    val_df = input_data[(input_data.index > train_end) & (input_data.index <= val_end)]
    test_df = input_data[input_data.index > val_end]
    
    print('Train Period: ' + str(np.min(train_df.index).strftime('%Y-%m-%d')) + ' to ' + str(np.max(train_df.index).strftime('%Y-%m-%d')) + '\n' + 
          'Validation Period: ' + str(np.min(val_df.index).strftime('%Y-%m-%d')) + ' to ' + str(np.max(val_df.index).strftime('%Y-%m-%d')) + '\n' +
          'Testing Period: ' + str(np.min(test_df.index).strftime('%Y-%m-%d')) + ' to ' + str(np.max(test_df.index).strftime('%Y-%m-%d')))
    return train_df, val_df, test_df


def compile_and_fit(model, window, epochs, patience=2, verbose = 0):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')
    model.compile(loss=tf.losses.MeanSquaredError(), 
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    
    history = model.fit(window.train, epochs = epochs,
                        validation_data=window.val,
                        verbose = verbose
                        #callbacks=[early_stopping]
                       )
    return history


def TrainModel(model, window, epochs = 50, patience = 5, verbose = 0, return_history = False):
    
    history = compile_and_fit(model, window, epochs, patience, verbose)
    
    if return_history:
        return model, history
    else:
        return model




def Turtle7DayTest(model, window, eval_dataset, pred_dataset, pred_column, starting_cash, test_start):
    
    all_predictions = model.predict(window.make_dataset(pred_dataset, shuffle = False))
    seven_day_deltas = [0 for i in range(len(pred_dataset) - len(all_predictions)-1)] + [np.max(np.cumsum(pred)) for pred in all_predictions] + [0]
    
    eval_df = eval_dataset[[pred_column]].copy()
    eval_df['SevenDayHigh'] = seven_day_deltas
    cash_reserves = starting_cash
    eval_df['Cash'] = 0
    eval_df['Cost'] = 0
    eval_df['Revenue'] = 0
    eval_df['Shares'] = 0
    eval_df['PortfolioValue'] = 0
    eval_df = eval_df[eval_df.index > test_start]
    
    for i in range(len(eval_df)-1):
        max_delta = eval_df['SevenDayHigh'].iloc[i]
        eval_df['Cash'].iloc[i] = cash_reserves
        if max_delta > 0:
            if eval_df[pred_column].iloc[i] < cash_reserves:
                eval_df['Cost'].iloc[i] = eval_df[pred_column].iloc[i]
                eval_df['Shares'].iloc[i+1] = eval_df['Shares'].iloc[i] + 1
                cash_reserves -= eval_df[pred_column].iloc[i]
            else:
                eval_df['Shares'].iloc[i+1] = eval_df['Shares'].iloc[i]
        elif eval_df['Shares'].iloc[i] > 0:
            eval_df['Revenue'].iloc[i] = eval_df[pred_column].iloc[i]
            eval_df['Shares'].iloc[i+1] = eval_df['Shares'].iloc[i] - 1
            cash_reserves += eval_df[pred_column].iloc[i]

        eval_df['PortfolioValue'].iloc[i] = (eval_df[pred_column].iloc[i] * eval_df['Shares'].iloc[i])
        
    final_cost = np.sum(eval_df['Cost'])
    final_revenue = np.sum(eval_df['Revenue'])
    final_assets = eval_df.iloc[-2]['PortfolioValue']
    final_cash = cash_reserves
    
    final_return = final_assets + final_cash #- final_cost 
    long_term_return = (eval_df[pred_column].iloc[-2] / eval_df[pred_column].iloc[0]) * starting_cash
    profit_score = final_return  / long_term_return
    
    print('Model Strategy Return: ' + str(final_return))
    print('Long Term Strategy Return: ' + str(long_term_return))
    print('Profit Score: ' + str(profit_score))


class FeedBack(tf.keras.Model):
    def __init__(self, units, out_steps, num_features):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.num_features = num_features
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_features)
    
    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x , *state = self.lstm_rnn(inputs)
        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state
        
    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the lstm state
        prediction, state = self.warmup(inputs)

        # Insert the first prediction
        predictions.append(prediction)

        # Run the rest of the prediction steps
        for n in range(1, self.out_steps):
         # Use the last prediction as input.
          x = prediction
          # Execute one lstm step.
        x, state = self.lstm_cell(x, states=state,
                                training=training)
        # Convert the lstm output to a prediction.
        prediction = self.dense(x)
        # Add the prediction to the output
        predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions


CONV_WIDTH = 3
OUT_STEPS = 7

one_shot_lstm_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units]
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(4, return_sequences=False),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    #tf.keras.layers.Reshape([OUT_STEPS, 1])
])

multi_dense_model = tf.keras.Sequential([
    # Take the last time step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, dense_units]
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation = 'relu'),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, 1])
])