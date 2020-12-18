from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from dateutil.relativedelta import relativedelta
import alpaca_trade_api as alp

def IdentifyStocksOfInterest():
    ###For now, will just return 10 cached tech stocks
    
    #symbols = pd.read_csv('symbols.csv')['Symbol'].tolist()
    with open('./energy_tickers.txt', 'r') as f:
        symbols = f.read().split('\n')
    
    return symbols[10:19]

def GetHistoricalData(symbols, api, end, start = None, open_or_close = 'open'):
    ###For now, we will explicitly say what the relevant stocks are,
    ###but in future this should be automated
    symbols_to_pull = np.unique(symbols)
    
    if isinstance(end, str):
        end = pd.to_datetime(end)
        
    if end is None:
        end = datetime.now()
    
    ### We will use 5 years of historical daily data, meaning we need ~ 1,265 trading days of data,
    n_days = 1265
    if start == None:
        start = end - relativedelta(days = n_days)
        
    all_quotes = []
    for sym in symbols_to_pull:
        quotes = api.polygon.historic_agg_v2(symbol = sym, multiplier = 1, timespan = 'day', _from = start, to = end).df
        quotes = quotes[['close']]
        quotes.rename(columns={'close': sym}, inplace = True)
        all_quotes.append(quotes)
    data = pd.concat(all_quotes, axis = 1)
        
    bad_cols = data.columns[data.isna().sum() > 0]
    if len(bad_cols) > 0:
        data = data.drop(bad_cols, axis = 1)
        print('Skipping {}, had missing values for the period.'.format(list(bad_cols)))
        
    return data

def GetDayQuotes(symbols, api, date, open_or_close = 'open'):
    
    if isinstance(date, str):
        date = pd.to_datetime(end)
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
        print('Skipping {}, had missing values for the period.'.format(list(bad_cols)))
    return dict(data.iloc[0])

def GetLongReturns(symbols_to_consider, api, start, end):

    starting_prices = GetDayQuotes(symbols_to_consider, api, start)
    ending_prices = GetDayQuotes(symbols_to_consider, api, end)

    valid_symbols = [sym for sym in symbols_to_consider if sym in starting_prices.keys() and ending_prices.keys()]
    invalid_symbols = [sym for sym in symbols_to_consider if sym not in valid_symbols]

    if len(invalid_symbols) > 0:
        print('Skipping {}, had missing values for the period.'.format(list(bad_cols)))

    returns = pd.DataFrame({'Start': starting_prices, 'End': ending_prices})
    returns['Return'] = returns['End'] / returns['Start']
    returns['ReturnRank'] = returns['Return'].rank(ascending=False)

    top_fund_return = returns[returns['ReturnRank']==1]['Return'][0]
    top_fund = returns[returns['ReturnRank']==1].index[0]
    top_five_return = np.mean(returns[returns['ReturnRank']<6]['Return'])
    top_five = list(returns[returns['ReturnRank']<6].index)
    weighted_return = np.mean(returns['Return'])

    print('Top Fund: ' + top_fund)
    print('Top Fund Return: {}'.format(top_fund_return))
    print('Top 5 Funds: {}'.format(top_five))
    print('Top 5 Fund Return: {}'.format(top_five_return))
    print('Overall weighted Return: {}'.format(weighted_return))

    return returns


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