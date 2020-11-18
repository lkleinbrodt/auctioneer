import matplotlib as mpl
import robin_stocks as r
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from dateutil.relativedelta import relativedelta

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