#%%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from config import *
from s3 import S3Client

import numpy as np
import torch.optim as optim
import torch.utils.data as data

WINDOW_SIZE = 120
PREDICTION_WINDOW = 60
BATCH_SIZE = 64
OUTPUT_NAME = 'test'


logger = create_logger(__name__)

if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

logger.info(f'Using: {DEVICE}')

import os
if not os.path.exists(ROOT_DIR / 'data' / 'models'):
    os.makedirs(ROOT_DIR / 'data' / 'models')

def get_time_series():
    logger.info('Loading data...')
    s3 = S3Client()
    # df = pd.read_parquet(ROOT_DIR/'data/top_10/BTC-USD.parquet')
    # df = pd.read_parquet(ROOT_DIR/'data/BTC-USD_15m.parquet')
    df = s3.read_csv('BTC-USD_15m.csv')
    df = df.set_index('start').sort_index()

    #TODO: find missing time periods and fill
    
    time_series = df['close'].values.reshape(-1, 1)
    return time_series

def normalize_ts(ts):
    scaler = StandardScaler()
    return scaler.fit_transform(ts) * 10
    

# def get_time_series():
#     logger.info('Loading data...')

#     df = pd.read_parquet(ROOT_DIR/'data/btc_price.parquet')
#     df['start'] = pd.to_datetime(df['start'], unit='s')

#     df.sort_values('start', inplace=True)

#     # Normalize the 'price' column using MinMaxScaler
    # scaler = StandardScaler()
    # df['price_normalized'] = scaler.fit_transform(df['close'].values.reshape(-1, 1)) * 10
    
#     time_series = df['price_normalized'].values.astype('float32')
    
#     return time_series

def split_time_series_sequentially(time_series, train_frac = .85, val_frac = .1):
    logger.info('Splitting data...')
    
    train_size = int(len(time_series) * train_frac)
    val_size = int(len(time_series) * val_frac)

    train, val, test = np.split(time_series, [train_size, train_size+val_size])
    
    return train, val, test



def create_dataset(dataset, window_size=1, prediction_window=1):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-window_size-prediction_window):
        feature = dataset[i:i+window_size]
        target = dataset[i+window_size+prediction_window]
        X.append(feature)
        y.append(target)
    X = torch.tensor(np.array(X), dtype=torch.float32)
    X = X.view(-1, window_size).to(DEVICE)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    y = y.view(-1, 1).to(DEVICE)
    
    return X,y

import random


def create_random_datasets(time_series, window_size=1, prediction_window=1, val_frac=.15):
    "is this a valid way to create a random dataset? LOL"
    X_train, y_train = [], []
    X_val, y_val = [], []

    for i in range(len(time_series)-window_size-prediction_window):
        feature = time_series[i:i+window_size]
        target = time_series[i+window_size+prediction_window]
        if random.random() < val_frac:
            X_val.append(feature)
            y_val.append(target)
        else:
            X_train.append(feature)
            y_train.append(target)
            
    X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
    X_train = X_train.view(-1, window_size).to(DEVICE)
    y_train = torch.tensor(np.array(y_train), dtype=torch.float32)
    y_train = y_train.view(-1, 1).to(DEVICE)
    X_val = torch.tensor(np.array(X_val), dtype=torch.float32)
    X_val = X_val.view(-1, window_size).to(DEVICE)
    y_val = torch.tensor(np.array(y_val), dtype=torch.float32)
    y_val = y_val.view(-1, 1).to(DEVICE)

    
    return X_train, y_train, X_val, y_val
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, prediction_window):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first =True)
        self.fc = nn.Linear(hidden_size, output_size)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.startup_params = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'output_size': output_size,
            'prediction_window': prediction_window,
        }
        

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x
    
    def save(self, path):

        path = open(path, 'wb')
        parent_dir = os.path.dirname(path)
        os.makedirs(parent_dir, exist_ok=True)
        
        torch.save(self.state_dict(), path)
        with open(path.replace('.pt', '_startup_params.json'), 'w') as f:
            json.dump(self.startup_params, f)
            
    def predict(self, x):
        #for some reason this crashes if not on the cpu?
        
        l = []
        preds = []
        self.eval()
        self = self.to('cpu')
        with torch.no_grad():
            for i in range(len(x) - self.input_size):
                tensor = x[i:i+self.input_size]
                tensor = torch.tensor(np.array(tensor), dtype = torch.float32)
                tensor = tensor.view(-1,self.input_size).to('cpu')
                pred = self(tensor)
                # pred = pred.tonumpy()
                preds.append(pred)
        
        self = self.to(DEVICE)
        return np.array(preds).squeeze()
        
    
def load_model_from_params(path):
    assert path[-3:] == '.pt'
    try:
        with open(path.replace('.pt', '_startup_params.json'), 'r') as f:
            startup_params = json.load(f)
    except:
        logger.info('Unable to load start up params')
    
    model = LSTM(**startup_params)
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    return model    

def main():
    s3 = S3Client()

    time_series = get_time_series()
    time_series = normalize_ts(time_series)
    
    test_frac = .05
    time_series, holdout = np.split(time_series, [int(len(time_series) * (1 - test_frac))])
    
    logger.info('Creating datasets...')
    X_train, y_train, X_val, y_val = create_random_datasets(time_series,window_size=WINDOW_SIZE, prediction_window=PREDICTION_WINDOW)
    
    hidden_size = 32
    output_size = 1  # Predicting 'price_normalized'

    # Initialize the LSTM model
    model = LSTM(WINDOW_SIZE, hidden_size, output_size, PREDICTION_WINDOW)
    model.to(DEVICE)
    
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    train_loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=BATCH_SIZE)
    val_loader = data.DataLoader(data.TensorDataset(X_val, y_val), shuffle=False, batch_size=128)

    n_epochs = 2000
    early_stop_count = 0
    patience = 50
    min_val_loss = float('inf')

    train_losses = []
    val_losses = []
    history_df = pd.DataFrame(columns=['train_loss', 'val_loss'])
    history_df.index.name = 'epoch'

    logger.info('Training model...')
    for epoch in range(n_epochs):
        batch_counter = 0
        train_loss = 0.0
        val_loss = 0.0

        model.train()
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            batch_counter += 1
            # logger.info(f'Epoch {epoch}, batch {batch_counter}, train loss: {loss.item()}')
        
        train_loss /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                X_batch, y_batch = batch
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                val_loss += loss.item() * X_batch.size(0)
            val_loss /= len(val_loader.dataset)
            
            logger.info("Epoch %d: train RMSE %.4f, val RMSE %.4f" % (epoch, train_loss, val_loss))
            
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            early_stop_count = 0
            logger.info('New best val score!')
            model.eval()
            torch.save(model.state_dict(), ROOT_DIR/f'data/models/{OUTPUT_NAME}lstm_best.pt')
            try:
                s3.save_model(model, f'models/{OUTPUT_NAME}/lstm_best.pt')
            except:
                logger.exception('Unable to save model to s3')
            logger.info('Done saving model to s3')
            
            test_preds = model.predict(holdout.squeeze())
            pred_df = pd.DataFrame({'actual': holdout.squeeze()[PREDICTION_WINDOW:]})
            pred_df.loc[PREDICTION_WINDOW:, 'preds'] = test_preds
            s3.write_csv(pred_df, f'models/{OUTPUT_NAME}/test_preds.csv', index = False)
            logger.info('Saved test preds...')
        else:
            early_stop_count += 1
            
            if early_stop_count >= patience:
                logger.info('Stopping early! {}'.format(epoch))
                break
            
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        history_df.loc[epoch] = pd.Series({'train_loss': train_loss, 'val_loss': val_loss}, name = epoch)
        history_df.to_csv(ROOT_DIR/'data/models/lstm_history.csv', index = True)
        
        s3.write_csv(history_df, f'models/{OUTPUT_NAME}/lstm_history.csv', index = True)
        
        
        
    
# %%
if __name__ == '__main__':
    main()