#%%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from s3 import S3Client
from io import BytesIO
import numpy as np
import torch.optim as optim
import torch.utils.data as data
import random
import optuna

from config import *
from dataloader import load_price_data

PRODUCT_ID = 'ETH-USD'
GRANULARITY = 'FIFTEEN_MINUTE'
PREDICTION_WINDOW = 24


logger = create_logger(__name__, file = ROOT_DIR/'data/logs/lstm.log')

if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

logger.info(f'Using: {DEVICE}')

import os
import json
if not os.path.exists(ROOT_DIR / 'data' / 'models'):
    os.makedirs(ROOT_DIR / 'data' / 'models')

def get_time_series(product_id, granularity):
    logger.info('Loading data...')
    s3 = S3Client()
    df = load_price_data(granularity, product_id)
    
    df = df.set_index('start').sort_index()

    #TODO: find missing time periods and fill
    
    time_series = df['close'].values.reshape(-1, 1)
    return time_series

def split_time_series_sequentially(time_series, train_frac = .85, val_frac = .1):
    logger.info('Splitting data...')
    
    train_size = int(len(time_series) * train_frac)
    val_size = int(len(time_series) * val_frac)

    train, val, test = np.split(time_series, [train_size, train_size+val_size])
    
    return train, val, test


def create_time_series_dataset(dataset, window_size=1, prediction_window=1):
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




def create_random_time_series_datasets(time_series, window_size=1, prediction_window=1, val_frac=.05):
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
    def __init__(self, input_size, hidden_size, prediction_window, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size = input_size, 
            hidden_size = hidden_size, 
            num_layers = num_layers, 
            dropout = dropout,
            batch_first =True
        )
        self.fc = nn.Linear(hidden_size, 1)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.startup_params = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'prediction_window': prediction_window,
            'num_layers': num_layers
        }
        

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :]) 
        #just the final prediction, weird because most tutorials dont have this
        #but it's the only way of getting this to work with proper "input_size" = 1
        #perhaps if you take out the reshape?
        #TODO: adjust the .predict method to account for this always wanting a 3d tensor (batch, seq, 1)
        return x
    
    def save(self, path):
        path = str(path)

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
                tensor = tensor.view(-1,self.input_size, 1).to('cpu')
                pred = self(tensor)
                # pred = pred.tonumpy()
                preds.append(pred)
        
        self = self.to(DEVICE)
        return np.array(preds).squeeze()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

    
def load_model_from_params(path):
    path = str(path)
    assert path[-3:] == '.pt'
    with open(path.replace('.pt', '_startup_params.json'), 'r') as f:
        startup_params = json.load(f)
    
    model = LSTM(**startup_params)
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    return model    

def load_model_from_s3(s3_path):
    s3 = S3Client()
    obj = s3.s3.get_object(Bucket=s3.bucket, Key = s3_path)
    state_dict = torch.load(BytesIO(obj['Body'].read()))
    startup_params = s3.load_json(s3_path.replace('.pt', '_startup_params.json'))
    model = LSTM(**startup_params)
    model.load_state_dict(state_dict)
    
    return model

def cumulative_return(returns):
    return (returns + 1).prod() - 1

def create_returns_and_targets(price_df, prediction_window):
    price_df = price_df.sort_values('start')
    price_df['returns'] = price_df['close'].pct_change()
    price_df['target'] = price_df['returns'].rolling(prediction_window).apply(cumulative_return).shift(-prediction_window + 1)
    returns = price_df['returns'].values[1:] * 100 
    targets = price_df['target'].values[1:] * 100
    return returns, targets

def create_random_target_datasets(returns, targets, window_size, val_frac = .05):

    X_train, y_train = [], []
    X_val, y_val = [], []

    for i in range(len(returns) - window_size):
        feature = returns[i:i+window_size]
        target = targets[i+window_size-1]
        
        if random.random() < val_frac:
            X_val.append(feature)
            y_val.append(target)
        else:
            X_train.append(feature)
            y_train.append(target)
            
    X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
    X_train = X_train.view(-1, window_size, 1).to(DEVICE)
    y_train = torch.tensor(np.array(y_train), dtype=torch.float32)
    y_train = y_train.view(-1, 1).to(DEVICE)
    X_val = torch.tensor(np.array(X_val), dtype=torch.float32)
    X_val = X_val.view(-1, window_size, 1).to(DEVICE)
    y_val = torch.tensor(np.array(y_val), dtype=torch.float32)
    y_val = y_val.view(-1, 1).to(DEVICE)

    return X_train, y_train, X_val, y_val

def define_model(trial):
        
    hidden_size = trial.suggest_int('hidden_size', 16, 128)
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0, .5)
    model = LSTM(
        input_size = 1, 
        hidden_size = hidden_size, 
        prediction_window=PREDICTION_WINDOW,
        num_layers = num_layers,
        dropout = dropout
    )
    return model.to(DEVICE)
    
def train(model, optimizer, train_loader, val_loader, output_dir, returns_holdout):
    
    #TODO: LR decay
    
    loss_fn = nn.MSELoss()
    n_epochs = 2000
    early_stop_count = 0
    patience = 50
    min_val_loss = float('inf')

    train_losses = []
    val_losses = []
    history_df = pd.DataFrame(columns=['train_loss', 'val_loss'])
    history_df.index.name = 'epoch'

    # logger.info('Training model...')
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
            # if batch_counter % 100 == 0:
            #     logger.info(f'Epoch {epoch}, batch {batch_counter}, train loss: {loss.item()}')
        
        train_loss /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            # logger.info('validation time')
            for batch in val_loader:
                X_batch, y_batch = batch
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                val_loss += loss.item() * X_batch.size(0)
            val_loss /= len(val_loader.dataset)
            
            
            
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            early_stop_count = 0
            best_epoch = epoch
            logger.info("Epoch %d: train RMSE %.4f, val RMSE %.4f" % (epoch, train_loss, val_loss))
            model.eval()
            model.save(output_dir/'lstm_best.pt')
            # try:
            #     s3.save_model(model, f'models/{output_name}/lstm_best.pt')
            # except:
            #     logger.exception('Unable to save model to s3')
            # logger.info('Done saving model to s3')
            
            test_preds = model.predict(returns_holdout)
            pred_df = pd.DataFrame({'actual': returns_holdout.squeeze()[model.input_size:]})
            pred_df['preds'] = test_preds
            pred_df.to_csv(output_dir/'test_preds.csv', index = False)
            # logger.info('Saved test preds...')
        else:
            early_stop_count += 1
            
            if early_stop_count >= patience:
                logger.info('Stopping early! {}'.format(epoch))
                break
            
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        history_df.loc[epoch] = pd.Series({'train_loss': train_loss, 'val_loss': val_loss}, name = epoch)
        history_df.to_csv(output_dir/'lstm_history.csv', index = True)
        
        # s3.write_csv(history_df, f'models/{output_name}/lstm_history.csv', index = True)
        
    return min_val_loss, best_epoch
        
def objective(trial, product_id):
    
    output_name = f'{product_id}_{GRANULARITY}'
    output_dir = ROOT_DIR/f'data/models/{output_name}/{trial.number}/'
    os.makedirs(output_dir, exist_ok = True)
    
    price_df = load_price_data(GRANULARITY, product_id)
    
    returns, targets = create_returns_and_targets(price_df, PREDICTION_WINDOW)
    
    test_frac = .01
    
    returns, returns_holdout = np.split(returns, [int(len(returns) * (1 - test_frac))])
    targets, targets_holdout = np.split(targets, [int(len(targets) * (1 - test_frac))])
    
    window_size = trial.suggest_int('window_size', 4 * 6, 4 * 24 * 7)
    
    # logger.info('Creating datasets...')
    
    X_train, y_train, X_val, y_val = create_random_target_datasets(
        returns, targets,
        window_size=window_size, 
    )
    
    model = define_model(trial)
    # logger.info(f"Model Size: {count_parameters(model)}")
    
    batch_size = trial.suggest_int('batch_size', 8, 128)
    train_loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=batch_size)
    val_loader = data.DataLoader(data.TensorDataset(X_val, y_val), shuffle=False, batch_size=64)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr = trial.suggest_float('lr', 1e-5, 1e-1),
    )
    
    min_val_loss, best_epoch = train(model, optimizer, train_loader, val_loader, output_dir, returns_holdout)

    trial.report(min_val_loss, best_epoch)
    
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    
    return min_val_loss
        
        
        
def non_optuna():
    output_name = f'{PRODUCT_ID}_{GRANULARITY}'
    # output_dir = ROOT_DIR/f'data/models/{output_name}/{trial.number}/'
    output_dir = ROOT_DIR/f'data/models/{output_name}/test/'
    os.makedirs(output_dir, exist_ok = True)
    
    price_df = load_price_data(GRANULARITY, PRODUCT_ID)
    
    returns, targets = create_returns_and_targets(price_df, PREDICTION_WINDOW)
    
    test_frac = .01
    
    returns, returns_holdout = np.split(returns, [int(len(returns) * (1 - test_frac))])
    targets, targets_holdout = np.split(targets, [int(len(targets) * (1 - test_frac))])
    
    window_size = 24
    
    # logger.info('Creating datasets...')
    
    X_train, y_train, X_val, y_val = create_random_target_datasets(
        returns, targets,
        window_size=window_size, 
    )
    
    model = model = LSTM(
        input_size = 1, 
        hidden_size = 32, 
        prediction_window=PREDICTION_WINDOW,
        num_layers = 2,
        dropout = .5
    ).to(DEVICE)
    
    logger.info(f"Model Size: {count_parameters(model)}")
    
    
    batch_size = 64
    train_loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=batch_size)
    val_loader = data.DataLoader(data.TensorDataset(X_val, y_val), shuffle=False, batch_size=128)
    optimizer = optim.Adam(
        model.parameters(),
    )
    
    min_val_loss, best_epoch = train(model, optimizer, train_loader, val_loader, output_dir, returns_holdout)
    
    return min_val_loss, best_epoch
# %%
if __name__ == '__main__':
    # non_optuna()
    best_results = {}
    for product in ['BTC-USD', 'ETH-USD', 'SOL-USD', 'MATIC-USD', 'LINK-USD']:
        try:
            study = optuna.create_study(direction = 'minimize')
            study.optimize(lambda trial: objective(trial, product), n_trials = 20)
            logger.info(f"Best trial for {product}: {study.best_trial.number}")
            best_results[product] = study.best_trial.number
        except:
            logger.exception('Optuna failed')
            continue
        
    with open(ROOT_DIR/'data/lstm_training_results.json', 'w') as f:
        json.dump(best_results, f)
