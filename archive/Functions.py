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

def IdentifyStocksOfInterest(method):
    #symbols = pd.read_csv('symbols.csv')['Symbol'].tolist()
    
    #with open('./Data/energy_tickers.txt', 'r') as f:
    #    symbols = f.read().split('\n')
    
    if method == 'handpick':
        symbols = ['AAPL', 'TSLA', 'APHA', 'GOOGL', 'GE', 'F', 'AAL', 'AMZN', 'MSFT', 'KIRK', 'MSTR', 'FB', 'BABA', 'BRK.A', 'V', 'JNJ', 'JPM', 'WMT', 'NVDA', 'PYPL', 'DIS']
        return symbols
    elif method.lower() == 'sp':
        return pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()

def GetHistoricalData(symbols, api, end_date = datetime.now(), n_data_points = 7500, time_interval = 'hour'):
    symbols_to_pull = np.unique(symbols)
    
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)

    data = pd.DataFrame()

    if time_interval == 'minute':
        time_step = 30
    elif time_interval == 'hour':
        time_step = 1500
    else:
        raise ValueError('Only can handle hour or minute right now')

    while data.shape[0] < n_data_points:
        start = end_date - relativedelta(days=time_step)
        data_list = []
        for sym in symbols_to_pull:
            quotes = api.polygon.historic_agg_v2(symbol = sym, multiplier = 1, timespan = time_interval, 
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

def GetLongReturns(symbols_to_consider, api, start, end):

    starting_prices = GetDayQuotes(symbols_to_consider, api, start)
    ending_prices = GetDayQuotes(symbols_to_consider, api, end)

    valid_symbols = [sym for sym in symbols_to_consider if sym in starting_prices.keys() and ending_prices.keys()]
    invalid_symbols = [sym for sym in symbols_to_consider if sym not in valid_symbols]

    if len(invalid_symbols) > 0:
        logging.debug('Skipping {}, had missing values for the period.'.format(invalid_symbols))

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
    date = df.index.max().strftime('%Y%m%d')
    #[os.remove(os.path.join('Logs/Tensorboard', f)) for f in os.listdir('Logs/Tensorboard')]
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir = 'Logs/Tensorboard/' + date)
    my_callbacks = [early_stopping, model_checkpoints, tensorboard]

    model.fit(
        X_train, Y_train, 
        epochs = MAX_EPOCHS, 
        validation_data = (X_test, Y_test), 
        batch_size = BATCH_SIZE, 
        callbacks = my_callbacks,
        verbose = 0
        )

    model.load_weights('models/checkpoints/')
    model.save_weights('models/TrainedModel')

    return model, scalers

def CheckHoliday(date):
    nyse = mcal.get_calendar('NYSE')
    while date.isoweekday() > 5 or date in nyse.holidays().holidays:
        date += timedelta(days = 1)
    return date

import pandas as pd
from scipy.stats import norm
import numpy as np
from scipy.optimize import minimize
import math

def drawdown(return_series: pd.Series):
    """
    Takes a time series of asset returns.
    Computes and returns a DF that contains:
    wealth index
    previous peaks
    percent drawdowns
    """
    
    wealth_index = 1000 * (1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    
    return pd.DataFrame({
        'Wealth': wealth_index,
        'Previous Peak': previous_peaks,
        'Drawdown': drawdowns
    })

def get_ffme_returns():
    me_m = pd.read_csv('data/Portfolios_Formed_on_ME_monthly_EW.csv', 
                      header = 0, index_col=0, parse_dates = True, na_values=-99.99)

    rets = me_m[['Lo 10', 'Hi 10']]
    rets.columns = ['SmallCap', 'LargeCap']
    rets = rets / 100
    rets.index = pd.to_datetime(rets.index, format = '%Y%m').to_period('M')
    
    return rets

def get_ind_returns():
    ind = pd.read_csv('data/ind30_m_vw_rets.csv', header = 0, index_col = 0, parse_dates = True)/100
    ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind_size():
    ind = pd.read_csv('data/ind30_m_size.csv', header = 0, index_col = 0, parse_dates = True)
    ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind


def get_ind_nfirms():
    ind = pd.read_csv('data/ind30_m_nfirms.csv', header = 0, index_col = 0, parse_dates = True)
    ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_total_market_index_returns():
    ind_mktcap = get_ind_nfirms() * get_ind_size()
    total_mktcap = ind_mktcap.sum(axis = 'columns')
    ind_capweight = ind_mktcap.divide(total_mktcap, axis = 'rows')
    total_market_return = (ind_capweight * get_ind_returns()).sum(axis='columns')
    return total_market_return



def get_hfi_returns():
    hfi = pd.read_csv('data/edhec-hedgefundindices.csv', 
                      header = 0, index_col=0, parse_dates = True, na_values=-99.99)
    hfi = hfi / 100
    hfi.index = hfi.index.to_period('M')
    
    return hfi

def skewness(r):
    """
    Alternative to scipy.stats.skew()
    computes skewness of a series or dataframe
    returns float or series
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof = 0) #using population SD
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    computes kurtosis of a series or dataframe
    returns float or series
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof = 0) #using population SD
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def is_normal(r, level = .01):
    """
    Applies JB test to determine if series is normal or not
    applies test at 1% level by default
    Returns True if hypothsesis of normality is accepted, False otherwise
    """
    
    statistics, p_value = scipy.stats.jarque_bera(r)
    
    return p_value > level

def semideviation(r):
    """
    returns the semi-deviation aka negative semidiviation of r
    r must be a series or dataFrame
    """
    
    return r[r<0].std(ddof=0)

def var_historic(r, level=5):
    """
    Returns historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    i.e. There is a level% chance the the returns in any given time period will
    be X or worse
    """
    if isinstance(r, pd.DataFrame):
        return r.agg(var_historic, level = level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError('expected r to be pandas Series or DataFrame')
        

def var_gaussian(r, level=5, modified = False):
    """
    returns the parametric Gaussian VaR of a series or DF
    """
    z = norm.ppf(level/100)
    
    if modified:
        s = skewness(r)
        k = kurtosis(r)
        z = (z + 
                 (z**2 - 1)*s/6 + 
                 (z**3 - 3*z)*(k-3)/24 - 
                 (2*z**3 - 5*z) * (s**2)/36
                        )
    return -(r.mean() + z*r.std(ddof=0))

def cvar_historic(r, level = 5):
    """
    Computes Conditional VaR of series or DF
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level = level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.agg(cvar_historic, level = level)
    else:
        raise TypeError('expected r to be series or DF')
        
def annualize_rets(r, periods_per_year):
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1
        
def annualize_vol(r, periods_per_year):
    """
    Annualizes teh vol of a set of returns
    """
    return r.std()*(periods_per_year**.5)

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret / ann_vol

def portfolio_return(weights, returns):
    """
    Weights -> Returns
    """
    return weights.T @ returns

def portfolio_vol(weights, covmat):
    """
    Weights -> Vol
    """
    return (weights.T @ covmat @ weights)**(.5)

def plot_ef2(n_points, er, cov, style = '.-'):
    if er.shape[0] !=2 or er.shape[0] !=2:
        raise ValueError('plot_ef2 can only plot 2 asset frontiers')
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({'Returns': rets, 'Volatility': vols})
    return ef.plot.line('Volatility', 'Returns', style = style)

def minimize_vol(target_return, er, cov):
    """
    target_ret -> W
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n) #equal weighting
    
    #Define Constraints
    bounds = ((0.0, 1.0), ) * n #makes n copies of the tuple
    return_is_target = { 
        'type': 'eq', #it's an equality constraint, should be 0 when succeeds
        'args': (er,), 
        'fun': lambda w, er: target_return - portfolio_return(w, er)
    }
    weights_sum_to_1= {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    results = minimize(portfolio_vol, init_guess,
                      args = (cov,), method = 'SLSQP',
                      options = {'disp': False},
                      constraints = (return_is_target, weights_sum_to_1),
                      bounds = bounds
                      )
    return results.x


def optimal_weights(n_points, er, cov):
    """
    _> list of weights to run the optimizer on to minimize the vol
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    
    return weights



def msr(riskfree_rate, er, cov):
    """
    RiskFreeRate + ER + COV -> W
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n) #equal weighting
    
    #Define Constraints
    bounds = ((0.0, 1.0), ) * n #makes n copies of the tuple
    weights_sum_to_1= {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    
    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        """
        Returns negative of Sharpe Ratio, given weights
        """
        
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate) / vol
    
    
    results = minimize(neg_sharpe_ratio, init_guess,
                      args = (riskfree_rate, er, cov,), method = 'SLSQP',
                      options = {'disp': False},
                      constraints = (weights_sum_to_1),
                      bounds = bounds
                      )
    return results.x

def gmv(cov):
    """
    Returns the weights of the Global Minimum Vol portfolio
    given covariance matrix
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1,n), cov)

def plot_ef(n_points, er, cov, style = '.-', riskfree_rate = .1, 
            show_cml = False, show_ew = False, show_gmv = False):
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({'Returns': rets, 'Volatility': vols})
    
    ax = ef.plot.line('Volatility', 'Returns', style = style)
    
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        ax.plot([vol_ew], [r_ew], color = 'goldenrod', marker = 'o', markersize=12)
    
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        ax.plot([vol_gmv], [r_gmv], color = 'midnightblue', marker = 'o', markersize=10)
    
    if show_cml:
        ax.set_xlim(left = 0)

        #Calculate MSR
        rf = riskfree_rate
        w_msr = msr(rf, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)

        #Add CML
        cml_1 = [0, vol_msr]
        cml_2 = [rf, r_msr]
        ax.plot(cml_1, cml_2, color = 'green', marker = 'o',
                linestyle='dashed', markersize = 12, linewidth = 2)

    return ax


def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03, drawdown=None):
    """
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset
    Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History
    """
    # set up the CPPI parameters
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start*floor
    peak = account_value
    if isinstance(risky_r, pd.Series): 
        risky_r = pd.DataFrame(risky_r, columns=["R"])

    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate/12 # fast way to set all values to a number
    # set up some DataFrames for saving intermediate values
    account_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    floorval_history = pd.DataFrame().reindex_like(risky_r)
    peak_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak*(1-drawdown)
        cushion = (account_value - floor_value)/account_value
        risky_w = m*cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1-risky_w
        risky_alloc = account_value*risky_w
        safe_alloc = account_value*safe_w
        # recompute the new account value at the end of this step
        account_value = risky_alloc*(1+risky_r.iloc[step]) + safe_alloc*(1+safe_r.iloc[step])
        # save the histories for analysis and plotting
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
        floorval_history.iloc[step] = floor_value
        peak_history.iloc[step] = peak
    risky_wealth = start*(1+risky_r).cumprod()
    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth, 
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "floor": floor,
        "risky_r":risky_r,
        "safe_r": safe_r,
        "drawdown": drawdown,
        "peak": peak_history,
        "floor": floorval_history
    }
    return backtest_result

def summary_stats(r, riskfree_rate=0.03):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r = r.aggregate(annualize_rets, periods_per_year=12)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=12)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=12)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_historic)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })

def gbm(n_years = 10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0, prices=True):
    """
    Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
    :param n_years:  The number of years to generate data for
    :param n_paths: The number of scenarios/trajectories
    :param mu: Annualized Drift, e.g. Market Return
    :param sigma: Annualized Volatility
    :param steps_per_year: granularity of the simulation
    :param s_0: initial value
    :return: a numpy array of n_paths columns and n_years*steps_per_year rows
    """
    # Derive per-step Model Parameters from User Specifications
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) + 1
    # the standard way ...
    # rets_plus_1 = np.random.normal(loc=mu*dt+1, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios))
    # without discretization error ...
    rets_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    ret_val = s_0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1
    return ret_val

def discount(t, r):
    """
    Compute the price of a pure discount bond that pays a dollar at time period t
    and r is the per-period interest rate
    returns a |t| x |r| Series or DataFrame
    r can be a float, Series or DataFrame
    returns a DataFrame indexed by t
    """
    discounts = pd.DataFrame([(r+1)**-i for i in t])
    discounts.index = t
    return discounts

def pv(l, r):
    """
    Compute the present value of a list of liabilities given by the time (as an index) and amounts
    """
    dates = l.index
    discounts = discount(dates, r)
    return discounts.multiply(l, axis = 'rows').sum()

def funding_ratio(assets, liabilities, r):
    """
    Computes the funding ratio of a series of liabilities, based on an interest rate and current value of assets
    """
    return pv(assets, r)/pv(liabilities, r)

def bond_cash_flows(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12):
    """
    Returns the series of cash flows generated by a bond,
    indexed by the payment/coupon number
    """
    n_coupons = round(maturity*coupons_per_year)
    coupon_amt = principal*coupon_rate/coupons_per_year
    coupons = np.repeat(coupon_amt, n_coupons)
    coupon_times = np.arange(1, n_coupons+1)
    cash_flows = pd.Series(data=coupon_amt, index=coupon_times)
    cash_flows.iloc[-1] += principal
    return cash_flows
    
def bond_price(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12, discount_rate=0.03):
    """
    Computes the price of a bond that pays regular coupons until maturity
    at which time the principal and the final coupon is returned
    This is not designed to be efficient, rather,
    it is to illustrate the underlying principle behind bond pricing!
    If discount_rate is a DataFrame, then this is assumed to be the rate on each coupon date
    and the bond value is computed over time.
    i.e. The index of the discount_rate DataFrame is assumed to be the coupon number
    """
    if isinstance(discount_rate, pd.DataFrame):
        pricing_dates = discount_rate.index
        prices = pd.DataFrame(index=pricing_dates, columns=discount_rate.columns)
        for t in pricing_dates:
            prices.loc[t] = bond_price(maturity-t/coupons_per_year, principal, coupon_rate, coupons_per_year,
                                      discount_rate.loc[t])
        return prices
    else: # base case ... single time period
        if maturity <= 0: return principal+principal*coupon_rate/coupons_per_year
        cash_flows = bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)
        return pv(cash_flows, discount_rate/coupons_per_year)

def macaulay_duration(flows, discount_rate):
    """
    Computes the Macaulay Duration of a sequence of cash flows, given a per-period discount rate
    """
    discounted_flows = discount(flows.index, discount_rate)*pd.DataFrame(flows)
    weights = discounted_flows/discounted_flows.sum()
    return np.average(flows.index, weights=weights.iloc[:,0])

def match_durations(cf_t, cf_s, cf_l, discount_rate):
    """
    Returns the weight W in cf_s that, along with (1-W) in cf_l will have an effective
    duration that matches cf_t
    """
    d_t = macaulay_duration(cf_t, discount_rate)
    d_s = macaulay_duration(cf_s, discount_rate)
    d_l = macaulay_duration(cf_l, discount_rate)
    return (d_l - d_t)/(d_l - d_s)

def inst_to_ann(r):
    """
    Convert an instantaneous interest rate to an annual interest rate
    """
    return np.expm1(r)

def ann_to_inst(r):
    """
    Convert an instantaneous interest rate to an annual interest rate
    """
    return np.log1p(r)

def cir(n_years = 10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None):
    """
    Generate random interest rate evolution over time using the CIR model
    b and r_0 are assumed to be the annualized rates, not the short rate
    and the returned values are the annualized rates as well
    """
    if r_0 is None: r_0 = b 
    r_0 = ann_to_inst(r_0)
    dt = 1/steps_per_year
    num_steps = int(n_years*steps_per_year) + 1 # because n_years might be a float
    
    shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0

    ## For Price Generation
    h = math.sqrt(a**2 + 2*sigma**2)
    prices = np.empty_like(shock)
    ####

    def price(ttm, r):
        _A = ((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B = (2*(math.exp(h*ttm)-1))/(2*h + (h+a)*(math.exp(h*ttm)-1))
        _P = _A*np.exp(-_B*r)
        return _P
    prices[0] = price(n_years, r_0)
    ####
    
    for step in range(1, num_steps):
        r_t = rates[step-1]
        d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
        rates[step] = abs(r_t + d_r_t)
        # generate prices at time t as well ...
        prices[step] = price(n_years-step*dt, rates[step])

    rates = pd.DataFrame(data=inst_to_ann(rates), index=range(num_steps))
    ### for prices
    prices = pd.DataFrame(data=prices, index=range(num_steps))
    ###
    return rates, prices


def bond_total_return(monthly_prices, principal, coupon_rate, coupons_per_year):
    """
    Computes the total return of a Bond based on monthly bond prices and coupon payments
    Assumes that dividends (coupons) are paid out at the end of the period (e.g. end of 3 months for quarterly div)
    and that dividends are reinvested in the bond
    """
    coupons = pd.DataFrame(data = 0, index=monthly_prices.index, columns=monthly_prices.columns)
    t_max = monthly_prices.index.max()
    pay_date = np.linspace(12/coupons_per_year, t_max, int(coupons_per_year*t_max/12), dtype=int)
    coupons.iloc[pay_date] = principal*coupon_rate/coupons_per_year
    total_returns = (monthly_prices + coupons)/monthly_prices.shift()-1
    return total_returns.dropna()



def bt_mix(r1, r2, allocator, **kwargs):
    """
    Runs a backtest of allocatin between two sets of returns
    r1 and r2 are T x N dataframes or returns where T is the time step index and N is the number of scenarios
    allocator is a function that takes returns and parameters, and produces an allocation as a Tx1 dataframe
    Returns T x N DF of resulting scenarios
    """
    
    if not r1.shape == r2.shape:
        raise ValueError('r1 and r2 need to be the same shape')
    weights = allocator(r1, r2, **kwargs)
    if not weights.shape == r1.shape:
        raise ValueError('Allocator returned weights that dont match r1')
    r_mix = weights * r1 + (1-weights) * r2
    return r_mix
    
    
def fixedmix_allocator(r1, r2, w1, **kwargs):
    """
    Produces a time series over T steps of allocations between PSP and GHP across N scenarios
    PSP and GHP are TxN df of returns, each column is a scenario
    Returns a TXN dataframe of PSP weights
    """
    return pd.DataFrame(data=w1, index = r1.index, columns=r1.columns)

def terminal_values(rets):
    """
    returns final value of a dollar at the end of the return period
    """
    return (rets+1).prod()

def terminal_stats(rets, floor = .8, cap=np.inf, name = 'Stats'):
    terminal_wealth = (rets+1).prod()
    breach = terminal_wealth < floor
    reach = terminal_wealth >= cap
    p_breach = breach.mean() if breach.sum() > 0 else np.nan
    p_reach = reach.mean() if reach.sum() > 0 else np.nan
    e_short = (floor-terminal_wealth[breach]).mean() if breach.sum() > 0 else np.nan
    e_surplus = (cap-terminal_wealth[reach]).mean() if reach.sum() > 0 else np.nan
    sum_stats = pd.DataFrame.from_dict({
        'mean': terminal_wealth.mean(),
        'std': terminal_wealth.std(),
        'p_breach': p_breach,
        'e_short': e_short,
        'p_reach': p_reach,
        'e_surplus': e_surplus
    }, orient = 'index', columns = [name])
    return sum_stats


def glidepath_allocator(r1, r2, start_glide = 1, end_glide = 0):
    """
    Simulates a TDF style move from r1 to r2
    """
    n_points = r1.shape[0]
    n_col = r1.shape[1]
    path = pd.Series(data = np.linspace(start_glide, end_glide, num = n_points))
    paths = pd.concat([path]*n_col, axis = 1)
    paths.index = r1.index
    paths.columns = r1.columns
    return paths


def floor_allocator(psp_r, ghp_r, floor, zc_prices, m=3):
    """
    Allocate between PSP and GHP with the goal to provide exposure to upside
    without violating floor
    Uses CPPI style risk budgeting to invest a multiple of cushion in PSP
    Returns DF with same shape as psp/ghp representing weights in gsp
    """
    
    if zc_prices.shape != psp_r.shape:
        raise ValueError('Psp and ZC prices must have same shape')
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index = psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value = floor * zc_prices.iloc[step] #PV of floor assuming today's rates and flat YC
        cushion = (account_value - floor_value) / account_value
        psp_w = (m*cushion).clip(0,1)
        ghp_w = 1-psp_w
        psp_alloc = account_value * psp_w
        ghp_alloc = account_value * ghp_w
        #Recompute account value at end of step
        account_value = psp_alloc * (1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        w_history.iloc[step] = psp_w
    return w_history

def drawdown_allocator(psp_r, ghp_r, maxdd, m=3):
    """
    Allocate between PSP and GHP with the goal to provide exposure to upside
    without violating floor
    Uses CPPI style risk budgeting to invest a multiple of cushion in PSP
    Returns DF with same shape as psp/ghp representing weights in gsp
    """
    
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    peak_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index = psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value = (1-maxdd) * peak_value
        cushion = (account_value - floor_value) / account_value
        psp_w = (m*cushion).clip(0,1)
        ghp_w = 1-psp_w
        psp_alloc = account_value * psp_w
        ghp_alloc = account_value * ghp_w
        #Recompute account value at end of step
        account_value = psp_alloc * (1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        peak_value = np.maximum(peak_value, account_value)
        w_history.iloc[step] = psp_w
    return w_history

############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED
############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED
############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED
############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED############DEPRECATED

# class WindowGenerator():
#     def __init__(self, input_width, label_width, shift, data, label_columns=None):
        
#         #Index the labels (and all columns)
#         self.label_columns = label_columns
#         if label_columns is not None:
#            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
#         self.column_indices = {name: i for i,name in enumerate(data.columns)}
        
#         # Work out the window parameters.
#         self.input_width = input_width
#         self.label_width = label_width
#         self.shift = shift
#         self.total_window_size = input_width + shift
#         self.input_slice = slice(0, input_width)
#         self.input_indices = np.arange(self.total_window_size)[self.input_slice]
#         self.label_start = self.total_window_size - self.label_width
#         self.labels_slice = slice(self.label_start, None)
#         self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        
#     def __repr__(self):
#         return '\n'.join([
#             f'Total window size: {self.total_window_size}',
#             f'Input indices: {self.input_indices}',
#             f'Label indices: {self.label_indices}',
#             f'Label column name(s): {self.label_columns}'])
    
#     def split_window(self, features):
#         inputs = features[:, self.input_slice, :]
#         labels = features[:, self.labels_slice, :]
#         if self.label_columns is not None:
#             labels = tf.stack(
#             [labels[:, :, self.column_indices[name]] for name in self.label_columns],
#                 axis = -1)
#         inputs.set_shape([None, self.input_width, None])
#         labels.set_shape([None, self.label_width, None])
#         return inputs, labels
    
#     def plot(self, plot_col, model=None, max_subplots=3):
#         inputs, labels = self.example
#         plt.figure(figsize=(12, 8))
#         plot_col_index = self.column_indices[plot_col]
#         max_n = min(max_subplots, len(inputs))
#         for n in range(max_n):
#             plt.subplot(3, 1, n+1)
#             plt.ylabel(f'{plot_col} [normed]')
#             plt.plot(self.input_indices, inputs[n, :, plot_col_index],
#              label='Inputs', marker='.', zorder=-10)
            
#             if self.label_columns:
#                 label_col_index = self.label_columns_indices.get(plot_col, None)
#             else:
#                 label_col_index = plot_col_index
            
#             if label_col_index is None:
#                 continue

#             plt.scatter(self.label_indices, labels[n, :, label_col_index],
#                         edgecolors='k', label='Labels', c='#2ca02c', s=64)
#             if model is not None:
#                 predictions = model(inputs)
#                 plt.scatter(self.label_indices, predictions[n, :, label_col_index],
#                           marker='X', edgecolors='k', label='Predictions',
#                           c='#ff7f0e', s=64)
#             if n == 0:
#                 plt.legend()
#         plt.xlabel('Time [d]')
    
#     def make_dataset(self, data, batch_size = 32):
#         data = np.array(data, dtype=np.float32)
#         ds = tf.keras.preprocessing.timeseries_dataset_from_array(
#             data=data,
#             targets=None,
#             sequence_length=self.total_window_size,
#             sequence_stride=1,
#            # shuffle=shuffle, shuffle now down in fit call
#             batch_size=batch_size
#         )
#         ds = ds.map(self.split_window)
        
#         return ds


# class FeedBack(tf.keras.Model):
#     def __init__(self, units, out_steps, num_features):
#         super().__init__()
#         self.out_steps = out_steps
#         self.units = units
#         self.num_features = num_features
#         self.lstm_cell = tf.keras.layers.LSTMCell(units)
#         # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
#         self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
#         self.dense = tf.keras.layers.Dense(num_features)
    
#     def warmup(self, inputs):
#         # inputs.shape => (batch, time, features)
#         # x.shape => (batch, lstm_units)
#         x , *state = self.lstm_rnn(inputs)
#         # predictions.shape => (batch, features)
#         prediction = self.dense(x)
#         return prediction, state
        
#     def call(self, inputs, training=None):
#         # Use a TensorArray to capture dynamically unrolled outputs.
#         predictions = []
#         # Initialize the lstm state
#         prediction, state = self.warmup(inputs)

#         # Insert the first prediction
#         predictions.append(prediction)

#         # Run the rest of the prediction steps
#         for n in range(1, self.out_steps):
#             # Use the last prediction as input.
#             x = prediction
#             # Execute one lstm step.
#             x, state = self.lstm_cell(x, states=state, training=training)
#             x, state = self.lstm_cell(x, states = state, training = training)
#             # Convert the lstm output to a prediction.
#             prediction = self.dense(x)
#             # Add the prediction to the output
#             predictions.append(prediction)

#         # predictions.shape => (time, batch, features)
#         predictions = tf.stack(predictions)
#         # predictions.shape => (batch, time, features)
#         predictions = tf.transpose(predictions, [1, 0, 2])
#         return predictions
    
# def CreateModel(data):
#     input_width = 30
#     label_width = 1
    
#     model = FeedBack(16, label_width, data.shape[1])
#     model.compile(loss = tf.losses.MeanSquaredError(), optimizer = tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
    
#     return model
    
# def TrainModel(model, target_symbol, input_data, epochs):
#     input_width = 30
#     label_width = 1
#     # perhaps just normalize? no differencing?
#     diff_data = input_data.diff()
#     diff_data.iloc[0,:] = 0

#     train_df = diff_data[:-label_width*3]
#     val_df = diff_data[-(input_width+label_width*3):]

#     window = WindowGenerator(input_width, label_width, label_width, train_df, label_columns=[target_symbol])

#     train_ds = window.make_dataset(train_df, batch_size=32)
#     val_ds = window.make_dataset(val_df, batch_size = label_width)

#     early_stop = tf.keras.callbacks.EarlyStopping(patience = 10)

#     checkpoints = tf.keras.callbacks.ModelCheckpoint('./models/checkpoint', save_best_only=True, save_weights_only=True)
#     history = model.fit(train_ds, validation_data=val_ds, shuffle=True, validation_steps=1, callbacks = [early_stop, checkpoints], epochs = epochs, verbose = 0)

#     model.load_weights('./models/checkpoint')
#     return model
    

# def Predict7DayHigh(model, target_symbol, input_data):

#     pred_data = np.reshape(np.array(diff_data[-input_width:]), newshape = (1, input_width, -1))
#     next_7_days = model.predict(tf.constant(pred_data)).reshape(label_width, -1)
#     symbol_seven_day_high = np.max(np.cumsum(next_7_days[:,diff_data.columns == target_symbol]))

#     return (target_symbol, symbol_seven_day_high)

# class WindowGenerator():
#     def __init__(self, input_width, label_width, shift, data, label_columns=None):
        
#         #Index the labels (and all columns)
#         self.label_columns = label_columns
#         if label_columns is not None:
#            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
#         self.column_indices = {name: i for i,name in enumerate(data.columns)}
        
#         # Work out the window parameters.
#         self.input_width = input_width
#         self.label_width = label_width
#         self.shift = shift
#         self.total_window_size = input_width + shift
#         self.input_slice = slice(0, input_width)
#         self.input_indices = np.arange(self.total_window_size)[self.input_slice]
#         self.label_start = self.total_window_size - self.label_width
#         self.labels_slice = slice(self.label_start, None)
#         self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        
#     def __repr__(self):
#         return '\n'.join([
#             f'Total window size: {self.total_window_size}',
#             f'Input indices: {self.input_indices}',
#             f'Label indices: {self.label_indices}',
#             f'Label column name(s): {self.label_columns}'])
    
#     def split_window(self, features):
#         inputs = features[:, self.input_slice, :]
#         labels = features[:, self.labels_slice, :]
#         if self.label_columns is not None:
#             labels = tf.stack(
#             [labels[:, :, self.column_indices[name]] for name in self.label_columns],
#                 axis = -1)
#         inputs.set_shape([None, self.input_width, None])
#         labels.set_shape([None, self.label_width, None])
#         return inputs, labels
    
#     def plot(self, plot_col, model=None, max_subplots=3):
#         inputs, labels = self.example
#         plt.figure(figsize=(12, 8))
#         plot_col_index = self.column_indices[plot_col]
#         max_n = min(max_subplots, len(inputs))
#         for n in range(max_n):
#             plt.subplot(3, 1, n+1)
#             plt.ylabel(f'{plot_col} [normed]')
#             plt.plot(self.input_indices, inputs[n, :, plot_col_index],
#              label='Inputs', marker='.', zorder=-10)
            
#             if self.label_columns:
#                 label_col_index = self.label_columns_indices.get(plot_col, None)
#             else:
#                 label_col_index = plot_col_index
            
#             if label_col_index is None:
#                 continue

#             plt.scatter(self.label_indices, labels[n, :, label_col_index],
#                         edgecolors='k', label='Labels', c='#2ca02c', s=64)
#             if model is not None:
#                 predictions = model(inputs)
#                 plt.scatter(self.label_indices, predictions[n, :, label_col_index],
#                           marker='X', edgecolors='k', label='Predictions',
#                           c='#ff7f0e', s=64)
#             if n == 0:
#                 plt.legend()
#         plt.xlabel('Time [d]')
    
#     def make_dataset(self, data):
#         data = np.array(data, dtype=np.float32)
#         ds = tf.keras.preprocessing.timeseries_dataset_from_array(
#             data=data,
#             targets=None,
#             sequence_length=self.total_window_size,
#             sequence_stride=1,
#             #shuffle=shuffle, shuffle now done at training
#             batch_size=32
#         )
#         ds = ds.map(self.split_window)
        
#         return ds


# def ProcessData(input_data, columns_of_interest):
#     df = input_data.dropna()
#     ori_df = df.copy()
#     dates = pd.to_datetime(df.pop('begins_at'))
    
#     #keep only the close price columns
#     #df = df[[col for col in df.columns if 'close_price' in col]]
#     df = df[columns_of_interest]
    
#     #Difference data and remove introduced NA rows
#     df = df.diff()
#     df = df[1:]
#     ori_df = ori_df[1:]
#     dates = dates[1:]
    
#     ori_df.set_index(dates, inplace = True)
#     df.set_index(dates, inplace = True)
    
#     return df, ori_df

# def TrainSplit(input_data, n_months_val, n_months_test):
#     n_months = len(np.unique(input_data.index.strftime('%m/%Y')))
#     val_end = np.max(input_data.index) - relativedelta(months = n_months_test)
#     train_end = val_end - relativedelta(months = n_months_val)
    
#     train_df = input_data[input_data.index <= train_end]
#     val_df = input_data[(input_data.index > train_end) & (input_data.index <= val_end)]
#     test_df = input_data[input_data.index > val_end]
    
#     print('Train Period: ' + str(np.min(train_df.index).strftime('%Y-%m-%d')) + ' to ' + str(np.max(train_df.index).strftime('%Y-%m-%d')) + '\n' + 
#           'Validation Period: ' + str(np.min(val_df.index).strftime('%Y-%m-%d')) + ' to ' + str(np.max(val_df.index).strftime('%Y-%m-%d')) + '\n' +
#           'Testing Period: ' + str(np.min(test_df.index).strftime('%Y-%m-%d')) + ' to ' + str(np.max(test_df.index).strftime('%Y-%m-%d')))
#     return train_df, val_df, test_df


# def compile_and_fit(model, window, epochs, patience=2, verbose = 0):
#     early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
#                                                       patience=patience,
#                                                       mode='min')
#     model.compile(loss=tf.losses.MeanSquaredError(), 
#                   optimizer=tf.optimizers.Adam(),
#                   metrics=[tf.metrics.MeanAbsoluteError()])
    
#     history = model.fit(window.train, epochs = epochs,
#                         validation_data=window.val,
#                         verbose = verbose
#                         #callbacks=[early_stopping]
#                        )
#     return history


# def TrainModel(model, window, epochs = 50, patience = 5, verbose = 0, return_history = False):
    
#     history = compile_and_fit(model, window, epochs, patience, verbose)
    
#     if return_history:
#         return model, history
#     else:
#         return model




# def Turtle7DayTest(model, window, eval_dataset, pred_dataset, pred_column, starting_cash, test_start):
    
#     all_predictions = model.predict(window.make_dataset(pred_dataset, shuffle = False))
#     seven_day_deltas = [0 for i in range(len(pred_dataset) - len(all_predictions)-1)] + [np.max(np.cumsum(pred)) for pred in all_predictions] + [0]
    
#     eval_df = eval_dataset[[pred_column]].copy()
#     eval_df['SevenDayHigh'] = seven_day_deltas
#     cash_reserves = starting_cash
#     eval_df['Cash'] = 0
#     eval_df['Cost'] = 0
#     eval_df['Revenue'] = 0
#     eval_df['Shares'] = 0
#     eval_df['PortfolioValue'] = 0
#     eval_df = eval_df[eval_df.index > test_start]
    
#     for i in range(len(eval_df)-1):
#         max_delta = eval_df['SevenDayHigh'].iloc[i]
#         eval_df['Cash'].iloc[i] = cash_reserves
#         if max_delta > 0:
#             if eval_df[pred_column].iloc[i] < cash_reserves:
#                 eval_df['Cost'].iloc[i] = eval_df[pred_column].iloc[i]
#                 eval_df['Shares'].iloc[i+1] = eval_df['Shares'].iloc[i] + 1
#                 cash_reserves -= eval_df[pred_column].iloc[i]
#             else:
#                 eval_df['Shares'].iloc[i+1] = eval_df['Shares'].iloc[i]
#         elif eval_df['Shares'].iloc[i] > 0:
#             eval_df['Revenue'].iloc[i] = eval_df[pred_column].iloc[i]
#             eval_df['Shares'].iloc[i+1] = eval_df['Shares'].iloc[i] - 1
#             cash_reserves += eval_df[pred_column].iloc[i]

#         eval_df['PortfolioValue'].iloc[i] = (eval_df[pred_column].iloc[i] * eval_df['Shares'].iloc[i])
        
#     final_cost = np.sum(eval_df['Cost'])
#     final_revenue = np.sum(eval_df['Revenue'])
#     final_assets = eval_df.iloc[-2]['PortfolioValue']
#     final_cash = cash_reserves
    
#     final_return = final_assets + final_cash #- final_cost 
#     long_term_return = (eval_df[pred_column].iloc[-2] / eval_df[pred_column].iloc[0]) * starting_cash
#     profit_score = final_return  / long_term_return
    
#     print('Model Strategy Return: ' + str(final_return))
#     print('Long Term Strategy Return: ' + str(long_term_return))
#     print('Profit Score: ' + str(profit_score))


# class FeedBack(tf.keras.Model):
#     def __init__(self, units, out_steps, num_features):
#         super().__init__()
#         self.out_steps = out_steps
#         self.units = units
#         self.num_features = num_features
#         self.lstm_cell = tf.keras.layers.LSTMCell(units)
#         # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
#         self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
#         self.dense = tf.keras.layers.Dense(num_features)
    
#     def warmup(self, inputs):
#         # inputs.shape => (batch, time, features)
#         # x.shape => (batch, lstm_units)
#         x , *state = self.lstm_rnn(inputs)
#         # predictions.shape => (batch, features)
#         prediction = self.dense(x)
#         return prediction, state
        
#     def call(self, inputs, training=None):
#         # Use a TensorArray to capture dynamically unrolled outputs.
#         predictions = []
#         # Initialize the lstm state
#         prediction, state = self.warmup(inputs)

#         # Insert the first prediction
#         predictions.append(prediction)

#         # Run the rest of the prediction steps
#         for n in range(1, self.out_steps):
#          # Use the last prediction as input.
#           x = prediction
#           # Execute one lstm step.
#         x, state = self.lstm_cell(x, states=state,
#                                 training=training)
#         # Convert the lstm output to a prediction.
#         prediction = self.dense(x)
#         # Add the prediction to the output
#         predictions.append(prediction)

#         # predictions.shape => (time, batch, features)
#         predictions = tf.stack(predictions)
#         # predictions.shape => (batch, time, features)
#         predictions = tf.transpose(predictions, [1, 0, 2])
#         return predictions


# CONV_WIDTH = 3
# OUT_STEPS = 7

# one_shot_lstm_model = tf.keras.Sequential([
#     # Shape [batch, time, features] => [batch, lstm_units]
#     # Adding more `lstm_units` just overfits more quickly.
#     tf.keras.layers.LSTM(4, return_sequences=False),
#     # Shape => [batch, out_steps*features]
#     tf.keras.layers.Dense(OUT_STEPS,
#                           kernel_initializer=tf.initializers.zeros),
#     # Shape => [batch, out_steps, features]
#     #tf.keras.layers.Reshape([OUT_STEPS, 1])
# ])

# multi_dense_model = tf.keras.Sequential([
#     # Take the last time step.
#     # Shape [batch, time, features] => [batch, 1, features]
#     tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
#     # Shape => [batch, 1, dense_units]
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dense(256, activation = 'relu'),
#     # Shape => [batch, out_steps*features]
#     tf.keras.layers.Dense(OUT_STEPS,
#                           kernel_initializer=tf.initializers.zeros),
#     # Shape => [batch, out_steps, features]
#     tf.keras.layers.Reshape([OUT_STEPS, 1])
# ])