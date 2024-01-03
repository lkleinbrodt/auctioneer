from config import *
from lstm import *
import zipfile
from scipy.optimize import minimize
s3 = S3Client()

best_trials = s3.load_json('models/best_trials.json')
granularity = 'FIFTEEN_MINUTE'

FEE_PERCENTAGE = .00

def load_dictionaries():
    
    models_dict = {}
    returns_dict = {}
    prices_dict = {}

    for product_id, best_trial in best_trials.items():
        if product_id not in ['BTC-USD', 'ETH-USD']:
            continue
        print(product_id)
        # s3.download_file(f"models/{product_id}_{granularity}/{best_trial}.zip", ROOT_DIR/"data/tmp.zip")
        # os.makedirs(ROOT_DIR/f"data/tmp/{product_id}_{granularity}", exist_ok=True)
        # with zipfile.ZipFile(ROOT_DIR/'data/tmp.zip', 'r') as zip_ref:
        #     zip_ref.extractall(ROOT_DIR/f"data/tmp/{product_id}_{granularity}")
        model = load_model_from_params(ROOT_DIR/f"data/tmp/{product_id}_{granularity}/lstm_best.pt")
        models_dict[product_id] = model
        #TODO: this can be more efficient
        price_df = load_price_data(GRANULARITY, product_id, s3 = False)
        price_df['start'] = pd.to_datetime(price_df['start'], unit = 's')
        
        price_df = price_df[price_df['start'] > '2023-07-01']
        
        returns, targets = create_returns_and_targets(price_df, PREDICTION_WINDOW)
        
        # #TODO: improve
        returns_holdout = returns[returns.index > '2023-08-01']
        returns_dict[product_id] = returns_holdout
        prices_dict[product_id] = price_df.set_index('start')['close'].loc[returns_holdout.index]
        
    return models_dict, returns_dict, prices_dict



def next_allocation(starting_position, expected_returns, starting_cash):
    min_cash = 10_000
    
    init_guess = starting_position
    
    def calculate_fees(starting_position, ending_position, fee_percentage):
        return np.sum(np.abs(ending_position - starting_position) * fee_percentage) #TODO: verify

    def calculate_cash(starting_position, ending_position, fee_percentage, starting_cash):
        fees = calculate_fees(starting_position, ending_position, fee_percentage)
        return starting_cash - fees + np.sum(starting_position - ending_position)

    def expected_value(final_positions):
        
        port_value = np.sum(np.dot(final_positions, expected_returns + 1))
        
        final_cash = calculate_cash(starting_position, final_positions, FEE_PERCENTAGE, starting_cash)
        
        expected_value = port_value + final_cash
        
        return -expected_value

    def cash_above_k(final_positions):
        fees = np.sum(np.abs(final_positions - starting_position) * FEE_PERCENTAGE) #TODO: verify
        final_cash = starting_cash - fees + np.sum(starting_position - final_positions)
        
        return final_cash - min_cash

    cash_constraint = {'type': 'ineq', 'fun': cash_above_k}
    
    lower_bound = 0
    upper_bound = (np.sum(starting_position) + starting_cash) / 3
    
    bounds = ((lower_bound, upper_bound),) * len(starting_position)

    results = minimize(
        expected_value,
        init_guess,
        method = 'SLSQP',
        constraints = [cash_constraint],
        bounds = bounds
    )
    
    final_cash = calculate_cash(starting_position, results.x, FEE_PERCENTAGE, starting_cash)

    # print(f"""
    #     Optimal Portfolio: {pd.Series(results.x).round(2)}.
    #     Fees incurred: {calculate_fees(starting_position, results.x, fee_percentage)}.
    #     Cash remaining: {final_cash}.
    #     """)
    
    return pd.Series(results.x, index = expected_returns.index).round(5), final_cash

def calculate_expected_returns(models_dict, returns_dict, ts):
    expected_returns = {}
    for product_id in models_dict.keys():
        #TODO: i still dont trust window size vs input dim
        tensor = returns_dict[product_id].loc[:ts].values
        #TODO: improve by limiting to window size
        tensor = tensor[-100:]
        tensor = torch.tensor(np.array(tensor), dtype = torch.float32)
        prediction = models_dict[product_id](tensor.reshape(1, -1, 1).to('cpu'))
        expected_returns[product_id] = prediction.item()
        
    return pd.Series(expected_returns)

def dollar_to_unit(portfolio, prices):
    return (portfolio / prices).round(5)

def unit_to_dollar(portfolio, prices):
    return (portfolio * prices).round(5)

def dca(price_df, starting_balance):
    cash = starting_balance
    holdings  = {id: 0 for id in price_df.columns}
    values = []
    dollar_per_asset_per_day = starting_balance / (price_df.shape[0] * price_df.shape[1])
    
    for i, row in price_df.iterrows():
        value = 0
        for product_id, price in row.items():
            holdings[product_id] += dollar_per_asset_per_day / price
            cash -= dollar_per_asset_per_day
            value += holdings[product_id] * price
        value += cash
        values.append(value)
        
    return pd.Series(values, index = price_df.index)

def dca_test(time_steps, price_df):
    price_df = price_df.loc[time_steps]
    
    dca_values = dca(price_df.loc[time_steps], 100_000)
    
    return dca_values
    
def test_period(time_steps, price_df):
    
    price_df = price_df.loc[time_steps]

    cash = 100_000
    portfolios = [pd.Series(0,index = models_dict.keys())]
    values = [cash]
    cashs = [cash]

    for ts in time_steps:
        # if ts == pd.to_datetime('2023-11-26 05:15:00' ):
        #     break
        current_prices = price_df.loc[ts]
        current_unit_positions = portfolios[-1]
        
        current_value = (current_unit_positions * current_prices).sum() + cash
        
        if values[-1] - current_value > 12_000:
            raise ValueError('woah')
        values.append(current_value)
        
        expected_returns = calculate_expected_returns(models_dict, returns_dict, ts)
        
        current_dollar_positions = unit_to_dollar(current_unit_positions, current_prices)
        
        optimal_dollar_positions, _ = next_allocation(current_dollar_positions, expected_returns, cash)
        
        optimal_unit_positions = dollar_to_unit(optimal_dollar_positions, current_prices)
        
        end_positions = pd.Series(0.0, index = models_dict.keys())
        
        diff = optimal_unit_positions - current_unit_positions
        diff = diff.sort_values()

        
        for product_id in diff.index:
            current_position = current_unit_positions.get(product_id, 0)
            next_position = optimal_unit_positions.get(product_id, 0)
            
            if next_position < current_position:
                amount_received = (current_position - next_position) * current_prices[product_id]
                
                if amount_received < 0:
                    raise ValueError("Negative amount received")
                
                cash += (amount_received * (1-FEE_PERCENTAGE))
                
                end_positions[product_id] = next_position
            elif next_position > current_position:
                amount_spent = (next_position - current_position) * current_prices[product_id]
                if amount_spent > cash:
                    print(f"Product: {product_id}")
                    print(f"Current Price: {current_prices[product_id]}")
                    print(f"Current Position: {current_position}")
                    print(f"Next Position: {next_position}")
                    print(f"Amount Spent: {amount_spent}")
                    print(f"Cash: {cash}")
                    raise ValueError("Not enough cash")
                cash -= (amount_spent * (1 + FEE_PERCENTAGE))
                end_positions[product_id] = next_position

            else:
                end_positions[product_id] = current_position
            
        portfolios.append(end_positions.copy())
        cashs.append(cash)
        
    portfolio_df = pd.DataFrame(portfolios[1:])
    portfolio_df.index = time_steps
    values = pd.Series(values[1:], index = time_steps)
    cashs = pd.Series(cashs[1:], index = time_steps)
    
    return values

def choose_random_slices(l, n_slices, min_length, max_length,  k = 1):
    assert len(l) >= (max_length * k)
    slices = []
    for _ in range(n_slices):
        length = random.randint(min_length * k, max_length * k)
        start = random.randint(0, len(l) - length)
        end = start + length
        slices.append(l[start:end:k])
    return slices


def rolling_slices(l, length, slice_step, list_step):
    slices = []
    for i in range(0, len(l) - length*slice_step, list_step):
        slices.append(l[i:i+length*slice_step:slice_step])
    return slices


models_dict, returns_dict, prices_dict = load_dictionaries()
price_df = pd.DataFrame(prices_dict)

step = (60 / 15) * 12
test_windows = choose_random_slices(
    price_df.index, 
    100, 
    300, 
    len(price_df) / step, 
    k = int(step)
)

test_windows = rolling_slices(
    price_df.index, 
    100, 
    int((60 / 15) * 12), 
    int((60 / 15) * 24 * 5)
)
    
    
import concurrent.futures

lstm_returns = []
dca_returns = []

def annualize_return(r, start_date, end_date):
    #TODO: make more flexibile to other periods
    n_days = max([(end_date - start_date).days, 1])
    return (r + 1) ** (365 / n_days) - 1

def calculate_returns(window):
    print(f"{window[0]} to {window[-1]}")
    lstm_values = test_period(window, price_df)
    dca_values = dca_test(window, price_df)
    
    lstm_return = annualize_return((lstm_values.iloc[-1] - lstm_values.iloc[0]) / lstm_values.iloc[0] , window[0], window[-1])
    dca_return = annualize_return((dca_values.iloc[-1] - dca_values.iloc[0]) / dca_values.iloc[0] , window[0], window[-1])
    
    return lstm_return, dca_return

with concurrent.futures.ThreadPoolExecutor(max_workers = 5) as executor:
    results = executor.map(calculate_returns, test_windows)
    for lstm_return, dca_return in results:
        lstm_returns.append(lstm_return)
        dca_returns.append(dca_return)

print(f"LSTM Average: {np.mean(lstm_returns)}")
print(f"DCA Average: {np.mean(dca_returns)}")
print(f"LSTM beats DCA in {np.sum(np.array(lstm_returns) > np.array(dca_returns))} out of {len(lstm_returns)} cases")
    