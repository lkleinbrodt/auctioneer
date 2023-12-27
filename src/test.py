
%load_ext autoreload
%autoreload 2
from lstm import *
from s3 import S3Client

s3 = S3Client()
# s3.download_file('models/test/lstm_best.pt', 'lstm_best.pt')
# s3.download_file('models/test/lstm_best_startup_params.json', 'lstm_best_startup_params.json')
# s3.read_csv('models/test/lstm_history.csv').set_index('epoch').plot()
model = load_model_from_params('lstm_best.pt')
time_series = get_time_series() 
train, val, test = split_time_series(time_series)
pd.Series(time_series).plot()
s3.read_csv('models/test/test_preds.csv')[['actual', 'preds']].plot()