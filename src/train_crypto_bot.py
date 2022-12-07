from auctioneer import *
from tempfile import TemporaryDirectory
import joblib


logger = create_logger('crypto_bot')

START_DATE = '2022-01-01' #none for full
END_DATE = '2021-03-01' #none for full

HISTORY_STEPS = 240
TARGET_STEPS = 30
MAX_EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = .0005

MODELS_PATH = '../models/'

def load_data(s3, start_date, end_date=None):
    logger.info('Reading data')

    data = load_s3_csv(s3, 'minute_crypto_prices.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.set_index('timestamp')

    if start_date is not None:
        logger.info(f'Original shape: {data.shape}')
        start_date = pd.to_datetime(start_date).tz_localize('US/Pacific')
        data = data.loc[start_date:]
        logger.info(f'After clipping start date: {data.shape}')

    if end_date is not None:
        logger.info(f'Original shape: {data.shape}')
        end_date = pd.to_datetime(end_date).tz_localize('US/Pacific')
        data = data.loc[:end_date]
        logger.info(f'After clipping end date: {data.shape}')
        
    return data

if not os.path.isdir(MODELS_PATH):
    os.mkdir(MODELS_PATH)
    os.mkdir(MODELS_PATH+'checkpoints/')

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



def train_encoder_model(df, history_steps, target_steps, max_epochs, batch_size, learning_rate):
    logger.info('Training Model')
    X_train, Y_train, X_test, Y_test, scalers = window_data(df, history_steps, target_steps)
    logger.info('Done windowing data')
    n_features = X_train.shape[2]
    
    model = encoder_model(history_steps, target_steps, n_features)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate, 
        decay_steps = int(X_train.shape[0] / batch_size) * 2, 
        decay_rate = .96
    )
    
    model.compile(
        optimizer = tf.keras.optimizers.Adam(), 
        loss = tf.keras.losses.Huber(),
        # learning_rate = lr_schedule
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(patience = 5, restore_best_weights = True)
    model_checkpoints = tf.keras.callbacks.ModelCheckpoint(MODELS_PATH+'checkpoints/', save_best_only = True, save_weights_only = True)
    date = df.index.max().strftime('%Y%m%d')
    #[os.remove(os.path.join('Logs/Tensorboard', f)) for f in os.listdir('Logs/Tensorboard')]
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir = MODELS_PATH + 'Tensorboard/' + date)
    my_callbacks = [early_stopping]

    model.fit(
        X_train, Y_train, 
        epochs = max_epochs, 
        validation_data = (X_test, Y_test), 
        batch_size = batch_size, 
        callbacks = my_callbacks,
        verbose = 1
        )

    # model.load_weights(MODELS_PATH+'checkpoints/')
    # model.save(MODELS_PATH+'TrainedModel')

    return model, scalers

def main():
    logger.info('---START---')
    s3 = create_s3()

    data = load_data(s3, START_DATE, END_DATE)

    model, scalers = train_encoder_model(
        df = data,
        history_steps=HISTORY_STEPS,
        target_steps=TARGET_STEPS,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
    logger.info('Done Training model')


    logger.info('Saving model and scaler')
    with TemporaryDirectory() as tempdir:
        logger.info('saving to temp folder')
        model.save(f"{tempdir}/TrainedModel")
        joblib.dump(scalers, f"{tempdir}/scalers.gz")

        logger.info('uploading to s3')
        s3.upload_file(f"{tempdir}/scalers.gz", S3_BUCKET, 'scalers.gz')
        upload_directory(s3, f"{tempdir}/TrainedModel", 'TrainedModel')
    
    logger.info('---END---')


if __name__ == '__main__':
    main()