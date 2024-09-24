import numpy as np
import math
import pandas as pd
import tensorflow as tf
import sys,os
from TimeSeriesGeneratorAdapt import MPTimeseriesGenerator
import keras
from scipy.stats import zscore
import scipy.io as sio
from keras.utils import multi_gpu_model
from tensorboard_callbacks import LRTensorBoard
from models import build_resnet
import datetime
import keras.backend as K
from keras.callbacks import EarlyStopping



def Train_Lamp_With_Target(window_size_input,nb_epochs, seg, target,merging = True, model_initialization=None, graph = None, session = None):
    K.clear_session()
    root_logging_path = 'log_files'
    np.random.seed(813306)
    from tensorflow import set_random_seed
    set_random_seed(5944)
    high_weight = 1
    low_thresh = -1
    high_thresh = 1
    initial_lr=1e-3
    optimizer_id = 'Adam'
    loss_function='mse'
    batch_size = 32
    init_epoch = 0
    sample_rate = 20
    num_outputs = 256
    n_input_series= 1
    train_test_split = 0.8
    channel_stride = 8
    lookbehind_seconds = 0
    lookahead_seconds = 0
    subsequence_stride = 256
    lookbehind = sample_rate * lookbehind_seconds
    lookahead = sample_rate * lookahead_seconds
    forward_sequences = lookahead + num_outputs
    subsequences_per_input = lookbehind + num_outputs + lookahead
    subsequences_per_input = subsequences_per_input // channel_stride
    matrix_profile_window = int(window_size_input)
    input_width = matrix_profile_window

    shuffle = True
    stateful_rnn = False
    optimizer = keras.optimizers.get(optimizer_id)
    conf = optimizer.get_config()
    conf['lr'] = initial_lr
    conf['epsilon'] = 1e-4


    if model_initialization is not None:
      with graph.as_default():
        with session.as_default():
          model = model_initialization
    else:
      with tf.device("/gpu:0"):
        model = build_resnet([input_width, n_input_series, subsequences_per_input], 96, num_outputs)
      model.compile(loss=loss_function,optimizer=optimizer)
    model.summary()
    
    if merging:
      ts = seg[:int(len(seg) * train_test_split)]
      mp = target[:len(ts) - matrix_profile_window + 1]
      mp_val = target[len(ts):]
      ts_val = seg[len(ts):len(ts)+len(mp_val)+ matrix_profile_window - 1]
      mp.shape = (mp.shape[0], 1)
      mp_val.shape = (mp_val.shape[0], 1)

      merge_points_train = np.nonzero(pd.isnull(ts))[0]
      if len(merge_points_train) == 0:
        merge_points_train = None

      merge_points_val = np.nonzero(pd.isnull(ts_val))[0]
      if len(merge_points_val) == 0:
        merge_points_val = None

      dataset_name = 'merged'
      models_path = 'merged_models'


    else:
      ts = seg
      mp = target

      #merge_points_train = np.nonzero(np.isnan(ts))[0]
      merge_points_train = np.nonzero(pd.isnull(ts))[0]
      if len(merge_points_train) == 0:
        merge_points_train = None

      mp.shape = (mp.shape[0], 1)
      dataset_name = 'trained_models'
      models_path = 'trained_retrained_models'


    tensorboard_log_path = 'tensorboard_logs'
    csv_log_path = 'csv_logs'
    logging_filename = 'dataset={}_width={}_optimizer={}_initlr={}_batchsz={}_stride={}_shuffle={}_lookbehind={}_lookahead={}_channelstride={}_outputs={}_stateful={}_weight={}_lowthresh={}_highthresh={}_started={}'.format(dataset_name, input_width, optimizer_id, initial_lr, batch_size, subsequence_stride, shuffle, lookbehind, lookahead, channel_stride, num_outputs, stateful_rnn, high_weight, low_thresh, high_thresh, str(datetime.datetime.now()).replace(' ', '-'))
    model_path = os.path.join(root_logging_path, models_path, logging_filename)
    tensorboard_path = os.path.join(root_logging_path, tensorboard_log_path , logging_filename)
    logging_path = os.path.join(root_logging_path, csv_log_path , logging_filename)
    logging_dir = os.path.join(root_logging_path, csv_log_path)

    if not os.path.exists(logging_dir):
      os.makedirs(logging_dir)
    if not os.path.exists(model_path):
      os.makedirs(model_path)
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    train_gen =  MPTimeseriesGenerator(ts, mp, num_input_timeseries=n_input_series, internal_stride=channel_stride, num_outputs=num_outputs, lookahead=forward_sequences, lookbehind=lookbehind, length=input_width, mp_window=matrix_profile_window, stride=subsequence_stride, important_upper_threshold=high_thresh, important_lower_threshold=low_thresh, important_weight=high_weight, batch_size=batch_size, shuffle=shuffle, merge_points=merge_points_train)
    print('Train size: ' + str(len(train_gen) * batch_size))

    if merging:
      valid_gen =  MPTimeseriesGenerator(ts_val, mp_val, num_input_timeseries=n_input_series, internal_stride=channel_stride, num_outputs=num_outputs,lookahead=forward_sequences, lookbehind=lookbehind, important_upper_threshold=high_thresh, important_lower_threshold=low_thresh, important_weight=high_weight, length=input_width, mp_window=matrix_profile_window, stride=num_outputs, batch_size=batch_size, merge_points=merge_points_val)
      print('Validation size: ' + str(len(valid_gen) * batch_size))
    else:
      valid_gen = None

    if not os.path.exists(model_path):
      os.makedirs(model_path)
    save_path = os.path.join(model_path,'weights.{epoch:02d}-{loss:.5f}.hdf5')
    checkpoint = keras.callbacks.ModelCheckpoint(save_path, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    tensorb = LRTensorBoard(log_dir=tensorboard_path)
    logging = keras.callbacks.CSVLogger(logging_path)

    #FIXME   
    # Early stop callback to stop when no change in loss for 4 epochs
    #early_stop = EarlyStopping(patience=4)
    if graph is not None:
      with graph.as_default():
        with session.as_default():
          hist = model.fit_generator(train_gen, workers=1, use_multiprocessing=False, validation_data=valid_gen, shuffle=shuffle, epochs=nb_epochs,verbose=1, initial_epoch=init_epoch)
    else:
      hist = model.fit_generator(train_gen, workers=1, use_multiprocessing=False, validation_data=valid_gen, shuffle=shuffle, epochs=nb_epochs,verbose=1, initial_epoch=init_epoch)

    return model        
