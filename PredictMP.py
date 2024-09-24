import pandas as pd
import numpy as np
import keras
from keras import backend as K
import tensorflow as tf
from TimeSeriesGeneratorAdapt import MPTimeseriesGenerator
def PredictMPValue(Array_LAMP_Models, matrix_profile_window, seg, graphs, sessions):

        #variables for neural network
    print(seg.shape) 
        #num time dimensions to consider
    n_input_series= 1
    sample_rate = 20
    lookbehind_seconds = 0
    lookahead_seconds = 0
    lookbehind = sample_rate * lookbehind_seconds
    num_outputs = 256
    lookahead = sample_rate * lookahead_seconds
    forward_sequences = lookahead + num_outputs
    subsequences_per_input = lookbehind + num_outputs + lookahead
    channel_stride=8
    high_weight = 1
    low_thresh = -1
    high_thresh = 1
    input_width=matrix_profile_window
    num_outputs=256

    predictions = np.zeros(((((len(seg)- matrix_profile_window + 1) // 256) * 256),len(Array_LAMP_Models)))   #initialize values
    ts = np.array(seg)
    mp = np.zeros((len(ts) - matrix_profile_window + 1, 1))
    
    #for i, LAMP_Model in enumerate(Array_LAMP_Models):
    test_gen =  MPTimeseriesGenerator(ts, mp, num_input_timeseries=n_input_series, internal_stride=channel_stride, num_outputs=num_outputs,lookahead=forward_sequences, lookbehind=lookbehind, important_upper_threshold=high_thresh, important_lower_threshold=low_thresh, important_weight=high_weight, length=input_width, mp_window=matrix_profile_window, stride=num_outputs, batch_size=128)
    for i, LAMP_Model in enumerate(Array_LAMP_Models):
        if graphs[i] is not None:
          curr_graph = graphs[i]
          print('Setting Session for model index ',i)
          with curr_graph.as_default():
            curr_session = sessions[i]
            with curr_session.as_default():
              predictions[:,i] = LAMP_Model.predict_generator(test_gen, verbose=1, use_multiprocessing=True, workers=6).flatten()
        else:
          predictions[:,i] = LAMP_Model.predict_generator(test_gen, verbose=1, use_multiprocessing=True, workers=6).flatten()
    return predictions




# Takes predictions array where each column represents predictions for a specific model
# returns any columns containing a value exceeding the threshold
def Get_Columns_With_Values_Exceeding_Threshold(Predictions,Threshold_Value):

    #for each column of Predictions:
    Column_Numbers_With_Values_Exceeding_Threshold = []
    for column_number in range(0,np.size(Predictions,1)):
        Preds_Of_Individual_Model = Predictions[:,column_number]
        for value_at_index in np.nditer(Preds_Of_Individual_Model):
            if value_at_index > Threshold_Value:
                Column_Numbers_With_Values_Exceeding_Threshold.append(column_number)
                break
                #add column to list of which models to check the children of
    return Column_Numbers_With_Values_Exceeding_Threshold
