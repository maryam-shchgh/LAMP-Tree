import os
import math
import numpy as np
import pandas as pd
import pyscamp as mp


SCAMP_PATH = '~/SCAMP/build/SCAMP'



def test():

    a = np.array(pd.read_csv('B.txt',header=None)).flatten()[1:100000]
    print('a shape: ', a.shape)
    has_gpu_support = mp.gpu_supported()

    if has_gpu_support:
        print('Running SCAMP')
        profile, index = mp.selfjoin(a, 100, pearson=True)
        print('Writing to file')
        np.savetxt('test_profile.txt',profile)
        np.savetxt('test_index.txt',index)
        print('done!')






def get_mp_with_SCAMP(window, node_segment,retrain_segment=None):
  
  print("----- Running SCAMP -----")  
  print('node seg shape:', node_segment.shape)
  print(node_segment.dtype) 
  np.savetxt('B.txt', node_segment)
 
  if retrain_segment is not None: 
    print('retrain seg shape:', retrain_segment.shape)
    np.savetxt('A.txt', retrain_segment)

    cmd = SCAMP_PATH + ' --output_pearson --profile_type=1NN --keep_rows --window=' + str(window) + ' --input_b_file_name=B.txt --input_a_file_name=A.txt --output_a_file_name=AB.txt --output_b_file_name=BA.txt'
    z = os.system(cmd) 


    mp_AB = np.array(pd.read_csv('AB.txt', header=None)).flatten()
    mp_BA = np.array(pd.read_csv('BA.txt', header=None)).flatten()
    print("----- SCAMP is done -----")  
    return mp_AB, mp_BA

  else:
    cmd = SCAMP_PATH + ' --output_pearson --profile_type=1NN --window=' + str(window) + ' --input_a_file_name=B.txt  --output_a_file_name=BB.txt'
    z = os.system(cmd) 

    mp_BB = np.array(pd.read_csv('BB.txt', header=None)).flatten()
    print("----- SCAMP is done -----")  
    return mp_BB





def Get_Segments(csv_file_path,segment_len,limit_num_segments=None):
    ascii_segment_list = []
    
    #opened_csv_file = open(csv_file_path, 'r').readlines()
    opened_csv_file = pd.read_csv(csv_file_path, 'r')
    segment_num = 1
    
    num_segments = math.floor(len(opened_csv_file)/segment_len)
    print('seg num: ',num_segments)
    if limit_num_segments is not None:
        for segment_number in range(1,limit_num_segments + 1):
            individual_segment = np.array(opened_csv_file[(segment_number - 1)*int(segment_len):segment_number*int(segment_len)])
            
            individual_segment.shape = (individual_segment.shape[0],1)
            ascii_segment_list.append(individual_segment)
    else:

        for segment_number in range(1,num_segments + 1):
            individual_segment = np.array(opened_csv_file[(segment_number - 1)*int(segment_len):segment_number*int(segment_len)])
            individual_segment.shape = (individual_segment.shape[0],1)
            ascii_segment_list.append(individual_segment)
    
    return ascii_segment_list


def join_series(lst):
  out = lst[0]
  if (len(lst) == 1):
    return out
  for ts in lst[1:]:
    out = np.concatenate((out.flatten(), np.array([np.nan]), ts.flatten()))
  return out


def join_mp(lst, window):
  out = lst[0]
  if (len(lst) == 1):
    return out
  for mp in lst[1:]:
    out = np.concatenate((out.flatten(), np.ones((window,)) * np.nan, mp.flatten()))
  return out 
