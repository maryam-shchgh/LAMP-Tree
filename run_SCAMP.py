
import argparse
import numpy as np
import datetime
import pandas as pd
import os
import sys
import scipy.io as sio
from PredictMP import PredictMPValue
from TrainLAMP import Train_Lamp_With_Target
import json
import shutil
from keras.models import load_model
import tensorflow as tf
import itertools
import random
import time
import math
from SCAMP import *
import time


def Get_test_data(test_data_file, bucket_width, init_seg=None, final_seg=None):
  print('Getting test segments...')
  test_segs = Get_Segments(test_data_file, bucket_width)
  if init_seg is None:
    init_seg = 0
  if final_seg is None:
    final_seg = len(test_segs)

  for seg_idx, seg in enumerate(test_segs):

    if seg_idx < init_seg:
      continue
    if seg_idx > final_seg :
      break
    print('Adding seg ', seg_idx, 'to test set.')
    if seg_idx == init_seg :
      test_data = [seg]
    else:
      test_data.append(seg)

  ts = join_series(test_data)
  print('Test set shape:', ts.shape)
  print('Test set is created.')
  return ts



def main():

  parser = argparse.ArgumentParser(description = 'Consturcting a tree based LAMP structure') 
  parser.add_argument('-Ts', type = str, required=True, help = 'Training data path')
  parser.add_argument('-w', type = int, required=True, help = 'Window size')
  parser.add_argument('-tile', type = int, required=True, help = 'Tile size')
  parser.add_argument('-test_tile', type = int, help = 'Tile size')
  parser.add_argument('-iSeg', type = int, help = 'Set the initial segment to a specific number')
  parser.add_argument('-fSeg', type = int, help = 'Set the final segment to a specific number')
  parser.add_argument('-test', action = 'store_true', help = 'Initializing a new Tree')
  parser.add_argument('-itSeg', type = int, help = 'Set the initial segment to a specific number')
  parser.add_argument('-ftSeg', type = int, help = 'Set the final segment to a specific number')
  parser.add_argument('-out', type = str, help = 'Output files path - Required when initializing')


  args = parser.parse_args()
  window_size = args.w
  train_data = args.Ts
  init_seg = args.iSeg
  final_seg = args.fSeg
  evaluation = args.test
  init_test_seg = args.itSeg
  final_test_seg = args.ftSeg
  tile_size = args.tile
  test_tile_size = args.test_tile
  out_path = args.out

  mp_path = os.path.join('Info','mp_results',out_path)
  SCAMP_time = [0]

  if evaluation:
    ts =  Get_test_data(train_data, test_tile_size, init_test_seg, final_test_seg)

  train_segs = Get_Segments(train_data, tile_size)

  if init_seg is None:
    init_seg = 0
  if final_seg is None:
    final_seg = len(train_segs)
  for seg_idx, LAMP_Data in enumerate(train_segs):
    if seg_idx > final_seg :
      break

    if evaluation : 
        if not os.path.exists(mp_path):
          os.makedirs(mp_path)
        name1 = 'AB_join_node_'+str(seg_idx)+'_and_test_data_tile_'+str(test_tile_size)
        name2 = 'BA_join_test_data_and_node_'+str(seg_idx)+'_tile_'+str(test_tile_size)
        print('Calculating: ', name1)
        fname1 = mp_path + '/' + name1
        fname2 = mp_path + '/' + name2
        if not os.path.exists(fname1) or not os.path.exists(fname2):
            start = time.time()
            mp_AB, mp_BA = get_mp_with_SCAMP(window_size, ts, LAMP_Data)
            end = time.time()
            SCAMP_time[0] += (end - start)
            f = mp_path + '/' + 'tot_SCAMP_Time.txt'
            np.savetxt(f, SCAMP_time, fmt='%1.4f')

            print('saving node results')
        if not os.path.exists(fname1):
            np.savetxt(fname1, mp_AB, fmt='%10.5f', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
            print(fname1, ' Saved!')
        else:
            print(">> ",fname1," exist. Saving skipped.")
        if not os.path.exists(fname2):
            np.savetxt(fname2, mp_BA, fmt='%10.5f', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
            print(fname2, ' Saved!')
        else:
            print(">> ",fname2," exist. Saving skipped.")
        

    else:
      for seg_idx2, LAMP_Data2 in enumerate(train_segs):
        if seg_idx2 > seg_idx :
          break
        else:
          if not os.path.exists(mp_path):
              os.makedirs(mp_path)
          if seg_idx2 == seg_idx : 
            name = 'self_join_node_'+str(seg_idx2)+'_tile_'+str(tile_size)
            print('Calculating: ', name)
            mp_BB = get_mp_with_SCAMP(window_size, LAMP_Data2)
            fname = mp_path + '/' + name
            print('saving node results')
            np.savetxt(fname, mp_BB, fmt='%10.5f', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
            print('Saved!')
          else:
            name1 = 'AB_join_node_'+str(seg_idx)+'_and_node_'+str(seg_idx2)+'_tile_'+str(tile_size)
            name2 = 'BA_join_node_'+str(seg_idx2)+'_and_node_'+str(seg_idx)+'_tile_'+str(tile_size)
            print('Calculating: ', name1)
            fname1 = mp_path + '/' + name1
            fname2 = mp_path + '/' + name2
            if not os.path.exists(fname1) or not os.path.exists(fname2):
                mp_AB, mp_BA = get_mp_with_SCAMP(window_size, LAMP_Data2, LAMP_Data)
                print('saving node results')
            if not os.path.exists(fname1):
                np.savetxt(fname1, mp_AB, fmt='%10.5f', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
                print(fname1, ' Saved!')
            else:
                print(">> ",fname1," exist. Saving skipped.")
            if not os.path.exists(fname2):
                np.savetxt(fname2, mp_BA, fmt='%10.5f', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
                print(fname2, ' Saved!')
            else:
                print(">> ",fname2," exist. Saving skipped.")




if __name__=="__main__":
    main()
  
