# === EVALUATION ==== #
from TreeStruct import TreeStruct
import numpy as np
import pandas as pd
import math
import os
import scipy.io as sio
import argparse
import time
import datetime
from TimeSeriesGeneratorAdapt import MPTimeseriesGenerator
from SCAMP import *
import warnings


#  X X X X X X X
#      X X X X X X X
#          X X X X X X X
#
#  X X X X X X X X X X X  
#
# W + (H - 1) * stride
#

# 
# Flattens a 2d strided representation into a flattened array where each overlapping segment is averaged
def flatten_preds(preds, stride):
  flat_preds = np.zeros((preds.shape[1] + (preds.shape[0] - 1) * stride,))
  width = preds.shape[1]
  # Sum of corresponding segments
  for i in range(0, preds.shape[0]):
    flat_preds[i*stride:i*stride+width] += preds[i,:]

  segs = preds.shape[1] // stride

  # Convert sum to mean for beginning and end of data
  for i in range(0, segs):
    flat_preds[i*stride:(i+1)*stride] /= (i+1)
    flat_preds[-(i+1)*stride:-i*stride] /= (i+1)
    #print('**',flat_preds)

  # Convert sum to mean for the rest of the data
  flat_preds[segs*stride:-segs*stride] /= segs

  return flat_preds



def flatten(preds, stride):
  flat_preds = np.zeros((preds.shape[1] + (preds.shape[0] - 1) * stride,))
  denm_preds = np.zeros((preds.shape[1] + (preds.shape[0] - 1) * stride,))
  width = preds.shape[1]
  # Sum of corresponding segments
  for i in range(0, preds.shape[0]):
    flat_preds[i*stride:i*stride+width] += preds[i,:]
    denm_preds[i*stride:i*stride+width] += 1

  return np.divide(flat_preds,denm_preds)







def exact_search(index, graph, num_buckets, bucket_width, stride, internal_stride, min_match, max_match):

  # Subsequences is a window by stride / internal_stride
  result = np.zeros(num_buckets)
  w_result = np.zeros(num_buckets)

  all_matches = set()
  for i in range(index, index+stride):
    if i in graph:
      for match in graph[i]:
        if match < max_match and match >= min_match: 
          all_matches.add(match)
  

  if len(all_matches) == 0:
    return result

  all_matches = sorted(all_matches)
  j = 0
  for i in range(num_buckets):
    while j < len(all_matches) and all_matches[j] < i * bucket_width :
      j += 1

    if j == len(all_matches):
      return result

    if all_matches[j] < (i+1) * bucket_width:
      result[i] = 1

  return result
   
    
def graph_const(edges, threshold):
  graph = {}
  v_0 = [item[0] for item in edges]
  v_1 = [item[1] for item in edges]
  weight = [item[2] for item in edges]
  
  for i, vertex in enumerate(v_0):
    if weight[i] < threshold:
      continue
    if vertex in graph:
      graph[vertex].append(v_1[i])
    else:
      graph.update({vertex : [v_1[i]]})
  for i, vertex in enumerate(v_1):
    if weight[i] < threshold:
      continue
    if vertex in graph:
      graph[vertex].append(v_0[i])
    else:
      graph.update({vertex : [v_0[i]]})

  return graph



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



def indexingAccuracy(num_buckets, flattened_predictions, actuals, th, memory, leavesTN, leavesTP, leavesFN, leavesFP, leavesMemory):

      TN = 0 
      TP = 0 
      FN = 0 
      FP = 0
      th1 = th
      th2 = th - (0.02*th)
      actual = None
      prediction = None

      for bucket_idx in range(num_buckets):

        preds = flattened_predictions[bucket_idx,:]
        acts = actuals[bucket_idx, :]

        for s in range(actuals.shape[1]):

            # Prediction state
            if acts[s] >= th1:
                actual = True
            else:
                actual = False

            #Without Margin
            #TODO
            if preds[s] >= th1:
                prediction = True
            else:
                prediction = False
            
            
            #Using Margin
            '''
            if preds[s] >= th1 : 
                prediction = True
            elif preds[s] < th2:
                prediction = False
            else:
                #TODO
                if preds[s] != leavesMemory[bucket_idx] and preds[s]>=th2:
                  prediction = True
                else:
                  prediction = False
            '''

            # Evaluation
            if actual and prediction:
                leavesTP[bucket_idx] += 1
                TP += 1
            elif actual and not(prediction):
                leavesFN[bucket_idx] += 1
                FN += 1
            elif not(actual) and not(prediction):
                leavesTN[bucket_idx] += 1
                TN += 1
            elif not(actual) and prediction:
                leavesFP[bucket_idx] += 1
                FP += 1

            #TODO
            leavesMemory[bucket_idx] = preds[s]

      return TN, TP, FN, FP, memory, leavesTN, leavesTP, leavesFN, leavesFP, leavesMemory





def runMSE(MSE_res, num_buckets, flattened_predictions, actuals, batch_idx):
      TN = 0 
      TP = 0 
      FN = 0 
      FP = 0 
      SSE = np.zeros((num_buckets))
      for bucket_idx in range(num_buckets):
        SSE[bucket_idx] = np.sum((flattened_predictions[bucket_idx,:] - actuals[bucket_idx, :]) ** 2)
        curr_MSE = SSE[bucket_idx]/ actuals.shape[1]
        #print('Bucket ' + str(bucket_idx) +  ' MSE: ', curr_MSE)  
        MSE_res[bucket_idx, batch_idx] = curr_MSE
      sum_of_squared_error = np.sum(SSE[:])
      return(sum_of_squared_error / (flattened_predictions.shape[0] * flattened_predictions.shape[1]))





def output_prediction_results(pPath,flattened_preds,exact, MSE, nodeMSE, num_buckets):
    #print('>> Total flattened predictions shape: ', flattened_preds.shape)
    pred_file = 'pred_corr.csv'
    p_file = os.path.join(pPath, pred_file)
    np.savetxt(p_file, flattened_preds, fmt='%1.4f', delimiter=',', newline='\n')
    print('>> Predicted correlations saved in "',p_file,'"')
    if exact:
      MSE_file = 'MSE.csv'
      mse_file = os.path.join(pPath, MSE_file)
      np.savetxt(mse_file, MSE, fmt='%1.4f', delimiter=',', newline='\n')
      print('>> Mean squared errors saved in "',mse_file,'"')
      for bucket_idx in range(num_buckets):
        nodeMSE[bucket_idx] = sum(MSE[bucket_idx, :]/len(MSE[bucket_idx]))
      MSE_per_node_file = 'MSE_per_Node.csv'
      mse_p_node_file = os.path.join(pPath, MSE_per_node_file)
      np.savetxt(mse_p_node_file, nodeMSE, fmt='%1.4f', delimiter=',', newline='\n')
      print('>> Mean squared errors per node saved in "',mse_p_node_file,'"')









def main():
  
  stride = 256
  #start_index = 150000000
  start_index = 0
  min_match = 0
  max_match = 20000000
  model_stride = 256
  num_overlapping_queries = 8
  model_internal_stride = 8

  #Command line args
  parser = argparse.ArgumentParser(description = '--- Evaluation ---') 
  parser.add_argument('-Es','-Test', required=True, type = str, help = 'Test set file directory')
  parser.add_argument('-th','-threshold', required=True, type = float, help = 'Correlation threshold')
  parser.add_argument('-w','-window', required=True, type = int, help = 'Window size')
  parser.add_argument('-o','-out', required=True, type = str, help = 'Output file name.')
  parser.add_argument('-so','-sout', type = str, help = 'Output file name.')
  parser.add_argument('-nleaf', required=True, type = int, help = 'Number of leaf nodes')
  parser.add_argument('-mp', type = int, help = 'Exact search mode: None - (1)compute MP - (2) use actual MP file')
  parser.add_argument('-actuals', type = str, help = 'Actual MP file')
  #parser.add_argument('-mp', action='store_true', help = 'Set to run the exact search')
  parser.add_argument('-s','-Struct', type = str, help = 'Tree structure file')
  parser.add_argument('-bw','-bwidth', type = int, help = 'Exact search bucket width')
  parser.add_argument('-tL','-treeLevel', type = int, help = 'Level of tree to do the prediction on - Do not set to use the whole structure for prediction.')
  parser.add_argument('-iSeg', type = int, help = 'Set the initial segment to a specific number')
  parser.add_argument('-fSeg', type = int, help = 'Set the final segment to a specific number')

  args = parser.parse_args()
  test_data_file = args.Es
  mp_mode = args.mp
  mp_file = args.actuals
  trained_structure = args.s
  bucket_width = args.bw
  mp_window = args.w
  threshold = args.th
  leaf_nodes_count = args.nleaf
  TreeLevel = args.tL
  out_fname = args.o
  scamp_out_fname = args.so
  init_seg = args.iSeg
  final_seg = args.fSeg


  num_buckets = leaf_nodes_count
  tot_TN = 0
  tot_TP = 0
  tot_FN = 0
  tot_FP = 0
  pred_time = 0
  predicting = False
  e_search = False
  mem = 0
  num_epochs = 0
  #TODO
  leavesTN = np.zeros((num_buckets,))
  leavesTP = np.zeros((num_buckets,))
  leavesFN = np.zeros((num_buckets,))
  leavesFP = np.zeros((num_buckets,))
  leavesM = np.zeros((num_buckets,))


  pred_res_path='Info/predictions/'+out_fname
  if not os.path.exists(pred_res_path):
    os.makedirs(pred_res_path)

  if test_data_file is None:
    print('Test file is not set. Use -Es to specify the test file.')
    return

  if init_seg is None and final_seg is None:
    print('Either initial segment index (-iSeg) or final segment index (-fSeg) has to be specified.')
    return

  ts =  Get_test_data(test_data_file, bucket_width, init_seg, final_seg)
  mp = np.zeros((len(ts) - mp_window + 1, 1))



  #Define a new object of 'TreeStrcut' class to do predictions
  if trained_structure is not None:
    tree = TreeStruct(trained_structure)
    predicting  = True
    if TreeLevel is None:
      print('Predicting on the whole struture.')
      pred_fname = 'Tree_predicted_mp.csv'
      pred_mp_file = os.path.join(pred_res_path, pred_fname)
    else:
      print('Predicting on level ',TreeLevel, 'of the struture.')
      pred_fname = 'L_{}_predicted_mp.csv'.format(TreeLevel)
      pred_mp_file = os.path.join(pred_res_path, pred_fname)

  SCAMP_time = 0
  # For each leaf node, do AB join with ts to generage
  # Shape of output is #leaves by (len(ts) - window + 1)
  if mp_mode == 1:
    print('Calculating the exact mp.')
    start = time.time()
    if TreeLevel is None:
      exact_reference = tree.get_exact_results(ts, mp_window, bucket_width, scamp_out_fname)
    else:
      exact_reference = tree.get_exact_results(ts, mp_window, bucket_width, scamp_out_fname, level=TreeLevel)
    end = time.time()
    SCAMP_time += end-start
    actual_fname = 'actual_mp.csv'
    actual_mp_file = os.path.join(pred_res_path, actual_fname)
    pd.DataFrame(exact_reference).to_csv(actual_mp_file,header=False)
    print('Total actual mp saved in "actual_mp.csv". ')
    e_search = True
  elif mp_mode == 2:
    warnings.warn('Loading MP file ...')
    if mp_file is None:
        return
    df = pd.read_csv(mp_file, header=None)
    warnings.warn('Converting MP dataframe to numpy array...')
    exact_reference = np.array(df.iloc[:,1:])
    print('size: ', exact_reference.shape)
    print('Actual MP file is loaded.')
    e_search = True


  if not(predicting) and not(e_search):
    print('Prediction/ Exact search is not set. At least specify the trained structure using -s flag.')
    return
  


  generator_stride = model_stride // num_overlapping_queries

  generator = MPTimeseriesGenerator(ts, mp , num_input_timeseries=1, internal_stride=model_internal_stride, num_outputs=model_stride, lookahead=model_stride, lookbehind=0, length=mp_window, mp_window=mp_window, stride=generator_stride, batch_size=num_overlapping_queries, shuffle=False, merge_points=None)

  results_per_query = np.zeros((len(generator),4))
  MSE_res = np.zeros((num_buckets, len(generator)))
  node_MSE = np.zeros((num_buckets))
  tot_pred_width = model_stride + (num_overlapping_queries -1) * generator_stride
  tot_predictions = np.zeros((num_buckets,len(generator), tot_pred_width))
  tot_flattened_predictions = np.zeros((num_buckets, tot_pred_width + (len(generator) - 1) * model_stride))

  for i, batch in enumerate(generator):

    num_epochs += 1  

    curr_TN = 0
    curr_TP = 0
    curr_FN = 0
    curr_FP = 0
    batch = batch[0]
    predictions = np.zeros((num_buckets,len(batch),model_stride))
    flattened_predictions = np.zeros((num_buckets,model_stride + (len(batch) - 1) * generator_stride))
    actuals = np.zeros((num_buckets,model_stride + (len(batch) - 1) * generator_stride))
    #SSE = np.zeros((num_buckets))


    for batch_idx, item in enumerate(batch):
      item = np.expand_dims(item, axis=0)

      if predicting:
        predictions[:,batch_idx,:], prediction_time = tree.predict_mp_vals(item, threshold, num_buckets, model_stride, TreeLevel)
        pred_time += prediction_time

    for bucket_idx in range(num_buckets):
      flattened_predictions[bucket_idx,:] = flatten(predictions[bucket_idx,:,:], generator_stride)

    tot_predictions[:, i,:flattened_predictions.shape[1]] =  flattened_predictions


    if e_search:
      st = start_index + i * model_stride
      #print('exact ref shape: ',exact_reference.shape)
      #print('start idx: ', st)
      #print('end idx: ', st+flattened_predictions.shape[1])
      actuals = exact_reference[:, st:st+flattened_predictions.shape[1]]
      #print('Actuals shape:', actuals.shape)
      

    if predicting and e_search:
      curr_TN, curr_TP, curr_FN, curr_FP, mem, leavesTN, leavesTP, leavesFN, leavesFP, leavesM= indexingAccuracy(num_buckets, flattened_predictions, actuals, threshold, mem, leavesTN, leavesTP, leavesFN, leavesFP, leavesM)
      batchMSE = runMSE(MSE_res, num_buckets, flattened_predictions, actuals, i)
      print('Predicting ... ',round((i/len(generator))*100,6))
      tot_TN += curr_TN
      tot_TP += curr_TP
      tot_FN += curr_FN
      tot_FP += curr_FP
      #print('MSE: ', batchMSE)

  for bucket_idx in range(num_buckets):
    tot_flattened_predictions[bucket_idx,:] = flatten(tot_predictions[bucket_idx,:,:], model_stride)


  if predicting:
    print('>> Total Number of epochs:', num_epochs)
    #print('>> Total SCAMP Time:', SCAMP_time)
    print('>> Total Prediction Time:', pred_time)
    print('>> Accuracy : ', (tot_TN+tot_TP)/(tot_TN+tot_TP+tot_FN+tot_FP))

    prec = None
    rec = None
    f1score = None

    print('TP: ', tot_TP, '| TN: ', tot_TN, '| FP: ', tot_FP, '| FN: ', tot_FN)


    if (tot_TP+tot_FP) != 0 :
      prec = tot_TP/(tot_TP+tot_FP)
    if (tot_TP+tot_FN) != 0 :
      rec = tot_TP/(tot_TP+tot_FN)
    if (prec is not None and rec is not None) or (prec != 0 and rec != 0):
      f1score = 2*(prec*rec)/(prec+rec)

    print('>> Precision : ', prec)
    print('>> Recall : ', rec)
    print('>> F1-score : ', f1score)

    leafPrecision = np.zeros((leaf_nodes_count,))
    leafRecall = np.zeros((leaf_nodes_count,))
    leafF1 = np.zeros((leaf_nodes_count,))


    leavesPrec = np.zeros((num_buckets,))
    leavesRec = np.zeros((num_buckets,))
    leavesF1 = np.zeros((num_buckets,))

    for b in range(num_buckets):
      if (leavesTP[b]+leavesFP[b]) != 0 :
        leavesPrec[b] = leavesTP[b]/(leavesTP[b]+leavesFP[b])
      if (leavesTP[b]+leavesFN[b]) != 0 :
        leavesRec[b] = leavesTP[b]/(leavesTP[b]+leavesFN[b])
      if leavesRec[b] is not None and leavesPrec[b] is not None:
        leavesF1[b] = 2*(leavesPrec[b]*leavesRec[b])/(leavesPrec[b]+leavesRec[b])

    print('>> leaves Precision : ', leavesPrec)
    print('>> leaves Recall : ', leavesRec)
    print('>> leaves F1-score : ', leavesF1)
    print('>> averages (p-r-f): ', sum(leavesPrec)/num_buckets,',',sum(leavesRec)/num_buckets,',',sum(leavesF1)/num_buckets)

    output_prediction_results(pred_res_path,tot_flattened_predictions,e_search, MSE_res, node_MSE, num_buckets)



if __name__=="__main__":
  main()
