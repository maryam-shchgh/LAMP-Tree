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
import pdb


class TreeStruct:
  
  def __init__(self, trained_structure=None, Levels_Const = None, curr_path = None, all_leaf = False):

    self._data_dict = {}
    self._model_dict = {}
    self._levels_dict = {}
    self._children_dict = {}
    self._parents_dict = {}
    self._temp_data_dict = {}
    self._temp_model_dict = {}
    self._retraining_track = {}
    self._loaded_model_dict = {}
    self._graph_session_dict = {}
    self._retraining_node_track = {}
    self._predicted_corr = {}
    self._predicted_high_corr = {}
    self.struct_fname = None
    self.prev_struct_file = None
    self._lim = None

    if Levels_Const is not None:
      self._lim = [int(limit) for limit in Levels_Const]
      self._struct = [{} for i in range(len(self._lim))]

    self._default_lim  = self._lim

    if all_leaf:
      if Levels_Const is not None:
        new_lim =  [1]
        for const in self._lim:
          new_lim[0] *= const
        self._lim = new_lim
        print('All leaf mode is ON. Structure level is set to ', self._lim)

    if trained_structure is not None:
      self = self.load(trained_structure)
 
    if self.struct_fname is None:
      self.struct_fname = 'Tree'
      for item in self._lim:
        self.struct_fname += ('_' + str(item))

    if curr_path is not None:
      self._struct_info_path = os.path.join('Info', curr_path)
      self._struct_json_file_path = os.path.join(self._struct_info_path, 'JSONs')
      if not os.path.exists(self._struct_info_path):
        os.makedirs(self._struct_info_path)
      if not os.path.exists(self._struct_json_file_path):
        os.makedirs(self._struct_json_file_path)



  def Train(self, train_segments, window_size, train_nb_epochs, retraining, max_nb_retraining, retrain_nb_epochs, init_seg = None, final_seg = None):

    if init_seg is None:
      init_seg = 0
    if final_seg is None:
      final_seg = len(train_segments)

    for seg_idx, LAMP_Data in enumerate(train_segments):
      if seg_idx < init_seg:
        continue
      if seg_idx > final_seg :
        break
      
      new_node_training_data = [LAMP_Data]
      print('Calculating train segment mp')
      #TODO
      keyword = 'self_join_node_'+str(seg_idx)+'_tile_'+str(tile_size)
      mp_path = os.path.join('Info','mp_results',scamp_out_path)
      mp_file = mp_path + '/' + keyword
      train_mp = np.loadtxt(mp_file)
      #train_mp = get_mp_with_SCAMP(window_size, LAMP_Data)
      new_node_training_target = [train_mp]
      name = 'node__{}__started__{}'.format(str(seg_idx),str(datetime.datetime.now()).replace(' ','_')) 
      print('TRAINING - Current Node: ',  name)

      if retraining:
        leaf_nodes = self.get_level_nodes(0)
        new_node_training_data, new_node_training_target = self.Retrain(new_node_training_data, new_node_training_target, max_nb_retraining, window_size, name, LAMP_Data, retrain_nb_epochs, train_nb_epochs)
      
      ts_data_concatenated = join_series(new_node_training_data)
      mp_data_concatenated = join_mp(new_node_training_target, window_size)
      print('Train ts shape:',ts_data_concatenated.shape)
      print('Train mp shape:',mp_data_concatenated.shape)

      LAMP_Model = Train_Lamp_With_Target(window_size, train_nb_epochs, ts_data_concatenated, mp_data_concatenated, False, None, None,None)
          

      node_info  = [name, LAMP_Model, LAMP_Data] 

      curr_struct_file = self.add(node_info, train_nb_epochs)

    self.Retrain(None, None, None, window_size, None, None, retrain_nb_epochs, train_nb_epochs)
      
    return curr_struct_file



  def Retrain(self, new_node_training_data, new_node_training_target, max_nb_retraining, window_size, name, LAMP_Data, retrain_nb_epochs, merge_nb_epochs):
    
    if name is not None:

      max_avr_leaf_node_retrain = int(0.7*max_nb_retraining)  

      leaf_nodes = self.get_level_nodes(0)

      if leaf_nodes is None:
        leaf_nodes_length = 0
      else:
        leaf_nodes_length = len(leaf_nodes)

      if leaf_nodes_length > 0:

        print('<< RETRAINING >>')
        random_indexes = self.gen_random_idx(leaf_nodes, max_nb_retraining)
        print('RETRAINING - Generated random indexes: ', random_indexes)
        retraining_candidates = [leaf_nodes[idx] for idx in random_indexes]  

        for node in retraining_candidates:

          new_node_training_data, new_node_training_target, curr_init_ret_count = self.Retrain_node(node, name, window_size, LAMP_Data, retrain_nb_epochs, new_node_training_data, new_node_training_target)
          print('Initial retraing count for node ', name, ' is ', curr_init_ret_count, 'times.')

        checked_parents = []
        for node in retraining_candidates:
          
          if node in self._parents_dict.keys():
        
            parent = self._parents_dict.get(node)
            if parent in checked_parents:
              continue
            else:
              checked_parents.append(parent)

            self.Retrain_parent_node(parent, window_size, merge_nb_epochs)
          
      return new_node_training_data, new_node_training_target

    else:

      #TODO
      print('Training is done - Final update of merged models.')
      children_candidates = list(self._parents_dict.keys())
      checked_parents = []
      for node in children_candidates:
          parent = self._parents_dict.get(node)
          if parent in checked_parents:
            continue
          else:
            checked_parents.append(parent)

          print('Final update of ', parent)
          self.Retrain_parent_node(parent, window_size, merge_nb_epochs, None, final=True)
          
      return
 

  def gen_random_idx(self, leaf_nodes, max_nb_retraining):

    leaf_nodes_length = len(leaf_nodes)
    indexes = list(range(leaf_nodes_length))
    selection_weights = []

    for idx in indexes:
      if leaf_nodes[idx] not in self._retraining_track.keys():
        new_node = leaf_nodes[idx]
        self._retraining_track[new_node] = 0

      target_node_count = self._retraining_track.get(leaf_nodes[idx])
      selection_weights.append(1/(target_node_count + 1))

    selection_weights = selection_weights / np.sum(selection_weights)
    print('Selection Probabilities: ',selection_weights)
      
    if len(selection_weights) < max_nb_retraining:
      max_retraining = len(selection_weights)
    else:
      max_retraining = max_nb_retraining
    random_indexes = np.random.choice(indexes, max_retraining, p=selection_weights, replace=False)

    return random_indexes



  def Retrain_node(self, node, name, window_size, LAMP_Data, retrain_nb_epochs, new_node_training_data, new_node_training_target):

    print('RETRAINING -', node,'- on -', name,'-')
    old_model, B_data = self.get_node_info(node, load_data = True)[-2:]
    print('RETRAINING - Running SCAMP')
    start = time.time()
    #TODO
    keyword1 = 'AB_join_node_'+str(name.split("__")[1])+'_and_node_'+str(node.split("__")[0].split("_")[-1])+'_tile_'+str(tile_size)
    keyword2 = 'BA_join_node_'+str(node.split("__")[0].split("_")[-1])+'_and_node_'+str(name.split("__")[1])+'_tile_'+str(tile_size)
    mp_path = os.path.join('Info','mp_results',scamp_out_path)
    mp_AB = mp_path + '/' + keyword1
    mp_BA = mp_path + '/' + keyword2
    new_mp_data = np.loadtxt(mp_AB)
    ba_mp = np.loadtxt(mp_BA)
    #new_mp_data, ba_mp = get_mp_with_SCAMP(window_size, B_data,LAMP_Data)
    end = time.time()
    new_node_training_data.append(B_data)
    new_node_training_target.append(ba_mp)

    if name not in self._retraining_track.keys():
      curr_init_ret_count = 1
      self._retraining_track[name] = curr_init_ret_count
      self._retraining_node_track[name]=[node]
    else:
      curr_init_ret_count = self._retraining_track.get(name) + 1
      self._retraining_track.update({name:curr_init_ret_count})
      self._retraining_node_track[name].append(node)

    print('RETRAINING - SCAMP done - t: ', round(end-start, 4), 'seconds')
    print('RETRAINING - Start retraining ...')

    start = time.time()
    if node in self._graph_session_dict.keys():
      node_graph = self._graph_session_dict.get(node)[0]
      node_session = self._graph_session_dict.get(node)[1]
    else:
      node_graph = None
      node_session = None

    
    new_model = Train_Lamp_With_Target(window_size, retrain_nb_epochs, LAMP_Data, new_mp_data, False, old_model, node_graph, node_session)
    end = time.time()
    print('RETRAINING Done! t:', round(end-start, 4), 'seconds')
        
    node_retrain_times = self._retraining_track.get(node) 
    node_retrain_times += 1
    self._retraining_track.update({node : node_retrain_times})           
    if node in self._retraining_node_track.keys():
      self._retraining_node_track[node].append(name)
    else:
      self._retraining_node_track[node]=[name]

    if self.update_model_data(node, None, retraining = True, model = new_model, graph = node_graph, session = node_session):
      print('RETRAINING: Model direcotry updated!')
    
    return new_node_training_data, new_node_training_target, curr_init_ret_count


  def Retrain_parent_node(self, parent, window_size, merge_nb_epochs, max_avr_leaf_node_retrain = None, final=False):

    all_children = self._children_dict.get(parent)
    curr_ch_retrain_sum = 0
    children_graphs = []
    children_sessions = []

    for child in all_children:
      #if child in self._graph_session_dict.keys():
      #  graph = self._graph_session_dict.get(child)[0]
      #  session = self._graph_session_dict.get(child)[1]
      #else:
      #  graph = None
      #  session = None
      #children_graphs.append(graph)
      #children_sessions.append(session)
      if max_avr_leaf_node_retrain is not None:
        curr_ch_retrain_sum += self._retraining_track.get(child)

          

    if max_avr_leaf_node_retrain is not None:
      curr_ch_avr_retrain = curr_ch_retrain_sum / len(all_children)
      if parent in self._retraining_track.keys():
        prev_retrain_count = self._retraining_track.get(parent)
      else:
        prev_retrain_count = 0
        self._retraining_track[parent] = prev_retrain_count
      print('Checking merged model ', parent, '--> Prev retrain count: ', prev_retrain_count)              
      print('Comparing- ', curr_ch_avr_retrain, 'vs. ', (max_avr_leaf_node_retrain + prev_retrain_count))
    
    if  (max_avr_leaf_node_retrain is not None and curr_ch_avr_retrain >= (max_avr_leaf_node_retrain + prev_retrain_count)) or final:

      print('RETRAINING - Updating merged model ', parent)
      children_models = []
      for ch_idx, child in enumerate(all_children):
        child_model, child_data = self.get_node_info(child, load_data = True)[-2:]
        children_models.append(child_model)
        if ch_idx == 0 :
          children_data = np.array(child_data)
        else:
          children_data = np.concatenate((children_data, child_data))

        if child in self._graph_session_dict.keys():
          graph = self._graph_session_dict.get(child)[0]
          session = self._graph_session_dict.get(child)[1]
        else:
          graph = None
          session = None
        children_graphs.append(graph)
        children_sessions.append(session)
      new_merged_model = self.Merge_Models(parent, window_size, merge_nb_epochs, children_models, children_data, children_graphs, children_sessions)
      if self.update_model_data(parent, None, retraining = True, model = new_merged_model):
        print('RETRAINING: Merged model direcotry updated!')

      if max_avr_leaf_node_retrain is not None:
        self._retraining_track.update({parent : (prev_retrain_count + 1)}) 

    return


 
  def add(self, new_model_and_data, merge_nb_epochs):

    n = len(self._lim)

    name = new_model_and_data[0]
    model = new_model_and_data[1]
    data = new_model_and_data[2]
   
    print('Adding model :', name)

    if name in self._model_dict:
      return self._struct
    elif name in self._temp_model_dict:
      return self._struct

    self.add_leaf_node(name, model, data,self._struct_info_path)

    if self.prev_struct_file is not None:
      os.remove(self.prev_struct_file)
    self.prev_struct_file = self.save(self.struct_fname)
    print('Struct file saved: ', datetime.datetime.now())
      
    return self.check_limit(0, merge_nb_epochs)


  def check_limit(self, level, merge_nb_epochs):
    
    n = len(self._lim)

    if len(self._struct[level]) < self._lim[level]:
      return self.prev_struct_file

    else:
        if level == (n-1):
          self._lim[level] += 1
          if self.prev_struct_file is not None:
            os.remove(self.prev_struct_file)
          self.struct_fname = 'Tree'
          for i, item in enumerate(self._lim):
            if i == (n-1):
              item -= 1
            self.struct_fname += ('_' + str(item))
          self.prev_struct_file = self.save(self.struct_fname)
          
          return self.prev_struct_file
        

        matrix_profile_window = 100
        

        if (level+1) in self._levels_dict.keys():
          name_idx = len(self._levels_dict.get(level+1))
        elif str(level+1) in self._levels_dict.keys():
          name_idx = len(self._levels_dict.get(str(level+1)))
        else:
          name_idx = 0
        merged_model_name = 'node_L_{}_i_{}__started__{}'.format(str(level+1), str(name_idx),str(datetime.datetime.now()).replace(' ','_')) 
        
        models_for_merge = []
        model_files_for_merge = []
        graphs = []
        sessions = []
        info_for_merge = self.get_level_info_for_merge(level)
        for m_name, m_info in info_for_merge.get('m_info').items():
           model_files_for_merge.append(m_name)
           models_for_merge.append(m_info.get('model'))
           graphs.append(m_info.get('graph'))
           sessions.append(m_info.get('session'))
        data_for_merging = info_for_merge.get('d_info') 


        print('Merging models:', [model for model in list(self._struct[level].keys())])
        merged_model = self.Merge_Models(merged_model_name, matrix_profile_window, merge_nb_epochs, models_for_merge, data_for_merging, graphs, sessions)
        
        #Creating path for the new structure and dataset
        new_struct_path = os.path.join(self._struct_info_path, merged_model_name)
        if not os.path.exists(new_struct_path):
          os.makedirs(new_struct_path)
         
        model_file, data_file = self.save_model_data_to_path(merged_model_name, merged_model, data_for_merging, new_struct_path)

        for item in model_files_for_merge:
          shutil.move(os.path.join(self._struct_info_path,item),new_struct_path)
          item_new_path = str(os.path.join(new_struct_path, item))
          self.update_model_data(item, item_new_path)
             

        #Saving the merged results
        self.update_structure_info(level, merged_model_name, model_files_for_merge, merged_model, data_for_merging, new_struct_path)

        if self.prev_struct_file is not None:
          os.remove(self.prev_struct_file)
        self.prev_struct_file = self.save(self.struct_fname)
        print('Struct file saved: ', datetime.datetime.now())

        self._struct[level] = {}

        target_level = level + 1
        self.add_node_to_levels_dict(target_level, merged_model_name)

        if level < (n-1) :
          self.check_limit(level+1, merge_nb_epochs)


  def Merge_Models(self, name, window_size, nb_epochs, models, data_segments, graphs, sessions):
    
    window = int(window_size)
    preds = []
    preds = PredictMPValue(models, window_size, data_segments, graphs, sessions)
    max_preds = np.amax(preds, axis=1)
    merged_model= Train_Lamp_With_Target(window_size, nb_epochs ,data_segments,max_preds, True, None)

    return merged_model



  def update_model_data(self, node, path, retraining = False, model = None, data = None, graph=None, session=None):

    if retraining:
      old_model_path_name = self._model_dict.get(node)
      if graph is not None:
        with graph.as_default():
          with session.as_default():
            saved_model = model.save(old_model_path_name)
            print('Model file substitute')
      else:
        saved_model = model.save(old_model_path_name)
        print('Model file substitute')

    else:
      self._model_dict[node] = str(os.path.join(path, node + '.h5'))
      self._data_dict[node] = str(os.path.join(path, node + '.mat'))

      for child in self.get_children(node):
        child_new_path = str(os.path.join(path, child))
        self.update_model_data(child, child_new_path)
    
    
    return 1
  

  def add_leaf_node(self, name, model, data, struct_files_path):

    
    if 0 in self._levels_dict.keys():
      name_idx = len(self._levels_dict.get(0))
    elif '0' in self._levels_dict.keys():
      name_idx = len(self._levels_dict.get('0'))
    else:
      name_idx = 0
    name = 'node_L_0_i_{}__started__{}'.format(str(name_idx),str(datetime.datetime.now()).replace(' ','_')) 
    
    new_struct_path = os.path.join(struct_files_path, name)
    if not os.path.exists(new_struct_path):
      os.makedirs(new_struct_path)
    
    self._struct[0][name] = []
    self._children_dict[name] = [] 
    self.add_node_to_levels_dict(0, name)
    self._model_dict[name] = str(os.path.join(new_struct_path, name + '.h5'))
    self._data_dict[name] = str(os.path.join(new_struct_path, name +'.mat')) 
    model_file, data_file = self.save_model_data_to_path(name, model, data, new_struct_path)


  def add_node_to_levels_dict(self, level, node):
    level = str(level)
    if level in self._levels_dict.keys():
      self._levels_dict[level].append(node)
    else:
      self._levels_dict[level] = [node]


  def update_structure_info(self, level, name, children, model, data, save_dir):
    self._struct[level+1][name] = [self._struct[level]]
    self._children_dict[name] = children
    for child in children:
      self._parents_dict[child] = name
    self._model_dict[name] = str(os.path.join(save_dir, name + '.h5'))
    self._data_dict[name] = str(os.path.join(save_dir, name +'.mat'))

  def save_model_data_to_path(self, name, model, data, path):
    model_file = model.save(os.path.join(path, name + '.h5'))
    data_file = sio.savemat(os.path.join(path, name +'.mat'),{'all_data':data})
    return model_file, data_file


  def get_level_info_for_merge(self, level):

    info_for_merge = {}
    info_for_merge['m_info'] = {}
    info_for_merge['d_info'] = None

    for idx, name in enumerate(list(self._struct[level].keys())):
      if name in self._temp_model_dict:
        model = self._temp_model_dict.get(name)
        graph = self._graph_session_dict.get(name)[0]
        session = self._graph_session_dict.get(name)[1]
      else:
        model_path = self._model_dict.get(name)
        if name in self._graph_session_dict.keys():
          graph = self._graph_session_dict.get(name)[0]
          session = self._graph_session_dict.get(name)[1]
          with graph.as_default():
            with session.as_default():
              model = load_model(model_path)
              self._temp_model_dict[name] = model
        else:
          graph = tf.Graph()
          with graph.as_default():
            session = tf.Session()
            with session.as_default():
              model = load_model(model_path)
              self._graph_session_dict[name] = [graph, session]

        self._temp_model_dict[name] = model
      info_for_merge['m_info'].update({name:{'model': model, 'graph': graph, 'session': session}})

      if name == list(self._struct[level].keys())[0]:
        if name in self._temp_data_dict:
          data = self._temp_data_dict.get(name)
        else:
          loaded_data = sio.loadmat(self._data_dict.get(name))
          data = np.array(loaded_data['all_data'])
      else:
        if name in self._temp_data_dict:
          data = np.concatenate((data, self._temp_data_dict.get(name)))
        else:
          loaded_data = sio.loadmat(self._data_dict.get(name))
          new_data = np.array(loaded_data['all_data'])
          data = np.concatenate((data, new_data))

    info_for_merge.update({'d_info':data})

    return info_for_merge



  def get_node_info(self, node, load_data = False):

    children = self._children_dict.get(node)
    
    if node in list(self._temp_model_dict.keys()):
      model =  self._temp_model_dict.get(node)
    else:
      graph = tf.Graph()
      with graph.as_default():
        session = tf.Session()
        with session.as_default():
          start = time.time()
          model = load_model(self._model_dict.get(node))
          end = time.time()
          print('Loading model time: ', end - start)
      self._graph_session_dict[node] = [graph, session]
      self._temp_model_dict[node] = model
    data = None
    if load_data:
      if node in self._temp_data_dict.keys():
        data = self._temp_data_dict.get(node)
      else:
        loaded_data = sio.loadmat(self._data_dict.get(node))
        data = np.array(loaded_data['all_data'])
        self._temp_data_dict[node] = data
    return node, children, model, data


  def count_leaf_nodes(self, node, count = 0):

    children = self.get_children(node)
    if children is not None and len(children)>0:
      for child in self.get_children(node):
        count += self.count_leaf_nodes(child)
      return count
    return 1


  def get_exact_results(self, time_series, window, tile_size, out_path, level=None):
    
    print('Running exact search using SCAMP.')
    if level is None:
      leaf_nodes = self.get_level_nodes(0)
    else:
      leaf_nodes = self.get_level_nodes(level)
    result = np.zeros((len(leaf_nodes), len(time_series)-window+1))
    for i, node in enumerate(leaf_nodes):
      node_info = self.get_node_info(node, load_data = True)
      node_seg = node_info[3]
      #TODO
      keyword1 = 'AB_join_node_'+str(i)+'_and_test_data_tile_'+str(tile_size)
      keyword2 = 'BA_join_test_data_and_node_'+str(i)+'_tile_'+str(tile_size)
      mp_path = os.path.join('Info','mp_results',out_path)
      mp_AB = mp_path + '/' + keyword1
      mp_BA = mp_path + '/' + keyword2
      #result[i,:] = np.loadtxt(mp_AB)
      #ba_mp = np.loadtxt(mp_BA)
      result[i,:] = np.loadtxt(mp_BA)
      ba_mp = np.loadtxt(mp_AB)
      #result[i, :], ba_mp = get_mp_with_SCAMP(window, node_seg, time_series) 
    return result

  def predict_mp_vals(self, segment, threshold, leaf_count, model_stride, tree_level=None):
    result = np.zeros((leaf_count, model_stride))

    if tree_level is not None:
      result, pred_t = self.predict_on_a_tree_level(tree_level, segment, threshold, leaf_nb = leaf_count, stride = model_stride, mp_out = True)

    else:

      result, pred_t = self.predict_on_whole_struct(segment, threshold, leaf_nb = leaf_count, stride=model_stride, mp_out = True)
    return result, pred_t


  def predict(self, segment, threshold, tree_level=None):
 
    result = []


    if tree_level is not None:
      result, pred_t = self.predict_on_a_tree_level(tree_level, segment, threshold, leaf_nb=None, stride=None,  mp_out = False)

    else:

      result, pred_t = self.predict_on_whole_struct(segment, threshold, leaf_nb=None, stride=None,  mp_out=False)
    print('subsequence prediction time: ',pred_t)
    return result, pred_t


  def predict_on_a_tree_level(self, level, test_segment, threshold, leaf_nb, stride, mp_out):
    #result = []
    pred_time = 0
    tree_level_nodes = self.get_level_nodes(level)
    result = np.zeros((leaf_nb, stride))

    for idx, node in enumerate(tree_level_nodes):
      if node not in self._temp_model_dict.keys():
        node_info =  self.get_node_info(node)
        model =  node_info[2]
      else:
        model = self._temp_model_dict.get(node)
      node_graph = self._graph_session_dict.get(node)[0]
      node_session = self._graph_session_dict.get(node)[1]
      with node_graph.as_default():
        with node_session.as_default():
          start = time.time()
          predictions = model.predict(test_segment)
          end = time.time()
      pred_time += (end -start)
      if mp_out:
         result[idx,:] = predictions.reshape((stride,))
      else:
        maximum = np.amax(predictions)
        print('>> Max: ',maximum)
        if maximum > threshold:
          result.append(1)
        else:
          result.append(0)

    return result, pred_time
  

  def predict_on_whole_struct(self, test_segment, threshold, leaf_nb, stride, mp_out):

    active_nodes = []
    pred_time = 0
    highest_full_level = 0
    res_idx = 0
    result = np.zeros((leaf_nb, stride))
    pred_track = {}    

    for struct_level in self._struct:
      if struct_level:
        highest_full_struct_level = self._struct.index(struct_level)

    for curr_level in reversed(range(highest_full_struct_level+1)):
      for idx, node in enumerate(self._struct[curr_level]):
        active_nodes.append(node)
        while len(active_nodes) > 0 :
          new_node = active_nodes[0]
          children, model, predictions, pred_time = self.run_pred(new_node, test_segment, pred_time)
          maximum = np.amax(predictions)
          #print('>> New Node:', new_node ,'|| Max: ',maximum)
          if maximum > threshold:
            if len(children) > 0 :
              active_nodes = [children if x == new_node else [x] for x in active_nodes]
              active_nodes = list(itertools.chain.from_iterable(active_nodes))
            else:
              curr_active_node = active_nodes.pop(0)
              if mp_out:
                pred_track.update({curr_active_node : predictions})
              else:
                result.append(1)
              
          else:
            curr_active_node = active_nodes.pop(0)
            if mp_out:
              pred_track.update({curr_active_node : predictions})
            else: 
              leaf_node_nb = self.count_leaf_nodes(new_node)
              for j in range(leaf_node_nb):
                result.append(0)

    for node_idx, node in enumerate(list(pred_track.keys())):
      #print('Node name: ', node)
      children_nodes = self.get_children(node)
      leaf_node_nb = self.count_leaf_nodes(new_node)
      curr_preds = pred_track.get(node)
      if children_nodes is None:  
        result[node_idx,:] =  curr_pred.reshape((stride,))
      else:
        for k in range(leaf_node_nb):
          result[node_idx,:] = curr_preds.reshape((stride,))
          node_idx += 1

    return result, pred_time


  def run_pred(self, node, seg, pred_time):
    if node not in self._temp_model_dict.keys():
      info =  self.get_node_info(node)
      children  = info[1]
      model =  info[2]
    else:
      model = self._temp_model_dict.get(node)
      children  = self.get_children(node)
    node_graph = self._graph_session_dict.get(node)[0]
    node_session = self._graph_session_dict.get(node)[1]
    with node_graph.as_default():
      with node_session.as_default():
        start = time.time()
        predictions = model.predict(seg)
        end = time.time()
    pred_time += (end -start)
    
    return children, model, predictions, pred_time



  def save(self, fname = None):

    if fname == None:
      raise 'File name must be specified'
      return 0

    structure = {}
    structure['struct'] = self._struct
    structure['fname'] = self.struct_fname
    structure['children'] = self._children_dict
    structure['parents'] = self._parents_dict
    structure['models'] = self._model_dict
    structure['datasets'] = self._data_dict
    structure['limit'] = self._lim
    structure['levels'] = self._levels_dict
    structure['retrain_count'] = self._retraining_track
    structure['retrain_node_name'] = self._retraining_node_track

    struct_json_fname = '{}__start_{}'.format(fname,str(datetime.datetime.now()).replace(' ','-'))
    struct_json_file = os.path.join(self._struct_json_file_path, struct_json_fname + '.json')

    print('saving structure ...')
    
    with open(struct_json_file,'w') as outputfile:
      json.dump(structure, outputfile)

    return struct_json_file


  def load(self,fname = None):

    with open(fname,'r') as inputfile:
      structure = json.load(inputfile)
    self._struct = structure.get('struct')
    self.struct_fname = structure.get('fname')
    self._children_dict = structure.get('children')
    self._parents_dict = structure.get('parents')
    self._model_dict = structure.get('models')
    self._data_dict = structure.get('datasets')
    self._lim = structure.get('limit')
    self._levels_dict = structure.get('levels')
    self._retraining_track = structure.get('retrain_count')
    self._retraining_node_track = structure.get('retrain_node_name')

    return self



  def result(self):
    return self._struct

  def get_children(self, node):
    return self._children_dict.get(node, None)


  def get_parents(self, node):
    return self._parents_dict.get(node, None)

  def get_struct_level_constraint(self):
    return self._lim


  def get_level_nodes(self, level):
    return self._levels_dict.get(str(level), None)

  def get_retraining_count_per_node(self):
    return self._retraining_track

  def get_retraining_segments(self, node):
    return self._retraining_node_track.get(node, None)

  def get_retraining_count_per_node(self, node=None):
    if node is None:
      return self._retraining_track
    else:
      return self._retraining_track.get(node, None)


  






def main():

  parser = argparse.ArgumentParser(description = 'Consturcting a tree based LAMP structure') 
  parser.add_argument('-T','-train', required=True, action = 'store_true', help = 'Set training flag to one if training')
  parser.add_argument('-Ts', type = str, required=True, help = 'Training data path')
  parser.add_argument('-w', type = int, required=True, help = 'Window size')
  parser.add_argument('-tile', type = int, required=True, help = 'Tile size')
  parser.add_argument('-Te','-tepochs', type = int, required=True, help = 'Number of epochs per training')
  parser.add_argument('-R','-retrain', action = 'store_true', help = 'Set retraining flag to one if retraining')
  parser.add_argument('-I','-init', action = 'store_true', help = 'Initializing a new Tree')
  parser.add_argument('-Re','-repochs', type = int, help = 'Number of epochs per retraining')
  parser.add_argument('-Rc','-rcount', type = int, help = 'Maximum number of retraining on each segment')
  parser.add_argument('-s', type = str, help = 'Tree structure file -  Needed when not Initializing')
  parser.add_argument('-out', type = str, help = 'Output files path - Required when initializing')
  parser.add_argument('-sout', type = str, help = 'SCAMP Output files path - Required when initializing')
  parser.add_argument('-L','-levels', nargs = '+' , type = int, help = 'Structure level constraints')
  parser.add_argument('-leafNodes', action = 'store_true', help = 'Train all leaf nodes and creat the structure at the end.')
  parser.add_argument('-iSeg', type = int, help = 'Set the initial segment to a specific number')
  parser.add_argument('-fSeg', type = int, help = 'Set the final segment to a specific number')


  args = parser.parse_args()
  init = args.I
  window_size = args.w
  struct_file = args.s
  training  = args.T
  retraining  = args.R
  struct_levels = args.L
  train_data = args.Ts
  train_nb_epochs = args.Te
  max_nb_retraining = args.Rc
  retrain_nb_epochs = args.Re
  all_leaf = args.leafNodes
  init_seg = args.iSeg
  final_seg = args.fSeg
  global tile_size
  global out_path
  global scamp_out_path
  scamp_out_path = args.sout
  out_path = args.out
  tile_size = args.tile


  if init:
    if struct_levels is None:
      print('>> Structure levels are not set. Use -L [L1 ... Lk].')
      return
    if struct_file is not None:
      print('>> Structure file should not be assigned when initializing a new structure.')
      return
    if out_path is None:
      print('>> A folder name for outputs is required. Use -out to set.')
      return

  else:
    if struct_file is None:
      print('>> If initializing a tree set -I and -L [L1 ... Lk].')  
      print('>> If continue training a trained structure set -s [Structure json file].') 
      return
    if struct_levels is not None:
      print('>> New structure level constraints cannot be assigned to a trained structure.')
      return 

  if retraining:
    if retrain_nb_epochs is None:
      print('>> Training number of epochs is not set. Use -Re [RE].')
      return
    if max_nb_retraining is None:
      print('>> Maximum number of retraining per segment is not set. Use -Rc [RC].')
      return


 
  if training:
    
    if init:
      print('limits: ',struct_levels)
      Tree = TreeStruct(None, struct_levels, out_path, all_leaf)
    else:
      Tree = TreeStruct(struct_file, None, out_path, all_leaf)

    curr_struct_file = None

    print('Creating the training segments')
    train_segs = Get_Segments(train_data, tile_size)
    print('Training segments created.')
    curr_struct_file = Tree.Train(train_segs, window_size, train_nb_epochs, retraining, max_nb_retraining, retrain_nb_epochs, init_seg, final_seg)

    print('Final Struct file is saved in - TreeStruct-Info - at  ', datetime.datetime.now())





if __name__=="__main__":
    main()
  
