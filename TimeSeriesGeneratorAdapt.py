import numpy as np
import sys
import math
from scipy.stats import zscore
from keras.utils import Sequence
from numpy.lib.stride_tricks import as_strided

class MPTimeseriesGenerator(Sequence):

    """Utility class for generating batches of temporal data.
    This class takes in a sequence of data-points gathered at
    equal intervals, along with time series parameters such as
    stride, length of history, etc., to produce batches for
    training/validation.
    # Arguments
        data: Time series data containing consecutive data points (timesteps).
        targets: Matrix profile values corresponding to subsequences in 'data' size should be len(data) - mp_window + 1 
        length: length of subsequences to extract (length input dimension)
        stride: Period between successive output sequences.
            For stride `s`, consecutive output samples would
            be centered around `data[i]`, `data[i+s]`, `data[i+2*s]`, etc.
        start_index: Data points earlier than `start_index` will not be used
            in the output sequences. This is useful to reserve part of the
            data for test or validation.
        end_index: Data points later than `end_index` will not be used
            in the output sequences. This is useful to reserve part of the
            data for test or validation.
        shuffle: Whether to shuffle output samples,
            or instead draw them in chronological order.
            This should be enabled for training CNNs but consider disabling for RNNs
        batch_size: Number of timeseries samples in each batch
            (except maybe the last one).
        mp_window: Length of the window used to calculate the mp values in targets (usually the same as length input param)
        important_upper_threshold: mp threshold above which to weight a training instance with important_weight
        important_lower_threshold: mp threshold below which to weight a training instance with important_weight
        important_weight: weight to give important instances
        num_outputs: number of mp values predicted by one instance. Starts at the current index and indexes forward by num_outputs * internal_stride, lookahead must be >= num_outputs
        lookahead: number of subsequences to draw in the present/future (including the current subsequence)
        lookbehind: number of subsequences to draw from the past
          lookahead + lookbehind is the size of the 'channels' input dimension
        internal stride: stride at which to draw subsequences in the interval [lookbehind, lookahead]
        recurrent: Whether to format inputs as [batch, timesteps, channels] (for RNNs), or [batch, length (timesteps), width (1), channels (lookahead + lookbehind)] (for CNNs) 

        Dimensions sizes are as follows with respect to the input:
          batch: batch_size
          timesteps: length 
          width: 1 (will change for multidimensional timeseries)
          channels: lookahead + lookbehind
    # Returns
        A [Sequence](/utils/#sequence) instance.
    """
    def __init__(self, data, targets, length,
                 stride=1,
                 start_index=0,
                 end_index=None,
                 shuffle=False,
                 batch_size=32,
                 mp_window=None,
                 important_upper_threshold=1,
                 important_lower_threshold=-1,
                 important_weight=1,
                 num_outputs=1,
                 lookahead=1,
                 lookbehind=0,
                 internal_stride=1,
                 recurrent=False,
                 num_input_timeseries=1,
                 autoencoder=False,
                 merge_points=None):
        if (lookahead + lookbehind) % internal_stride != 0:
          raise ValueError("Internal stride must evenly divide number of input subsequences")
        self.lookahead = lookahead
        self.lookbehind = lookbehind
        self.num_outputs = num_outputs
        self.num_steps = (lookahead + lookbehind) // internal_stride
        print('Lookahead' + str(self.lookahead))
        print('Lookbehind' + str(self.lookbehind))
        print('Num outputs' + str(self.num_outputs))
        print('Num steps' + str(self.num_steps))
	
        self.upper_thresh = important_upper_threshold
        self.lower_thresh = important_lower_threshold
        self.high_weight=important_weight
        if len(data.shape) == 1:
          data.shape = (data.shape[0], 1)
        self.data = data
        if len(targets.shape) == 1:
          targets.shape = (targets.shape[0], 1)
        self.targets = targets
        self.sublen = length
        self.mp_window = mp_window
        self.stride = stride
        self.merge_points = merge_points
        if start_index < lookbehind:
          start_index = lookbehind
        self.start_index = start_index
        if end_index is None or end_index + lookahead >= len(targets):
          end_index = targets.size - 1 - lookahead
            
        self.end_index = end_index
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.internal_stride = internal_stride
        self.recurrent=recurrent
        self.n_series=num_input_timeseries
        self.autoencoder=autoencoder
        self.using_weights=False

        if self.start_index > self.end_index:
            raise ValueError('`start_index+length=%i > end_index=%i` '
                             'is disallowed, as no part of the sequence '
                             'would be left to be used as current step.'
                             % (self.start_index, self.end_index))
        if self.lookahead < self.num_outputs:
            raise ValueError('Lookahead must be greater than or equal to the number of timesteps required by the output length')
        # Perform initial work to shuffle data
        self.on_epoch_end()

    def __len__(self):
        # This forces all batches to be the same size and cuts off any additional data that doesn't fit in a full batch
        return math.ceil(len(self.indexes) / self.batch_size)
    def _empty_batch(self, num_rows, consec_steps, num_outputs):
        samples_shape = [num_rows, consec_steps, self.n_series, self.sublen]
        #samples_shape.extend(self.data.shape[1:])
        targets_shape = [num_rows, num_outputs, self.n_series]
        #targets_shape.extend(self.targets.shape[1:])
        return np.empty(samples_shape), np.empty(targets_shape)
    def _get_weights(self, targets):
        weights = np.copy(targets)
        weights[weights > self.upper_thresh] = self.high_weight
        weights[weights < self.lower_thresh] = self.high_weight
        weights[weights != self.high_weight] = 1
        return weights
    # Extract subsequences from time series x of length m with stride striding
    def subsequences(self,x, m, striding, limit):
      n = math.ceil((len(x) - m + 1) / striding)
      if n < limit:
        raise ValueError('Problem encountered in subsequences, n = {}, len(x) = {}, limit = {}'.format(n,x.shape,limit))
      if n > limit:
        n = limit
      s = x.itemsize
      return as_strided(x, shape=(n,m), strides=(s*striding, s), writeable=False)
    # Returns a batch of data given an index
    def __getitem__(self, index):
        if index < 0:
          raise ValueError('index must be positive.')
        if index >= len(self):
          raise ValueError('index out of bounds')
        assert(index >= 0)
        assert(index < len(self))
        #print(index)
        # Grab a batch of data indexes
        rows = self.indexes[index*self.batch_size:min((index+1)*self.batch_size, len(self.indexes))]
        # Allocate space
        samples, targets = self._empty_batch(len(rows), self.num_steps, self.num_outputs)
        # For each time series dimension
        for i in range(self.n_series):
            # For each index (input element)
            for j, row in enumerate(rows):
                # Extract the subsequences
                #Maryam-------------------------
                #if row < len(self.data) - self.lookahead - self.sublen :
                #---------------------------------
                segment = self.data[row - self.lookbehind:row+self.lookahead+self.sublen-1, i]
                
                if (len(segment) != self.lookahead + self.sublen - 1):
                  print('row:',row)
                  print('Target Sum:',(self.lookahead + self.sublen - 1))
                  sys.stdout.flush()
                  print('|segment|:',len(segment))
                  print('lookahead:',self.lookahead)
                  print('lookbehind:',self.lookbehind)
                  print('sublen',self.sublen)
                  print('j:',j)
                  raise ValueError('segment is abnormal')
                 
                samples[j,:,i,:] = self.subsequences(segment,self.sublen,self.internal_stride, self.num_steps)
                # Z-normalize the subsequences
                mu = np.mean(samples[j, :, i, -self.mp_window:], axis=1)
                sigma = np.std(samples[j, :, i, -self.mp_window:], axis=1)
                samples[j, :, i, :] = (samples[j, :, i, :] - mu[:,np.newaxis]) / sigma[:,np.newaxis]
                # Outputs start from current row and proceed into the future
                targets[j, :, i] = self.targets[row:row + self.num_outputs, i]
        if np.isnan(np.sum(samples)):
          raise ValueError('Input contained NaN')
        if np.isnan(np.sum(targets)):
          raise ValueError('Train target contained NaN')
        if self.recurrent:
          samples = np.transpose(samples, axes=[0,1,2,3])
          samples.shape = samples.shape[:-1]
        else:
          samples = np.transpose(samples, axes=[0,3,2,1])
        if self.autoencoder:
          return samples, samples
        if self.using_weights:
          return samples, targets, self._get_weights(targets)
        return samples, targets

    def perturb(self, indexes, stride, end_index):
      perturbations = np.random.randint(0, high=stride, size=len(indexes))
      if indexes[-1] + perturbations[-1] > end_index:
        perturbations[-1] = 0;
      return indexes + perturbations 
        
    def on_epoch_end(self):
      self.indexes = np.arange(self.start_index, self.end_index + 1, self.stride)
      # Make sure we don't select subsequences which span explicitly separate time series
      if self.merge_points is not None:
        mask = np.ones((len(self.indexes),),dtype=bool) 
        # Get start of extraction windows
        starts = self.indexes - self.lookbehind
        # Get end of extraction windows
        # Adding self.stride is techinically unnecessary but it allows perturb to work with merge points easily
        ends = self.indexes + self.lookahead + self.sublen - 1 + self.stride
        for offset in self.merge_points:
          # Make sure that there are no extraction windows which overlap a merge point
          mask[np.logical_and(offset >= starts, offset <= ends)] = False
        self.indexes = self.indexes[mask]
      # Change ordering each epoch
      if self.shuffle:
        self.indexes = self.perturb(self.indexes, self.stride, self.end_index)
        np.random.shuffle(self.indexes)
