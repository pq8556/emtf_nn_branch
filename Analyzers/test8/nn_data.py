import numpy as np
from sklearn.model_selection import train_test_split
from itertools import chain

from nn_logging import getLogger
logger = getLogger()


# ______________________________________________________________________________
def muon_data(filename, create_encoder):
  try:
    logger.info('Loading muon data from {0} ...'.format(filename))
    with np.load(filename) as loaded:
      the_variables = loaded['variables']
      the_parameters = loaded['parameters']
    logger.info('Loaded the variables with shape {0}'.format(the_variables.shape))
    logger.info('Loaded the parameters with shape {0}'.format(the_parameters.shape))
  except:
    logger.error('Failed to load data from file: {0}'.format(filename))

  assert(the_variables.shape[0] == the_parameters.shape[0])

  encoder = create_encoder(the_variables, the_parameters)
  x, y, dxy, dz, x_mask, x_road = encoder.get_x(), encoder.get_y(), \
      encoder.get_dxy(), encoder.get_dz(), encoder.get_x_mask(), encoder.get_x_road()
  logger.info('Loaded the encoded variables with shape {0}'.format(x.shape))
  logger.info('Loaded the encoded parameters with shape {0}'.format(y.shape))
  assert(np.isfinite(x).all())
  assert(np.isfinite(y).all())
  return x, y, dxy, dz, x_mask, x_road

def muon_data_split(filename, create_encoder, test_size=0.5, no_warn=True):
  x, y, dxy, dz, x_mask, x_road = muon_data(filename, create_encoder)

  # Split dataset in training and testing
  x_train, x_test, y_train, y_test, dxy_train, dxy_test, dz_train, dz_test, \
      x_mask_train, x_mask_test, x_road_train, x_road_test = \
      train_test_split(x, y, dxy, dz, x_mask, x_road, test_size=test_size)
  logger.info('Loaded # of training and testing events: {0}'.format((x_train.shape[0], x_test.shape[0])))

  # Check for cases where the number of events in the last batch could be too few
  if not no_warn:
    validation_split = 0.1
    batch_size = 128
    #train_num_samples = int(x_train.shape[0] * (1.0-validation_split))
    #if (train_num_samples%batch_size) < 100:
    #  logger.warning('The last batch for training could be too few! ({0}%{1})={2}. Please change test_size.'.format(train_num_samples, batch_size, train_num_samples%batch_size))
    #  logger.warning('Try this formula: int(int({0}*{1})*{2}) % 128'.format(x.shape[0], 1.0-test_size, 1.0-validation_split))
    train_num_samples = int(x_train.shape[0] * 2 * (1.0-validation_split))
    if (train_num_samples%batch_size) < 100:
      logger.warning('The last batch for training after mixing could be too few! ({0}%{1})={2}. Please change test_size.'.format(train_num_samples, batch_size, train_num_samples%batch_size))
      logger.warning('Try this formula: int(int({0}*{1})*2*{2}) % 128'.format(x.shape[0], 1.0-test_size, 1.0-validation_split))
  return x_train, x_test, y_train, y_test, dxy_train, dxy_test, dz_train, dz_test, x_mask_train, x_mask_test, x_road_train, x_road_test


# ______________________________________________________________________________
def pileup_data(filename, create_encoder):
  try:
    logger.info('Loading pileup data from {0} ...'.format(filename))
    with np.load(filename) as loaded:
      the_variables = loaded['variables']
      the_parameters = loaded['parameters']
      aux = loaded['aux']
    logger.info('Loaded the variables with shape {0}'.format(the_variables.shape))
    logger.info('Loaded the auxiliary PU info with shape {0}'.format(aux.shape))
  except:
    logger.error('Failed to load data from file: {0}'.format(filename))

  assert(the_variables.shape[0] == the_parameters.shape[0])
  assert(the_variables.shape[0] == aux.shape[0])
  assert(aux.shape[1] == 4)  # jobid, ievt, highest_part_pt, highest_track_pt

  encoder = create_encoder(the_variables, the_parameters)
  x, y, dxy, dz, x_mask, x_road = encoder.get_x(), encoder.get_y(), \
      encoder.get_dxy(), encoder.get_dz(), encoder.get_x_mask(), encoder.get_x_road()
  logger.info('Loaded the encoded variables with shape {0}'.format(x.shape))
  logger.info('Loaded the encoded parameters with shape {0}'.format(y.shape))
  logger.info('Loaded the encoded auxiliary PU info with shape {0}'.format(aux.shape))
  assert(np.isfinite(x).all())
  assert(np.isfinite(y).all())
  return x, y, dxy, dz, x_mask, x_road, aux

def pileup_data_split(filename, create_encoder, test_job=159):
  x, y, dxy, dz, x_mask, x_road, aux = pileup_data(filename, create_encoder)

  # Split dataset in training and testing
  split = aux[:,0].astype(np.int32) < test_job
  def my_train_test_split(*arrays):
    train, test = split, ~split
    return list(chain.from_iterable((a[train], a[test]) for a in arrays))

  x_train, x_test, y_train, y_test, dxy_train, dxy_test, dz_train, dz_test, \
      x_mask_train, x_mask_test, x_road_train, x_road_test, aux_train, aux_test = \
      my_train_test_split(x, y, dxy, dz, x_mask, x_road, aux)
  logger.info('Loaded # of training and testing events (PU): {0}'.format((x_train.shape[0], x_test.shape[0])))
  return x_train, x_test, y_train, y_test, dxy_train, dxy_test, dz_train, dz_test, x_mask_train, x_mask_test, x_road_train, x_road_test, aux_train, aux_test


# ______________________________________________________________________________
def mix_training_inputs(x_train, y_train, pu_x_train, pu_y_train, tile=10):
  assert(len(pu_y_train) == 2)
  assert(pu_x_train.shape[0] == pu_y_train[0].shape[0])
  assert(pu_x_train.shape[0] == pu_y_train[1].shape[0])
  num_samples = pu_x_train.shape[0]
  index_array = np.arange(num_samples)
  index_array_ext = np.tile(index_array, tile)  # choose tile to make sure pu_x_train_ext has more entries than x_train
  pu_x_train_ext = pu_x_train[index_array_ext]
  pu_y_train_ext = [pu_y_train[0][index_array_ext], pu_y_train[1][index_array_ext]]

  assert(len(y_train) == 2)
  assert(x_train.shape[0] == y_train[0].shape[0])
  assert(x_train.shape[0] == y_train[1].shape[0])
  if not (pu_x_train_ext.shape[0] >= x_train.shape[0]):
    raise Exception('pu_x_train_ext is required to have more entries than x_train. Make sure {0} >= {1}'.format(pu_x_train_ext.shape[0], x_train.shape[0]))
  num_samples = x_train.shape[0]
  index_array = np.arange(num_samples)
  #np.random.shuffle(index_array)

  try:
    from keras.engine.training import _make_batches as make_batches
  except ImportError:
    from keras.engine.training_utils import make_batches

  sample_batch_size = 128
  batches = make_batches(num_samples, sample_batch_size)

  x_train_new = np.zeros((num_samples*2, x_train.shape[1]), dtype=np.float32)
  y_train_new = [np.zeros((num_samples*2,), dtype=np.float32), np.zeros((num_samples*2,), dtype=np.float32)]

  for batch_index, (batch_start, batch_end) in enumerate(batches):
    batch_ids = index_array[batch_start:batch_end]
    x_train_new[batch_start*2:batch_start*2 + (batch_end-batch_start)] = x_train[batch_ids]
    x_train_new[batch_start*2 + (batch_end-batch_start):batch_end*2] = pu_x_train_ext[batch_ids]
    y_train_new[0][batch_start*2:batch_start*2 + (batch_end-batch_start)] = y_train[0][batch_ids]
    y_train_new[0][batch_start*2 + (batch_end-batch_start):batch_end*2] = pu_y_train_ext[0][batch_ids]
    y_train_new[1][batch_start*2:batch_start*2 + (batch_end-batch_start)] = y_train[1][batch_ids]
    y_train_new[1][batch_start*2 + (batch_end-batch_start):batch_end*2] = pu_y_train_ext[1][batch_ids]

  logger.info('Mixed muon data with pileup data. x_train_new has shape {0}, y_train_new has shape {1},{2}'.format(x_train_new.shape, y_train_new[0].shape, y_train_new[1].shape))
  return x_train_new, y_train_new
