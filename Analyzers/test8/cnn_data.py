import numpy as np

#from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from nn_logging import getLogger
logger = getLogger()


# ______________________________________________________________________________
def cnn_data(filename):
  try:
    logger.info('Loading cnn data from {0} ...'.format(filename))
    loaded = np.load(filename)
    the_image_pixels   = loaded['image_pixels']
    the_image_channels = loaded['image_channels']
    the_labels         = loaded['labels']
    the_parameters     = loaded['parameters']  # q/pt, phi, eta, best_sector
    logger.info('Loaded the images with shape {0},{1}'.format(the_image_pixels.shape, the_image_channels.shape))
    logger.info('Loaded the labels with shape {0}'.format(the_labels.shape))
    logger.info('Loaded the parameters with shape {0}'.format(the_parameters.shape))
  except:
    logger.error('Failed to load data from file: {0}'.format(filename))

  assert(the_image_pixels.shape[0] == the_image_channels.shape[0])
  assert(the_image_pixels.shape[0] == the_labels.shape[0])
  assert(the_image_pixels.shape[0] == the_parameters.shape[0])

  return the_image_pixels, the_image_channels, the_labels, the_parameters

def cnn_data_split(filename, test_size=0.5, shuffle=True, nentries=None):
  images_px, images_ch, labels, parameters = cnn_data(filename)

  if nentries is not None:
    images_px, images_ch, labels, parameters = images_px[:nentries], images_ch[:nentries], labels[:nentries], parameters[:nentries]

  data = train_test_split(images_px, images_ch, labels, parameters, test_size=test_size, shuffle=shuffle)
  logger.info('Loaded # of training and testing events: {0}'.format((data[0].shape[0], data[1].shape[0])))

  #(images_px_train, images_px_test, images_ch_train, images_ch_test, labels_train, labels_test, parameters_train, parameters_test) = data
  return data
