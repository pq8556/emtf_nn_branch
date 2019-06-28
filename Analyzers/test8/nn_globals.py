from nn_logging import getLogger
logger = getLogger()

# ______________________________________________________________________________
# Globals

mask_value = 100.

reg_pt_scale = 100.
reg_dxy_scale = 0.4

discr_pt_cut_low = 4.
discr_pt_cut_med = 8.
discr_pt_cut_high = 14.

discr_loss_weight = 20.

l1_reg = 0.0
l2_reg = 0.0

infile_muon = '../test7/histos_tba.27.npz'
infile_pileup = '../test7/histos_tbd.27.npz'
infile_highpt = '../test7/histos_tbe.27.npz'
infile_displ1 = '../test7/histos_tba_displ1.27.npz'
infile_displ2 = '../test7/histos_tba_displ2.27.npz'

infile_muon_run3 = '../test7/histos_tba_run3.27.npz'
infile_pileup_run3 = '../test7/histos_tbd_run3.27.npz'
infile_highpt_run3 = '../test7/histos_tbe_run3.27.npz'

infile_muon_omtf = '../test7/histos_tba_omtf.27.npz'
infile_pileup_omtf = '../test7/histos_tbd_omtf.27.npz'
infile_highpt_omtf = '../test7/histos_tbe_omtf.27.npz'


# ______________________________________________________________________________
# Import all the libs
import os
import sys
os.environ['KERAS_BACKEND'] = 'tensorflow'
OLD_STDOUT = sys.stdout

logger.info('Using cmssw {0}'.format(os.environ['CMSSW_VERSION'] if 'CMSSW_VERSION' in os.environ else 'n/a'))

import numpy as np
np.random.seed(2023)
logger.info('Using numpy {0}'.format(np.__version__))

import tensorflow as tf
logger.info('Using tensorflow {0}'.format(tf.__version__))

import keras
from keras import backend as K
#K.set_epsilon(1e-08)
#K.set_session(tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4, allow_soft_placement=True)))
logger.info('Using keras {0}'.format(keras.__version__))
logger.info('.. list devices: {0}'.format(K.get_session().list_devices()))

import scipy
logger.info('Using scipy {0}'.format(scipy.__version__))

import sklearn
logger.info('Using sklearn {0}'.format(sklearn.__version__))

import matplotlib as mpl
import matplotlib.pyplot as plt
logger.info('Using matplotlib {0}'.format(mpl.__version__))
#%matplotlib inline

