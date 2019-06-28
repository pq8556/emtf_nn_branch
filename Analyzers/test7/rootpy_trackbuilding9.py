import numpy as np
np.random.seed(2026)

import os, sys, datetime
from six.moves import range, zip, map, filter

from rootpy.plotting import Hist, Hist2D, Graph, Efficiency
from rootpy.tree import Tree, TreeChain, TreeModel, FloatCol, IntCol, ShortCol
from rootpy.io import root_open
#from rootpy.memory.keepalive import keepalive
from ROOT import gROOT, TH1
gROOT.SetBatch(True)
TH1.AddDirectory(False)

import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)


# ______________________________________________________________________________
# Utilities

# Enums
kDT, kCSC, kRPC, kGEM, kME0 = 0, 1, 2, 3, 4

# Globals
eta_bins = (0.8, 1.24, 1.56, 1.7, 1.8, 1.98, 2.16, 2.4)
eta_bins = eta_bins[::-1]
#pt_bins = (-0.5, -0.365, -0.26, -0.155, -0.07, 0.07, 0.155, 0.26, 0.365, 0.5)
pt_bins = (-0.49376795, -0.38895044, -0.288812, -0.19121648, -0.0810074, 0.0810074, 0.19121648, 0.288812, 0.38895044, 0.49376795)
#pt_bins = (-0.49349323, -0.38373062, -0.28128058, -0.18467896, -0.07760702, 0.07760702, 0.18467896, 0.28128058, 0.38373062, 0.49349323)
nlayers = 16  # 5 (CSC) + 4 (RPC) + 3 (GEM) + 4 (DT)
#superstrip_size = 32  # 'quadstrip' unit (4 * 8)

assert(len(eta_bins) == 7+1)
assert(len(pt_bins) == 9+1)

# Functions
def root_sum_square(x, y):
  return np.sqrt(x*x + y*y)

def delta_phi(lhs, rhs):  # in radians
  rad = lhs - rhs
  while rad <  -np.pi:  rad += np.pi*2
  while rad >= +np.pi:  rad -= np.pi*2
  return rad

def delta_theta(lhs, rhs):  # in radians
  rad = lhs - rhs
  return rad

def range_phi_deg(deg):
  while deg < -180.:
    deg += 360.
  while deg >= +180.:
    deg -= 360.
  return deg

def range_theta_deg(deg):
  deg = np.abs(deg)
  while deg >= 180.:
    deg -= 180.
  if deg >= 180./2:
    deg = 180. - deg
  return deg

def calc_phi_loc_deg_from_glob(glob, sector):
  # glob in deg, sector [1-6]
  glob = range_phi_deg(glob)
  loc = glob - 15. - (60. * (sector-1))
  return loc

def calc_phi_loc_int(glob, sector):
  # glob in deg, sector [1-6]
  loc = calc_phi_loc_deg_from_glob(glob, sector)
  if (loc + 22.) < 0.:
    loc += 360.
  loc = (loc + 22.) * 60.
  phi_int = int(round(loc))
  return phi_int

def calc_phi_loc_deg(bits):
  loc = float(bits)/60. - 22.
  return loc

def calc_phi_glob_deg(loc, sector):
  # loc in deg, sector [1-6]
  glob = loc + 15. + (60. * (sector-1))
  if glob >= 180.:
    glob -= 360.
  return glob

def calc_theta_int(theta, endcap):
  # theta in deg, endcap [-1,+1]
  if endcap == -1:
    theta = 180. - theta
  theta = (theta - 8.5) * 128./(45.0-8.5)
  theta_int = int(round(theta))
  return theta_int

def calc_theta_rad_from_eta(eta):
  # returns theta in [0-pi] rad
  theta = np.arctan2(1.0, np.sinh(eta))
  return theta

def calc_theta_deg_from_eta(eta):
  # returns theta in [0-180] deg
  return np.rad2deg(calc_theta_rad_from_eta(eta))

def calc_theta_deg_from_int(theta_int):
  theta_deg = float(theta_int) * (45.0-8.5)/128. + 8.5;
  return theta_deg

def calc_eta_from_theta_rad(theta_rad):
  eta = -1. * np.log(np.tan(theta_rad/2.))
  return eta

def calc_eta_from_theta_deg(theta_deg, endcap):
  # theta in deg, endcap [-1,+1]
  theta_deg = range_theta_deg(theta_deg)
  theta_rad = np.deg2rad(theta_deg)
  eta = calc_eta_from_theta_rad(theta_rad)
  if endcap == -1:
    eta = -eta
  return eta

def extrapolate_to_emtf(phi, invpt, eta):  # phi in radians
  # 1.204 is the magic constant at eta of 1.9
  eta_sf = np.sinh(1.9) / np.sinh(np.abs(eta))
  return phi - 1.204 * invpt * eta_sf

def find_sector(phi):  # phi in radians
  dphi = delta_phi(phi, np.pi/12)  # sector 1 starts at 15 deg
  dphi = int(np.floor(dphi/(np.pi/3)))  # divide by 60 deg
  if dphi < 0:
    sector = 7 + dphi
  else:
    sector = 1 + dphi
  return sector

def find_endcap(eta):
  endcap = +1 if eta >= 0. else -1
  return endcap

def find_endsec(endcap, sector):
  endsec = (sector - 1) if endcap == 1 else (sector - 1 + 6)
  return endsec

def find_pt_bin(pt):
  ipt = np.digitize((pt,), pt_bins[1:])[0]  # skip lowest edge
  ipt = np.clip(ipt, 0, len(pt_bins)-2)
  return ipt

def find_eta_bin(eta):
  ieta = np.digitize((np.abs(eta),), eta_bins[1:])[0]  # skip lowest edge
  ieta = np.clip(ieta, 0, len(eta_bins)-2)
  return ieta

def find_pattern_x(emtf_phi):
  return (emtf_phi+16)//32  # divide by 'quadstrip' unit (4 * 8)

def find_pattern_x_inverse(x):
  return (x*32)  # multiply by 'quadstrip' unit (4 * 8)

def calculate_d0(invPt, phi, xv, yv, B=3.811):
  _invPt = np.asarray(invPt, dtype=np.float64)   # needs double precision
  _invPt = np.where(np.abs(_invPt) < 1./10000, np.sign(_invPt+1e-15) * 1./10000, _invPt)
  _R = -1.0 / (0.003 * B * _invPt)               # R = -pT/(0.003 q B)  [cm]
  _xc = xv - (_R * np.sin(phi))                  # xc = xv - R sin(phi)
  _yc = yv + (_R * np.cos(phi))                  # yc = yv + R cos(phi)
  _d0 = _R - (np.sign(_R) * np.hypot(_xc, _yc))  # d0 = R - sign(R) * sqrt(xc^2 + yc^2)
  return _d0.astype(np.float32, casting='same_kind')


# ______________________________________________________________________________
# Long functions

# Decide EMTF hit layer number
class EMTFLayer(object):
  def __init__(self):
    lut = np.zeros((5,5,5), dtype=np.int32) - 99  # (type, station, ring) -> layer
    lut[1,1,4] = 0  # ME1/1a
    lut[1,1,1] = 0  # ME1/1b
    lut[1,1,2] = 1  # ME1/2
    lut[1,1,3] = 1  # ME1/3
    lut[1,2,1] = 2  # ME2/1
    lut[1,2,2] = 2  # ME2/2
    lut[1,3,1] = 3  # ME3/1
    lut[1,3,2] = 3  # ME3/2
    lut[1,4,1] = 4  # ME4/1
    lut[1,4,2] = 4  # ME4/2
    lut[2,1,2] = 5  # RE1/2
    lut[2,1,3] = 5  # RE1/3
    lut[2,2,2] = 6  # RE2/2
    lut[2,2,3] = 6  # RE2/3
    lut[2,3,1] = 7  # RE3/1
    lut[2,3,2] = 7  # RE3/2
    lut[2,3,3] = 7  # RE3/3
    lut[2,4,1] = 8  # RE4/1
    lut[2,4,2] = 8  # RE4/2
    lut[2,4,3] = 8  # RE4/3
    lut[3,1,1] = 9  # GE1/1
    lut[3,2,1] = 10 # GE2/1
    lut[4,1,1] = 11 # ME0
    lut[0,1,1] = 12 # MB1
    lut[0,2,1] = 13 # MB2
    lut[0,3,1] = 14 # MB3
    lut[0,4,1] = 15 # MB4
    self.lut = lut

  def __call__(self, hit):
    index = (hit.type, hit.station, hit.ring)
    emtf_layer = self.lut[index]
    return emtf_layer

# Decide EMTF hit zones
class EMTFZone(object):
  def __init__(self):
    lut = np.zeros((5,5,5,7,2), dtype=np.int32) - 99  # (type, station, ring) -> [zone] x [min_theta,max_theta]
    lut[1,1,4][0] = 4,17    # ME1/1a
    lut[1,1,4][1] = 16,26   # ME1/1a
    lut[1,1,4][2] = 24,37   # ME1/1a
    lut[1,1,4][3] = 34,45   # ME1/1a
    lut[1,1,4][4] = 41,53   # ME1/1a
    lut[1,1,1][0] = 4,17    # ME1/1b
    lut[1,1,1][1] = 16,26   # ME1/1b
    lut[1,1,1][2] = 24,37   # ME1/1b
    lut[1,1,1][3] = 34,45   # ME1/1b
    lut[1,1,1][4] = 41,53   # ME1/1b
    lut[1,1,2][4] = 46,54   # ME1/2
    lut[1,1,2][5] = 52,88   # ME1/2
    lut[1,1,2][6] = 78,88   # ME1/2
    lut[1,1,3][6] = 98,125  # ME1/3
    #
    lut[1,2,1][0] = 4,17    # ME2/1
    lut[1,2,1][1] = 16,25   # ME2/1
    lut[1,2,1][2] = 24,36   # ME2/1
    lut[1,2,1][3] = 34,44   # ME2/1
    lut[1,2,1][4] = 41,49   # ME2/1
    lut[1,2,2][5] = 53,90   # ME2/2
    lut[1,2,2][6] = 77,111  # ME2/2
    #
    lut[1,3,1][0] = 4,17    # ME3/1
    lut[1,3,1][1] = 16,25   # ME3/1
    lut[1,3,1][2] = 24,36   # ME3/1
    lut[1,3,1][3] = 34,40   # ME3/1
    lut[1,3,2][4] = 44,54   # ME3/2
    lut[1,3,2][5] = 52,90   # ME3/2
    lut[1,3,2][6] = 76,96   # ME3/2
    #
    lut[1,4,1][0] = 4,17    # ME4/1
    lut[1,4,1][1] = 16,25   # ME4/1
    lut[1,4,1][2] = 24,35   # ME4/1
    lut[1,4,2][3] = 38,44   # ME4/2
    lut[1,4,2][4] = 41,54   # ME4/2
    lut[1,4,2][5] = 52,90   # ME4/2
    #
    lut[2,1,2][4] = 52,56   # RE1/2
    lut[2,1,2][5] = 52,84   # RE1/2
    lut[2,1,2][6] = 80,120  # RE1/2
    lut[2,1,3][6] = 80,120  # RE1/3
    lut[2,2,2][5] = 56,88   # RE2/2
    lut[2,2,2][6] = 76,112  # RE2/2
    lut[2,2,3][6] = 76,112  # RE2/3
    lut[2,3,1][0] = 4,17    # RE3/1
    lut[2,3,1][1] = 16,25   # RE3/1
    lut[2,3,1][2] = 24,36   # RE3/1
    lut[2,3,2][3] = 40,40   # RE3/2
    lut[2,3,2][4] = 40,52   # RE3/2
    lut[2,3,2][5] = 48,84   # RE3/2
    lut[2,3,3][3] = 40,40   # RE3/3
    lut[2,3,3][4] = 40,52   # RE3/3
    lut[2,3,3][5] = 48,84   # RE3/3
    lut[2,3,3][6] = 80,92   # RE3/3
    lut[2,4,1][0] = 4,17    # RE4/1
    lut[2,4,1][1] = 16,25   # RE4/1
    lut[2,4,1][2] = 24,31   # RE4/1
    lut[2,4,2][3] = 36,44   # RE4/2
    lut[2,4,2][4] = 44,52   # RE4/2
    lut[2,4,2][5] = 52,84   # RE4/2
    lut[2,4,3][3] = 36,44   # RE4/3
    lut[2,4,3][4] = 44,52   # RE4/3
    lut[2,4,3][5] = 52,84   # RE4/3
    #
    lut[3,1,1][1] = 16,26   # GE1/1
    lut[3,1,1][2] = 24,37   # GE1/1
    lut[3,1,1][3] = 35,45   # GE1/1
    lut[3,1,1][4] = 40,52   # GE1/1
    lut[3,2,1][0] = 7,19    # GE2/1
    lut[3,2,1][1] = 18,24   # GE2/1
    lut[3,2,1][2] = 23,36   # GE2/1
    lut[3,2,1][3] = 34,46   # GE2/1
    #
    lut[4,1,1][0] = 4,17    # ME0
    lut[4,1,1][1] = 16,23   # ME0
    #
    lut[0,1,1][6] = 92,130  # MB1
    lut[0,2,1][6] = 108,138 # MB2
    lut[0,3,1][6] = 126,144 # MB3
    self.lut = lut

  def __call__(self, hit):
    index = (hit.type, hit.station, hit.ring)
    entry = self.lut[index]
    answer = (entry[:,0] <= hit.emtf_theta) & (hit.emtf_theta <= entry[:,1])
    zones = np.nonzero(answer)
    if isinstance(zones, tuple):
      zones = zones[0]
    return zones

# Decide EMTF hit bend
class EMTFBend(object):
  def __call__(self, hit):
    emtf_bend = np.int32(hit.bend)
    if hit.type == kCSC:
      # Special case for ME1/1a:
      # rescale the bend to the same scale as ME1/1b
      if hit.station == 1 and hit.ring == 4:
        emtf_bend = np.round(emtf_bend.astype(np.float32) * 0.026331/0.014264).astype(np.int32)
        emtf_bend = np.clip(emtf_bend, -32, 31)
      emtf_bend *= hit.endcap
      emtf_bend = np.sign(emtf_bend) * (np.abs(emtf_bend)/2)  # from 1/32-strip unit to 1/16-strip unit
    #elif hit.type == kGEM:
    #  emtf_bend *= hit.endcap
    elif hit.type == kME0:
      emtf_bend = np.clip(emtf_bend, -64, 63)  # currently in 1/2-strip unit
    elif hit.type == kDT:
      if hit.quality >= 4:
        emtf_bend = np.clip(emtf_bend, -512, 511)
      else:
        #emtf_bend = np.int32(0)
        emtf_bend = np.clip(emtf_bend, -512, 511)
    else:  # kRPC, kGEM
      emtf_bend = np.int32(0)
    return emtf_bend

# Decide EMTF hit bend (old version)
class EMTFOldBend(object):
  def __init__(self):
    self.lut = np.array([5, -5, 4, -4, 3, -3, 2, -2, 1, -1, 0], dtype=np.int32)

  def __call__(self, hit):
    if hit.type == kCSC:
      clct = int(hit.pattern)
      bend = self.lut[clct]
      bend *= hit.endcap
    #elif hit.type == kGEM:
    #  bend = np.int32(hit.bend)
    #  bend *= hit.endcap
    elif hit.type == kME0:
      bend = np.int32(hit.bend)
    elif hit.type == kDT:
      bend = np.int32(hit.bend)
    else:  # kRPC, kGEM
      bend = np.int32(0)
    return bend

# Decide EMTF hit phi (integer unit)
class EMTFPhi(object):
  def __call__(self, hit):
    emtf_phi = np.int32(hit.emtf_phi)
    if hit.type == kCSC:
      if hit.station == 1:
        if hit.ring == 1:
          bend_corr_lut = (-2.0832, 2.0497)  # ME1/1b (r,f)
        elif hit.ring == 4:
          bend_corr_lut = (-2.4640, 2.3886)  # ME1/1a (r,f)
        elif hit.ring == 2:
          bend_corr_lut = (-1.3774, 1.2447)  # ME1/2 (r,f)
        else:
          bend_corr_lut = (-0, 0)            # ME1/3 (r,f): no correction
        bend_corr = bend_corr_lut[int(hit.fr)]
        bend_corr *= hit.bend
        bend_corr *= hit.endcap
        emtf_phi += int(round(bend_corr))
      else:
        pass
    else:
      pass
    return emtf_phi

# Decide EMTF hit phi (integer unit) (old version)
class EMTFOldPhi(object):
  def __init__(self):
    self.ph_pattern_corr_lut = np.array([0, 0, 5, 5, 5, 5, 2, 2, 2, 2, 0], dtype=np.int32)
    self.ph_pattern_corr_sign_lut = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0], dtype=np.int32)
    self.ph_init_lut = np.array([
      1324,1926,2527,1332,1932,2532,1415,2015,2615,1332,1932,2532,726,732,815,732,3128,3725,4325,3132,3732,4332,3214,3815,4415,3131,3732,4331,1316,2516,
      3716,1332,1932,2532,3132,3732,4332,116,732,2580,3781,4980,1964,2564,3164,3764,4364,4964,1380,1364,2580,3780,4980,1964,2564,3164,3764,4364,4964,1380,
      1364,1326,1923,2527,1332,1932,2532,1415,2015,2614,1331,1931,2531,725,732,815,731,3123,3724,4326,3132,3732,4332,3214,3814,4415,3131,3731,4331,1316,
      2516,3715,1332,1932,2532,3132,3731,4331,116,732,2580,3780,4980,1964,2564,3164,3763,4363,4964,1380,1364,2580,3780,4979,1964,2564,3164,3763,4363,4963,
      1380,1363,1323,1926,2525,1331,1932,2532,1415,2015,2615,1331,1931,2531,726,732,816,732,3124,3727,4325,3132,3732,4332,3215,3815,4415,3131,3731,4332,
      1316,2516,3716,1332,1932,2532,3131,3731,4332,116,732,2580,3780,4980,1964,2564,3164,3764,4364,4964,1381,1364,2580,3780,4980,1964,2564,3164,3764,4364,
      4964,1380,1364,1324,1929,2531,1332,1932,2532,1416,2015,2615,1331,1932,2531,725,732,815,732,3123,3728,4327,3132,3732,4332,3215,3815,4416,3132,3731,
      4332,1316,2516,3716,1332,1932,2532,3132,3733,4332,116,732,2580,3781,4980,1964,2564,3165,3765,4365,4964,1380,1364,2580,3781,4981,1964,2564,3164,3765,
      4365,4965,1380,1364,1325,1925,2524,1332,1932,2532,1415,2015,2615,1331,1931,2531,727,732,815,731,3124,3726,4325,3132,3732,4332,3215,3815,4415,3132,
      3731,4331,1316,2516,3716,1332,1932,2532,3132,3732,4332,116,732,2580,3780,4980,1964,2564,3164,3764,4364,4964,1380,1364,2580,3780,4980,1964,2564,3164,
      3764,4364,4964,1380,1364,1321,1927,2524,1332,1932,2532,1415,2015,2615,1331,1931,2532,725,732,815,731,3128,3727,4326,3133,3732,4332,3215,3815,4415,
      3131,3731,4332,1316,2516,3716,1332,1932,2532,3132,3732,4332,116,732,2580,3780,4980,1964,2564,3164,3764,4364,4964,1380,1364,2580,3780,4980,1964,2564,
      3164,3764,4364,4964,1380,1364,1979,2578,3178,1972,2572,3172,1890,2489,3090,1973,2573,3173,1380,1372,1289,1375,3779,4380,4978,3772,4372,4972,3689,4289,
      4889,3772,4373,4972,2588,3788,4988,1972,2572,3172,3772,4372,4972,1388,1372,1324,2524,3724,1340,1940,2540,3140,3740,4340,124,740,1324,2524,3724,1340,
      1940,2540,3140,3740,4340,124,740,1979,2578,3179,1972,2572,3172,1889,2489,3089,1973,2573,3173,1378,1372,1289,1372,3778,4380,4982,3772,4372,4972,3689,
      4289,4890,3773,4373,4973,2588,3788,4988,1972,2572,3172,3772,4372,4972,1388,1372,1324,2524,3724,1340,1940,2540,3140,3740,4340,124,740,1324,2524,3724,
      1340,1940,2540,3140,3740,4340,124,740,1977,2580,3179,1972,2572,3172,1889,2489,3089,1975,2572,3173,1382,1372,1289,1372,3779,4379,4979,3772,4372,4972,
      3688,4289,4889,3773,4373,4973,2588,3788,4988,1972,2572,3172,3772,4372,4972,1388,1372,1324,2524,3724,1340,1940,2540,3140,3740,4340,124,740,1324,2524,
      3724,1340,1940,2540,3140,3740,4340,124,740,1979,2577,3180,1972,2572,3172,1889,2489,3089,1973,2573,3173,1379,1372,1289,1373,3780,4378,4979,3772,4372,
      4972,3689,4289,4889,3773,4373,4973,2588,3788,4988,1972,2572,3172,3772,4372,4972,1388,1372,1324,2524,3724,1340,1940,2540,3140,3740,4340,124,740,1324,
      2524,3724,1340,1940,2541,3141,3741,4341,124,740,1978,2580,3179,1972,2572,3172,1889,2489,3089,1973,2573,3173,1380,1372,1290,1373,3780,4378,4981,3772,
      4372,4972,3689,4290,4889,3775,4372,4977,2588,3787,4987,1972,2572,3172,3772,4372,4972,1388,1372,1324,2523,3723,1340,1940,2540,3140,3740,4340,124,740,
      1324,2523,3724,1341,1941,2541,3141,3741,4341,124,741,1979,2581,3178,1973,2573,3173,1890,2490,3090,1973,2575,3173,1382,1373,1290,1377,3779,4380,4981,
      3773,4373,4973,3690,4290,4890,3774,4374,4976,2589,3789,4989,1973,2573,3173,3773,4373,4973,1388,1373,1325,2525,3725,1341,1941,2541,3141,3741,4341,124,
      741,1325,2525,3725,1341,1941,2541,3141,3741,4341,124,742
    ], dtype=np.int32)
    assert(len(self.ph_init_lut) == 2*6*61)

  def __call__(self, hit):
    emtf_phi = np.int32(hit.emtf_phi)
    clct_pattern = np.int32(hit.pattern)
    if hit.type == kCSC:
      # Is this chamber mounted in reverse direction?
      # (i.e., phi vs. strip number is reversed)
      ph_reverse = (hit.endcap == 1 and hit.station >= 3) or (hit.endcap == -1 and hit.station < 3)

      # Is this 10-deg or 20-deg chamber?
      is_10degree = (hit.station == 1) or (hit.station >= 2 and hit.ring == 2)  # ME1 and ME2,3,4/2

      pc_station = -1
      pc_chamber = -1
      if hit.neighbor == 0:
        if hit.station == 1:  # ME1: 0 - 8, 9 - 17
          pc_station = hit.subsector-1
          pc_chamber = hit.cscid-1
        else:                 # ME2,3,4: 18 - 26, 27 - 35, 36 - 44
          pc_station = hit.station
          pc_chamber = hit.cscid-1
      else:
        if hit.station == 1:  # ME1n: 45 - 47
          pc_station = 5
          pc_chamber = (hit.cscid-1)/3
        else:                 # ME2n,3n,4n: 48 - 53
          pc_station = 5
          pc_chamber = (hit.station) * 2 - 1 + (0 if (hit.cscid-1 < 3) else 1)
      assert(pc_station != -1)
      assert(pc_chamber != -1)

      pc_lut_id = pc_chamber
      if pc_station == 0:    # ME1 sub 1: 0 - 11
        pc_lut_id = pc_lut_id + 9 if (hit.ring == 4) else pc_lut_id
      elif pc_station == 1:  # ME1 sub 2: 16 - 27
        pc_lut_id += 16
        pc_lut_id = pc_lut_id + 9 if (hit.ring == 4) else pc_lut_id
      elif pc_station == 2:  # ME2: 28 - 36
        pc_lut_id += 28
      elif pc_station == 3:  # ME3: 39 - 47
        pc_lut_id += 39
      elif pc_station == 4:  # ME4 : 50 - 58
        pc_lut_id += 50
      elif pc_station == 5 and pc_chamber < 3:  # neighbor ME1: 12 - 15
        pc_lut_id = pc_lut_id + 15 if (hit.ring == 4) else pc_lut_id + 12
      elif pc_station == 5 and pc_chamber < 5:  # neighbor ME2: 37 - 38
        pc_lut_id += 28 + 9 - 3
      elif pc_station == 5 and pc_chamber < 7:  # neighbor ME3: 48 - 49
        pc_lut_id += 39 + 9 - 5
      elif pc_station == 5 and pc_chamber < 9:  # neighbor ME4: 59 - 60
        pc_lut_id += 50 + 9 - 7
      assert(pc_lut_id < 61)

      fw_sector = hit.sector-1
      fw_endcap = 0 if (hit.endcap == 1) else 1
      fw_strip = hit.strip  # already starts from 0

      # Apply phi correction from CLCT pattern number
      clct_pat_corr = self.ph_pattern_corr_lut[clct_pattern]
      clct_pat_corr_sign_ = self.ph_pattern_corr_sign_lut[clct_pattern]
      clct_pat_corr_sign = 1 if (clct_pat_corr_sign_ == 0) else -1
      if fw_strip == 0 and clct_pat_corr_sign == -1:
        clct_pat_corr = 0

      if is_10degree:
        eighth_strip = fw_strip << 2  # full precision, uses only 2 bits of pattern correction
        eighth_strip += clct_pat_corr_sign * (clct_pat_corr >> 1)
      else:
        eighth_strip = fw_strip << 3  # multiply by 2, uses all 3 bits of pattern correction
        eighth_strip += clct_pat_corr_sign * (clct_pat_corr >> 0)
      assert(eighth_strip >= 0)

      # Multiplicative factor for eighth_strip
      factor = 1024
      if hit.station == 1 and hit.ring == 4:
        factor = 1707
      elif hit.station == 1 and hit.ring == 1:
        factor = 1301
      elif hit.station == 1 and hit.ring == 3:
        factor = 947

      ph_tmp = (eighth_strip * factor) >> 10
      ph_tmp_sign = 1 if (ph_reverse == 0) else -1

      endsec_pc_lut_id = (fw_endcap * 6 + fw_sector) * 61 + pc_lut_id
      fph = self.ph_init_lut[endsec_pc_lut_id] + ph_tmp_sign * ph_tmp
      assert(0 <= fph and fph < 5000)
      #
      emtf_phi = fph
    else:  # non-CSC
      pass
    return emtf_phi

# Decide EMTF hit theta (integer unit)
class EMTFTheta(object):
  def __call__(self, hit):
    emtf_theta = np.int32(hit.emtf_theta)
    if hit.type == kDT:
      # wire -1 means no theta SL
      # quality 0&1 are RPC digis
      if (hit.wire == -1) or (hit.quality < 2):
        if hit.station == 1:
          emtf_theta = 112
        elif hit.station == 2:
          emtf_theta = 122
        elif hit.station == 3:
          emtf_theta = 131
      else:
        pass
    else:
      pass
    return emtf_theta

# Decide EMTF hit z-position (floating-point)
class EMTFZee(object):
  def __init__(self):
    self.lut = np.array([599.0, 696.8, 827.1, 937.5, 1027, 708.7, 790.9, 968.8, 1060, 566.4, 794.8, 539.3, 0, 0, 0, 0], dtype=np.float32)
    assert(self.lut.size == nlayers)

  def __call__(self, hit):
    return self.lut[hit.emtf_layer]

# Decide EMTF hit quality
class EMTFQuality(object):
  def __call__(self, hit):
    emtf_qual = np.int32(hit.quality)
    if hit.type == kCSC or hit.type == kME0:
      # front chamber -> +1
      # rear chamber  -> -1
      if int(hit.fr) == 1:
        emtf_qual *= +1
      else:
        emtf_qual *= -1
    elif hit.type == kRPC or hit.type == kGEM:
      emtf_qual = np.int32(0)
    else:  # kDT
      pass
    return emtf_qual

# Decide EMTF hit time (integer unit)
class EMTFTime(object):
  def __call__(self, hit):
    #emtf_time = hit.time
    emtf_time = np.int32(hit.bx)
    return emtf_time

# Decide EMTF road quality (by pattern straightness)
class EMTFRoadQuality(object):
  def __init__(self):
    # First 9 patterns for prompt muons  : -1/2 <= q/pT <= +1/2
    # Next 9 patterns for displaced muons: -1/14 <= q/pT <= +1/14, -120 <= d0 <= 120
    # Total is 18 patterns.
    # ipt   0  1  2  3  4  5  6  7  8
    # strg  1  3  5  7  9  7  5  3  1
    # ipt   9 10 11 12 13 14 15 16 17
    # strg  0  2  4  6  8  6  4  2  0
    self.lut = np.array([1,3,5,7,9,7,5,3,1,0,2,4,6,8,6,4,2,0], dtype=np.int32)
    assert(self.lut.size == 18)

  def __call__(self, ipt):
    return self.lut[ipt]

# Decide EMTF road sort code (by hit composition)
class EMTFRoadSortCode(object):
  def __init__(self):
    # 10   9      8      7    6      5    4    3..0
    #      ME1/1  ME1/2  ME2         ME3  ME4  qual
    #                         RE1&2  RE3  RE4
    # ME0         GE1/1       GE2/1
    # MB1  MB2                MB3&4
    self.lut = np.array([9,8,7,5,4,6,6,5,4,8,6,10,10,9,6,6], dtype=np.int32)
    assert(self.lut.size == nlayers)

  def __call__(self, road_quality, road_hits_layers):
    sort_code = np.int32(0)
    for hit_lay in road_hits_layers:
      mlayer = self.lut[hit_lay]
      sort_code |= (1 << mlayer)
    assert((0 <= road_quality) and (road_quality < 16))
    sort_code |= road_quality
    return sort_code

find_emtf_layer = EMTFLayer()
find_emtf_zones = EMTFZone()
find_emtf_bend = EMTFBend()
find_emtf_old_bend = EMTFOldBend()
find_emtf_phi = EMTFPhi()
find_emtf_old_phi = EMTFOldPhi()
find_emtf_theta = EMTFTheta()
find_emtf_zee = EMTFZee()
find_emtf_qual = EMTFQuality()
find_emtf_time = EMTFTime()
find_emtf_road_quality = EMTFRoadQuality()
find_emtf_road_sort_code = EMTFRoadSortCode()

def is_emtf_singlemu(mode):
  return mode in (11,13,14,15)

def is_emtf_doublemu(mode):
  #return mode in (7,10,12) + (11,13,14,15)
  return mode in (9,10,12) + (11,13,14,15)  # replace 2-3-4 with 1-4

def is_emtf_muopen(mode):
  return mode in (3,5,6,9) + (7,10,12) + (11,13,14,15)

def is_emtf_singlehit(mode):
  return bool(mode & (1 << 3))

def is_emtf_singlehit_me2(mode):
  return bool(mode & (1 << 2))

# Decide EMTF legit hit
def is_emtf_legit_hit(hit):
  def check_bx(hit):
    if hit.type == kCSC:
      return hit.bx in (-1,0)
    elif hit.type == kDT:
      return hit.bx in (-1,0)
    else:
      return hit.bx == 0
  def check_phi(hit):
    if hit.type == kME0:
      return hit.emtf_phi > 0
    elif hit.type == kDT:
      return hit.emtf_phi > 0
    else:
      return True
  return check_bx(hit) and check_phi(hit)

def is_emtf_images_hit(hit):
  def check_quality(hit):
    # quality 0&1 are RPC digis
    if hit.type == kDT:
      return hit.quality >= 2
    else:
      return True
  def check_phi(hit):
    if hit.type == kME0:
      return hit.emtf_phi > 0
    elif hit.type == kDT:
      return hit.emtf_phi > 0
    else:
      return True
  return check_quality(hit) and check_phi(hit)

def is_valid_for_run2(hit):
  is_csc = (hit.type == kCSC)
  is_rpc = (hit.type == kRPC)
  is_irpc = (hit.type == kRPC) and ((hit.station == 3 or hit.station == 4) and hit.ring == 1)
  is_omtf = (hit.type == kRPC) and ((hit.station == 1 or hit.station == 2) and hit.ring == 3)
  return (is_csc or (is_rpc and not is_irpc and not is_omtf))

# From https://stackoverflow.com/a/31539746
def weighted_percentile(data, percents, weights=None):
  ''' percents in units of 1%
  weights specifies the frequency (count) of data.
  '''
  if weights is None:
    return np.percentile(data, percents)
  ind=np.argsort(data)
  d=data[ind]
  w=weights[ind]
  p=1.*w.cumsum()/w.sum()*100
  y=np.interp(percents, p, d)
  return y


# ______________________________________________________________________________
# Data Formats

PARTICLE_NVARS = 6

class Particle(object):
  def __init__(self, pt, eta, phi, q, vx, vy, vz):
    self.pt = pt
    self.eta = eta
    self.phi = phi
    self.q = q
    self.vx = vx
    self.vy = vy
    self.vz = vz

  def to_parameters(self):
    # Convert into an entry in a numpy array
    # (q/pT, phi, eta, vx, vy, vz)
    parameters = np.array((np.true_divide(self.q, self.pt) if self.pt > 0. else 0.,
                           self.phi, self.eta, self.vx, self.vy, self.vz), dtype=np.float32)
    return parameters

class PatternBank(object):
  def __init__(self, bankfile):
    with np.load(bankfile) as data:
      patterns_phi = data['patterns_phi']
      #patterns_theta = data['patterns_theta']
    self.x_array = patterns_phi
    #self.y_array = patterns_theta
    assert(self.x_array.dtype == np.int32)
    #assert(self.y_array.dtype == np.int32)
    #assert(self.x_array.shape == (len(pt_bins)-1, len(eta_bins)-1, nlayers, 3))
    assert(self.x_array.shape == ((len(pt_bins)-1)*2, len(eta_bins)-1, nlayers, 3))  # using 18 patterns
    #assert(self.y_array.shape == (len(pt_bins)-1, len(eta_bins)-1, nlayers, 3))

class Hit(object):
  def __init__(self, _id, emtf_layer, emtf_phi, emtf_theta, emtf_bend,
               emtf_qual, emtf_time, old_emtf_phi, old_emtf_bend,
               sim_tp):
    self.id = _id  # (_type, station, ring, endsec, fr, bx)
    self.emtf_layer = emtf_layer
    self.emtf_phi = emtf_phi
    self.emtf_theta = emtf_theta
    self.emtf_bend = emtf_bend
    self.emtf_qual = emtf_qual
    self.emtf_time = emtf_time
    self.old_emtf_phi = old_emtf_phi
    self.old_emtf_bend = old_emtf_bend
    self.sim_tp = sim_tp

  @property
  def _type(self):
    return self.id[0]

  @property
  def station(self):
    return self.id[1]

  @property
  def ring(self):
    return self.id[2]

  @property
  def endsec(self):
    return self.id[3]

  @property
  def fr(self):
    return self.id[4]

  @property
  def bx(self):
    return self.id[5]

ROAD_LAYER_NVARS = 9  # each layer in the road carries 9 variables
ROAD_LAYER_NVARS_P1 = ROAD_LAYER_NVARS + 1  # plus layer mask
ROAD_INFO_NVARS = 4

class Road(object):
  def __init__(self, _id, hits, mode, quality, sort_code, phi_median, theta_median):
    self.id = _id  # (endcap, sector, ipt, ieta, iphi)
    self.hits = hits
    self.mode = mode
    self.quality = quality
    self.sort_code = sort_code
    self.phi_median = phi_median
    self.theta_median = theta_median

  @property
  def zone(self):
    return self.id[3]

  def to_variables(self):
    # Convert into an entry in a numpy array
    # At the moment, each entry carries (nlayers * (9+1)) + 4 values
    amap = {}
    (endcap, sector, ipt, ieta, iphi) = self.id
    #road_info = (ipt, ieta, iphi)
    road_info = (ipt, ieta, self.phi_median, self.theta_median)
    #np.random.shuffle(self.hits)  # randomize the order
    for hit in self.hits:
      hit_lay = hit.emtf_layer
      if hit_lay not in amap:
        amap[hit_lay] = hit
    arr = np.zeros((ROAD_LAYER_NVARS_P1 * nlayers) + ROAD_INFO_NVARS, dtype=np.float32)
    arr[0*nlayers:ROAD_LAYER_NVARS*nlayers] = np.nan                # variables (n=nlayers * 9)
    arr[ROAD_LAYER_NVARS*nlayers:ROAD_LAYER_NVARS_P1*nlayers] = 1.0 # mask      (n=nlayers * 1)
    arr[ROAD_LAYER_NVARS_P1*nlayers:] = road_info                   # road info (n=4)
    for lay, hit in amap.iteritems():
      ind = [i*nlayers + lay for i in xrange(ROAD_LAYER_NVARS)]
      arr[ind] = (hit.emtf_phi, hit.emtf_theta, hit.emtf_bend, hit.emtf_qual, hit.emtf_time,
                  hit.ring, hit.fr, hit.old_emtf_phi, hit.old_emtf_bend)
      ind = (ROAD_LAYER_NVARS*nlayers + lay)
      arr[ind] = 0.0  # unmask
    return arr

class Track(object):
  def __init__(self, _id, hits, mode, quality, zone, xml_pt, pt, q, y_pred, y_discr, emtf_phi, emtf_theta):
    self.id = _id  # (endcap, sector, ipt, ieta, iphi)
    self.hits = hits
    self.mode = mode
    self.quality = quality
    self.zone = zone
    self.xml_pt = xml_pt
    self.pt = pt
    self.q = q
    self.y_pred = y_pred
    self.y_discr = y_discr
    self.emtf_phi = emtf_phi
    self.emtf_theta = emtf_theta
    self.phi = calc_phi_glob_deg(calc_phi_loc_deg(emtf_phi), _id[1])
    self.eta = calc_eta_from_theta_deg(calc_theta_deg_from_int(emtf_theta), _id[0])

# Save particle list as a numpy array
def particles_to_parameters(particles):
  parameters = np.zeros((len(particles), PARTICLE_NVARS), dtype=np.float32)
  for i, part in enumerate(particles):
    parameters[i] = part.to_parameters()
  return parameters

# Save road list as a numpy array
def roads_to_variables(roads):
  variables = np.zeros((len(roads), (ROAD_LAYER_NVARS_P1 * nlayers) + ROAD_INFO_NVARS), dtype=np.float32)
  for i, road in enumerate(roads):
    variables[i] = road.to_variables()
  return variables

class RaggedTensorValue(object):
  # Based on
  #   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/ragged/ragged_tensor_value.py
  def __init__(self, values, row_splits):
    if not (isinstance(row_splits, (np.ndarray, np.generic)) and
            row_splits.dtype == np.int64 and row_splits.ndim == 1):
      raise TypeError("row_splits must be a 1D int64 numpy array")
    if not isinstance(values, (np.ndarray, np.generic, RaggedTensorValue)):
      raise TypeError("values must be a numpy array or a RaggedTensorValue")
    self._values = values
    self._row_splits = row_splits

  row_splits = property(
      lambda self: self._row_splits,
      doc="""The split indices for the ragged tensor value.""")
  values = property(
      lambda self: self._values,
      doc="""The concatenated values for all rows in this tensor.""")
  dtype = property(
      lambda self: self._values.dtype,
      doc="""The numpy dtype of values in this tensor.""")

  @property
  def shape(self):
    """A tuple indicating the shape of this RaggedTensorValue."""
    return (self._row_splits.shape[0] - 1,) + (None,) + self._values.shape[1:]

  def to_list(self):
    """Returns this ragged tensor value as a nested Python list."""
    if isinstance(self._values, RaggedTensorValue):
      values_as_list = self._values.to_list()
    else:
      values_as_list = self._values.tolist()
    return [
        values_as_list[self._row_splits[i]:self._row_splits[i + 1]]
        for i in range(self._row_splits.shape[0] - 1)
    ]

  def to_array(self):
    """Returns this ragged tensor value as a nested Numpy array."""
    arr = np.empty((self._row_splits.shape[0] - 1,), dtype=np.object)
    for i in range(self._row_splits.shape[0] - 1):
      arr[i] = self._values[self._row_splits[i]:self._row_splits[i + 1]]
    return arr

def create_ragged_array(pylist):
  # Based on
  #   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/ragged/ragged_factory_ops.py
  ragged_rank = 1

  # Build the splits for each ragged rank, and concatenate the inner values
  # into a single list.
  nested_splits = []
  values = pylist
  for dim in range(ragged_rank):
    nested_splits.append([0])
    concatenated_values = []
    for row in values:
      nested_splits[dim].append(nested_splits[dim][-1] + len(row))
      concatenated_values.extend(row)
    values = concatenated_values

  values = np.asarray(values)
  for row_splits in reversed(nested_splits):
    row_splits = np.asarray(row_splits, dtype=np.int64)
    values = RaggedTensorValue(values, row_splits)
  return values


# ______________________________________________________________________________
# Modules

PATTERN_X_CENTRAL = 31  # pattern bin number 31 is the central
PATTERN_X_SEARCH_MIN = 33
#PATTERN_X_SEARCH_MAX = 154-10
PATTERN_X_SEARCH_MAX = 154-10+12  # account for DT

# Pattern recognition module
class PatternRecognition(object):
  def __init__(self, bank, omtf_input=False, run2_input=False):
    self.bank = bank
    self.cache = dict()  # cache for pattern results
    self.omtf_input = omtf_input
    self.run2_input = run2_input

  def _create_road_hit(self, hit):
    hit_id = (hit.type, hit.station, hit.ring, hit.endsec, hit.fr, hit.bx)
    emtf_bend = find_emtf_bend(hit)
    emtf_qual = find_emtf_qual(hit)
    emtf_time = find_emtf_time(hit)
    sim_tp = hit.sim_tp1
    myhit = Hit(hit_id, hit.lay, hit.emtf_phi, hit.emtf_theta, emtf_bend,
                emtf_qual, emtf_time, hit.old_emtf_phi, hit.old_emtf_bend,
                sim_tp)
    return myhit

  def _create_road(self, road_id, road_hits):
    # Find road modes
    road_mode = 0
    road_mode_csc = 0
    road_mode_me0 = 0  # zones 0,1
    road_mode_me12 = 0 # zone 4
    road_mode_csc_me12 = 0 # zone 4
    road_mode_mb1 = 0  # zone 6
    road_mode_mb2 = 0  # zone 6
    road_mode_me13 = 0 # zone 6
    #road_mode_me22 = 0 # zone 6

    for hit in road_hits:
      (_type, station, ring, endsec, fr, bx) = hit.id
      road_mode |= (1 << (4 - station))

      if _type == kCSC or _type == kME0:
        road_mode_csc |= (1 << (4 - station))

      if _type == kME0 and bx == 0:
        road_mode_me0 |= (1 << 1)
      elif _type == kCSC and station == 1 and (ring == 1 or ring == 4) and bx == 0:
        road_mode_me0 |= (1 << 0)

      if _type == kCSC and station == 1 and (ring == 2 or ring == 3):  # pretend as station 2
        road_mode_me12 |= (1 << (4 - 2))
      elif _type == kRPC and station == 1 and (ring == 2 or ring == 3):  # pretend as station 2
        road_mode_me12 |= (1 << (4 - 2))
      else:
        road_mode_me12 |= (1 << (4 - station))

      if _type == kCSC and station == 1 and (ring == 2 or ring == 3):  # pretend as station 2
        road_mode_csc_me12 |= (1 << (4 - 2))
      elif _type == kCSC:
        road_mode_csc_me12 |= (1 << (4 - station))

      if _type == kDT and station == 1:
        road_mode_mb1 |= (1 << 1)
      elif _type == kDT and station >= 2:
        road_mode_mb1 |= (1 << 0)
      elif _type == kCSC and station >= 1 and (ring == 2 or ring == 3):
        road_mode_mb1 |= (1 << 0)
      elif _type == kRPC and station >= 1 and (ring == 2 or ring == 3):
        road_mode_mb1 |= (1 << 0)

      if _type == kDT and station == 2:
        road_mode_mb2 |= (1 << 1)
      elif _type == kDT and station >= 3:
        road_mode_mb2 |= (1 << 0)
      elif _type == kCSC and station >= 1 and (ring == 2 or ring == 3):
        road_mode_mb2 |= (1 << 0)
      elif _type == kRPC and station >= 1 and (ring == 2 or ring == 3):
        road_mode_mb2 |= (1 << 0)

      if _type == kCSC and station == 1 and (ring == 2 or ring == 3):
        road_mode_me13 |= (1 << 1)
      elif _type == kCSC and station >= 2 and (ring == 2 or ring == 3):
        road_mode_me13 |= (1 << 0)
      elif _type == kRPC and station == 1 and (ring == 2 or ring == 3):
        road_mode_me13 |= (1 << 1)
      elif _type == kRPC and station >= 2 and (ring == 2 or ring == 3):
        road_mode_me13 |= (1 << 0)

      #if _type == kCSC and station == 2 and (ring == 2 or ring == 3):
      #  road_mode_me22 |= (1 << 1)
      #elif _type == kCSC and station >= 3 and (ring == 2 or ring == 3):
      #  road_mode_me22 |= (1 << 0)
      #elif _type == kRPC and station == 2 and (ring == 2 or ring == 3):
      #  road_mode_me22 |= (1 << 1)
      #elif _type == kRPC and station >= 3 and (ring == 2 or ring == 3):
      #  road_mode_me22 |= (1 << 0)

    # Create road
    myroad = None
    (endcap, sector, ipt, ieta, iphi) = road_id

    # Apply SingleMu requirement
    # + (zones 0,1) any road with ME0 and ME1
    # + (zone 4) any road with ME1/1, ME1/2 + one more station
    # + (zone 5) any road with 2 stations
    # + (zone 6) any road with MB1+MB2, MB1+MB3, MB1+ME1/3, MB1+ME2/2, MB2+MB3, MB2+ME1/3, MB2+ME2/2, ME1/3+ME2/2
    if ((is_emtf_singlemu(road_mode) and is_emtf_muopen(road_mode_csc)) or \
        (ieta in (0,1) and road_mode_me0 == 3) or \
        (ieta in (4,) and is_emtf_singlemu(road_mode_me12) and is_emtf_muopen(road_mode_csc_me12)) or \
        (ieta in (5,) and is_emtf_doublemu(road_mode) and is_emtf_muopen(road_mode_csc)) or \
        (ieta in (6,) and (road_mode_mb1 == 3 or road_mode_mb2 == 3 or road_mode_me13 == 3)) ):
      road_quality = find_emtf_road_quality(ipt)
      road_sort_code = find_emtf_road_sort_code(road_quality, [hit.emtf_layer for hit in road_hits])
      road_phi_median = 0    # to be determined later
      road_theta_median = 0  # to be determined later
      myroad = Road(road_id, road_hits, road_mode, road_quality, road_sort_code, road_phi_median, road_theta_median)
    return myroad

  def _apply_patterns_in_zone(self, hit_zone, hit_lay):
    result = self.cache.get((hit_zone, hit_lay), None)
    if result is not None:
      return result

    # Retrieve patterns with (ipt, ieta, lay, pattern)
    patterns_x0 = self.bank.x_array[:, hit_zone, hit_lay, 0]
    patterns_x1 = self.bank.x_array[:, hit_zone, hit_lay, 2]
    result = []
    for ipt, (x0, x1) in enumerate(np.column_stack((patterns_x0, patterns_x1))):
      patterns_iphi = np.arange(x0, x1+1, dtype=np.int32)
      patterns_ipt = np.full_like(patterns_iphi, ipt, dtype=np.int32)
      result.append(np.column_stack((patterns_ipt, patterns_iphi)))
    result = np.concatenate(result)
    self.cache[(hit_zone, hit_lay)] = result
    return result

  def _apply_patterns(self, endcap, sector, sector_hits):
    amap = {}  # road_id -> road_hits

    # Loop over hits
    for ihit, hit in enumerate(sector_hits):
      hit_x = find_pattern_x(hit.emtf_phi)
      #hit_y = hit.emtf_theta
      hit_lay = hit.lay
      hit_zones = hit.zones

      myhit = None

      # Loop over the zones that the hit is belong to
      for hit_zone in hit_zones:
        if not self.omtf_input:
          if hit_zone == 6:  # ignore zone 6
            continue

        # Pattern recognition
        result = self._apply_patterns_in_zone(hit_zone, hit_lay)

        # Loop over the results from pattern recognition
        for index in result:
          ipt, iphi = index
          iphi = hit_x - iphi
          ieta = hit_zone

          # Full range is 0 <= iphi <= 154. but a reduced range is sufficient (27% saving on patterns)
          if PATTERN_X_SEARCH_MIN <= iphi <= PATTERN_X_SEARCH_MAX:
            # Create and associate 'myhit' to road ids
            if myhit is None:
              myhit = self._create_road_hit(hit)
            road_id = (endcap, sector, ipt, ieta, iphi)
            amap.setdefault(road_id, []).append(myhit)  # append hit to road

    # Create roads
    roads = []
    for road_id, road_hits in amap.iteritems():
      myroad = self._create_road(road_id, road_hits)
      if myroad is not None:
        roads.append(myroad)
    return roads

  def run(self, hits):
    roads = []

    legit_hits = filter(is_emtf_legit_hit, hits)

    # Split by sector
    sector_mode_array = np.zeros((12,), dtype=np.int32)
    sector_hits_array = np.empty((12,), dtype=np.object)
    for ind in np.ndindex(sector_hits_array.shape):
      sector_hits_array[ind] = []

    # Loop over hits
    for ihit, hit in enumerate(legit_hits):
      hit.endsec = find_endsec(hit.endcap, hit.sector)
      hit.lay = find_emtf_layer(hit)
      assert(hit.lay != -99)

      # Save the old phi & bend values
      hit.old_emtf_phi = find_emtf_old_phi(hit)
      hit.old_emtf_bend = find_emtf_old_bend(hit)

      if hit.type == kCSC:
        sector_mode_array[hit.endsec] |= (1 << (4 - hit.station))
      elif hit.type == kME0:
        sector_mode_array[hit.endsec] |= (1 << (4 - 1))
      elif hit.type == kDT:
        sector_mode_array[hit.endsec] |= (1 << (4 - 1))
      sector_hits_array[hit.endsec].append(hit)

    # Loop over sector processors
    for endcap in (-1, +1):
      for sector in (1, 2, 3, 4, 5, 6):
        endsec = find_endsec(endcap, sector)
        sector_mode = sector_mode_array[endsec]
        sector_hits = sector_hits_array[endsec]

        # Provide early exit if no hit in stations 1&2 (check CSC, ME0, DT)
        if not is_emtf_singlehit(sector_mode) and not is_emtf_singlehit_me2(sector_mode):
          continue

        # Remove all RPC hits
        #sector_hits = [hit for hit in sector_hits if hit.type != kRPC]

        # Remove all non-Run 2 hits
        if self.run2_input:
          sector_hits = list(filter(is_valid_for_run2, sector_hits))

        # Loop over sector hits
        for ihit, hit in enumerate(sector_hits):
          hit.emtf_phi = find_emtf_phi(hit)
          hit.emtf_theta = find_emtf_theta(hit)
          hit.zones = find_emtf_zones(hit)

        # Apply patterns to the sector hits
        sector_roads = self._apply_patterns(endcap, sector, sector_hits)
        sector_roads.sort(key=lambda x: x.id)
        roads += sector_roads
    return roads


# Road cleaning module
# - reject ghost roads and out-of-time roads
# - needs to be replaced by something firmware-friendly
class RoadCleaning(object):
  def __init__(self):
    pass

  def select_bx_zero(self, road):
    bx_counter1 = 0  # count hits with BX <= -1
    bx_counter2 = 0  # count hits with BX == 0
    bx_counter3 = 0  # count hits with BX >= +1
    layer_has_been_used = set()
    for hit in road.hits:
      if hit.emtf_layer not in layer_has_been_used:
        layer_has_been_used.add(hit.emtf_layer)
        if hit.bx <= -1:
          bx_counter1 += 1
        elif hit.bx == 0:
          bx_counter2 += 1
        elif hit.bx >= +1:
          bx_counter3 += 1
    #trk_bx_zero = (bx_counter1 < 2 and bx_counter2 >= 2)
    trk_bx_zero = (bx_counter1 <= 3 and bx_counter2 >= 2 and bx_counter3 <= 2)
    return trk_bx_zero

  def run(self, roads):
    # Skip if no roads
    if len(roads) == 0:
      return []

    # road_id = (endcap, sector, ipt, ieta, iphi)
    amap = {road.id : road for road in roads}

    # sorted list of road_id's
    road_ids = sorted(amap.keys())

    def is_adjacent(prev, curr):
      # adjacent if (x,y,z') == (x,y,z+1)
      return (prev[:-1] == curr[:-1]) and ((prev[-1] + 1) == curr[-1])

    def make_row_splits(lst):
      # assume 'lst' is sorted
      row_splits = []
      row_splits.append(0)
      if len(lst) == 0:
        return row_splits

      prev = lst[0]
      for i in xrange(1,len(lst)):
        curr = lst[i]
        if not is_adjacent(prev, curr):
          row_splits.append(i)
        prev = curr
      row_splits.append(len(lst))
      return row_splits

    def pick_the_median(lst):
      middle = 0 if len(lst) == 0 else (len(lst)-1)//2
      return lst[middle]

    # Make road clusters (groups)
    splits = make_row_splits(road_ids)
    assert(len(splits) >= 2)

    # Loop over the groups, pick the road with best sort code in each group
    tmp_clean_roads = []            # the "best" roads in each group
    tmp_clean_roads_sortcode = []   # keep track of the sort code of each group
    tmp_clean_roads_groupinfo = []  # keep track of the iphi range of each group

    for igroup in xrange(len(splits)-1):
      group = [road_ids[i] for i in xrange(splits[igroup], splits[igroup+1])]

      best_sort_code = -1
      for i in xrange(len(group)):
        road_id = group[i]
        road = amap[road_id]
        if best_sort_code < road.sort_code:
          best_sort_code = road.sort_code

      best_roads = []
      for i in xrange(len(group)):
        road_id = group[i]
        road = amap[road_id]
        if best_sort_code == road.sort_code:
          best_roads.append(road)

      best_road = pick_the_median(best_roads)
      tmp_clean_roads.append(best_road)
      tmp_clean_roads_sortcode.append(best_sort_code)

      _get_iphi = lambda x: x[4]
      iphi_range = (_get_iphi(group[0]), _get_iphi(group[-1]))  # first and last road_id's in the iphi group
      tmp_clean_roads_groupinfo.append(iphi_range)

    if len(tmp_clean_roads) == 0:
      return []

    # Sort by 'sort code'
    ind = np.argsort(tmp_clean_roads_sortcode)[::-1]  # sort reverse
    tmp_clean_roads = [tmp_clean_roads[i] for i in ind]
    tmp_clean_roads_groupinfo = [tmp_clean_roads_groupinfo[i] for i in ind]

    # Loop over the sorted roads, kill the siblings
    clean_roads = []

    for i in xrange(len(tmp_clean_roads)):
      keep = True

      # Check for intersection in the iphi range
      for j in xrange(i):
        road_i = tmp_clean_roads[i]
        road_j = tmp_clean_roads[j]
        group_i = tmp_clean_roads_groupinfo[i]
        group_j = tmp_clean_roads_groupinfo[j]
        # No intersect between two ranges (x1, x2), (y1, y2): (x2 < y1) || (x1 > y2)
        # Intersect: !((x2 < y1) || (x1 > y2)) = (x2 >= y1) and (x1 <= y2)
        # Allow +/-2 due to extrapolation-to-EMTF error
        _get_endsec = lambda x: x[:2]
        if (_get_endsec(road_i.id) == _get_endsec(road_j.id)) and (group_i[1]+2 >= group_j[0]) and (group_i[0]-2 <= group_j[1]):
          keep = False
          break

      # Do not share ME1/1, ME1/2, ME0, MB1, MB2
      if keep:
        road_i = tmp_clean_roads[i]
        hits_i = [(hit.endsec*100 + hit.emtf_layer, hit.emtf_phi) for hit in road_i.hits if hit.emtf_layer in (0,1,11,12,13)]

        for j in xrange(i):
          road_j = tmp_clean_roads[j]
          hits_j = [(hit.endsec*100 + hit.emtf_layer, hit.emtf_phi) for hit in road_j.hits if hit.emtf_layer in (0,1,11,12,13)]
          if set(hits_i).intersection(hits_j):  # has sharing
            keep = False
            break

      # Finally, check consistency with BX=0
      if keep:
        road_i = tmp_clean_roads[i]
        if self.select_bx_zero(road_i):
          clean_roads.append(road_i)
    return clean_roads


# Road slimming module
class RoadSlimming(object):
  def __init__(self, bank):
    self.bank = bank

  def run(self, roads):
    def pick_the_median(lst):
      middle = 0 if len(lst) == 0 else (len(lst)-1)//2
      return lst[middle]

    slim_roads = []

    # Loop over roads
    for road in roads:
      ipt, ieta, iphi = road.id[2:]

      # Retrieve the phi offset terms for each emtf_layer
      patterns_xc = self.bank.x_array[ipt, ieta, :, 1]
      patterns_xc = find_pattern_x_inverse(patterns_xc)

      # Find median phi and theta
      road_phi_median = np.sort(np.array([hit.emtf_phi - patterns_xc[hit.emtf_layer] for hit in road.hits]))
      road_phi_median = pick_the_median(road_phi_median)

      road_theta_median = np.sort(np.array([hit.emtf_theta for hit in road.hits]))
      road_theta_median = pick_the_median(road_theta_median)

      # Loop over all the emtf_layer's, select unique hit for each emtf_layer
      slim_road_hits = []

      for hit_lay in xrange(nlayers):
        sort_criteria = []  # for sorting hits

        for ihit, hit in enumerate(road.hits):
          if hit_lay == hit.emtf_layer:
            dphi = np.abs(hit.emtf_phi - (road_phi_median + patterns_xc[hit.emtf_layer]))
            dtheta = np.abs(hit.emtf_theta - road_theta_median)
            neg_qual = -np.abs(hit.emtf_qual)
            sort_criteria.append((ihit, dphi, dtheta, neg_qual))

        # Find the best hit, which is (max qual, min dtheta, min dphi, min ihit)
        if sort_criteria:
          best_ihit = min(sort_criteria, key=lambda x: (x[3], x[2], x[1], x[0]))[0]
          best_hit = road.hits[best_ihit]
          slim_road_hits.append(best_hit)

      slim_road = Road(road.id, slim_road_hits, road.mode, road.quality, road.sort_code, road_phi_median, road_theta_median)
      slim_roads.append(slim_road)
    return slim_roads


# pT assignment module
class PtAssignment(object):
  def __init__(self, kerasfile, omtf_input=False, run2_input=False):
    (model_file, model_weights_file, model_run3_file, model_run3_weights_file, model_omtf_file, model_omtf_weights_file) = kerasfile
    self.omtf_input = omtf_input
    self.run2_input = run2_input

    self.reg_pt_scale = 100.
    self.reg_dxy_scale = 0.4

    # Get encoders
    from nn_encode import create_encoder
    from nn_encode_run3 import create_encoder as create_encoder_run3
    from nn_encode_omtf import create_encoder as create_encoder_omtf
    from functools import partial
    self.create_encoder = partial(create_encoder, reg_pt_scale=self.reg_pt_scale, reg_dxy_scale=self.reg_dxy_scale)
    self.create_encoder_run3 = partial(create_encoder_run3, reg_pt_scale=self.reg_pt_scale, reg_dxy_scale=self.reg_dxy_scale)
    self.create_encoder_omtf = partial(create_encoder_omtf, reg_pt_scale=self.reg_pt_scale, reg_dxy_scale=self.reg_dxy_scale)

    # Load Keras models
    from nn_models import load_my_model, update_keras_custom_objects
    update_keras_custom_objects()
    self.loaded_model = load_my_model(name=model_file, weights_name=model_weights_file)
    self.loaded_model_run3 = load_my_model(name=model_run3_file, weights_name=model_run3_weights_file)
    self.loaded_model_omtf = load_my_model(name=model_omtf_file, weights_name=model_omtf_weights_file)
    self.loaded_model.trainable = False
    self.loaded_model_run3.trainable = False
    self.loaded_model_omtf.trainable = False
    assert(not self.loaded_model.updates)
    assert(not self.loaded_model_run3.updates)
    assert(not self.loaded_model_omtf.updates)

  def predict(self, x):
    if self.omtf_input:
      encoder = self.create_encoder_omtf(x)
      loaded_model = self.loaded_model_omtf
    elif self.run2_input:
      encoder = self.create_encoder_run3(x)
      loaded_model = self.loaded_model_run3
    else:
      encoder = self.create_encoder(x)
      loaded_model = self.loaded_model

    x_new = encoder.get_x()
    y = loaded_model.predict(x_new)
    z = encoder.get_x_mask()
    t = encoder.get_x_road()

    assert len(y) == 2
    y[0] /= self.reg_pt_scale
    y = np.moveaxis(np.asarray(y),0,-1)  # shape (2, n, 1) -> shape (n, 1, 2)
    return (x_new, y, z, t)

  def run(self, x):
    x_new = np.array([], dtype=np.float32)
    y = np.array([], dtype=np.float32)
    z = np.array([], dtype=np.float32)
    t = np.array([], dtype=np.float32)
    if len(x) == 0:
      return (x_new, y, z, t)

    (x_new, y, z, t) = self.predict(x)
    return (x_new, y, z, t)


# Track producer module
class TrackProducer(object):
  def __init__(self, omtf_input=False, run2_input=False):
    self.omtf_input = omtf_input
    self.run2_input = run2_input

    self.discr_pt_cut_low = 4.
    self.discr_pt_cut_med = 8.
    self.discr_pt_cut_high = 14.

    self.s_min = 0.
    self.s_max = 60.
    self.s_nbins = 120
    self.s_step = (self.s_max - self.s_min)/self.s_nbins
    self.s_lut =[ 2.4605,  2.0075,  1.9042,  2.0762,  2.4325,  2.9043,  3.4101,  3.9232,
                  4.4403,  4.9856,  5.5775,  6.2036,  6.8515,  7.5126,  8.1807,  8.8570,
                  9.5343, 10.2031, 10.8651, 11.5340, 12.2164, 12.9187, 13.6537, 14.4093,
                 15.1559, 15.8731, 16.5513, 17.2402, 17.9719, 18.7379, 19.5292, 20.3469,
                 21.1514, 21.9302, 22.6964, 23.4417, 24.1086, 24.7471, 25.4113, 26.1038,
                 26.7868, 27.4820, 28.2311, 29.0478, 29.9305, 30.8285, 31.6537, 32.3950,
                 33.1279, 33.8928, 34.6529, 35.4154, 36.2441, 37.1817, 38.2494, 39.2588,
                 40.1019, 40.8765, 41.6557, 42.4564, 43.2505, 44.0659, 44.9429, 45.8573,
                 46.7469, 47.6586, 48.6987, 49.6689, 50.3389, 50.9753, 51.7242, 52.4922,
                 53.2630, 54.0344, 54.8061, 55.5778, 56.3496, 57.1213, 57.8931, 58.6648,
                 59.4366, 60.2083, 60.9800, 61.7518, 62.5235, 63.2953, 64.0670, 64.8387,
                 65.6104, 66.3822, 67.1539, 67.9256, 68.6974, 69.4691, 70.2408, 71.0125,
                 71.7843, 72.5560, 73.3277, 74.0995, 74.8712, 75.6429, 76.4146, 77.1864,
                 77.9581, 78.7298, 79.5015, 80.2733, 81.0450, 81.8167, 82.5884, 83.3602,
                 84.1319, 84.9036, 85.6754, 86.4471, 87.2188, 87.9905, 88.7623, 89.5340]
    #self.s_lut = np.linspace(self.s_min, self.s_max, num=self.s_nbins+1)[:-1]
    self.s_step = np.asarray(self.s_step)
    self.s_lut = np.asarray(self.s_lut)

  def get_trigger_pt(self, y_pred):
    xml_pt = np.abs(1.0/y_pred)
    if xml_pt <= 2.:  # do not use the LUT if below 2 GeV
      return xml_pt

    def digitize(x, bins=(self.s_nbins, self.s_min, self.s_max)):
      x = np.clip(x, bins[1], bins[2]-1e-5)
      x = (x - bins[1]) / (bins[2] - bins[1]) * bins[0]
      binx = x.astype(np.int32)
      if binx == bins[0]-1:  # avoid boundary
        binx -= 1
      return binx

    def interpolate(x, x0, x1, y0, y1):
      y = (x - x0) / (x1 - x0) * (y1 - y0) + y0
      return y

    binx = digitize(xml_pt)
    x0, x1 = binx * self.s_step, (binx+1) * self.s_step
    y0, y1 = self.s_lut[binx], self.s_lut[binx+1]
    trg_pt = interpolate(xml_pt, x0, x1, y0, y1)
    assert(trg_pt > 2.)
    return trg_pt

  def pass_trigger(self, ndof, mode, strg, zone, theta_median, y_pred, y_discr):
    ipt1 = strg.astype(np.int32)
    ipt2 = find_pt_bin(y_pred)
    quality1 = find_emtf_road_quality(ipt1)
    quality2 = find_emtf_road_quality(ipt2)
    strg_ok = (quality2 <= (quality1+1))

    xml_pt = np.abs(1.0/y_pred)

    # Apply cuts
    trigger = (y_discr < 0.)  # default: False
    if self.omtf_input:
      if xml_pt > self.discr_pt_cut_high:  # >14 GeV (98.0% coverage)
        trigger = (y_discr > 0.6043)
      elif xml_pt > self.discr_pt_cut_med: # 8-14 GeV (98.0% coverage)
        trigger = (y_discr > 0.2905)
      elif xml_pt > self.discr_pt_cut_low: # 4-8 GeV (98.0% coverage)
        trigger = (y_discr > 0.2000)
      else:
        trigger = (y_discr >= 0.) and strg_ok
    elif self.run2_input:
      if xml_pt > self.discr_pt_cut_high:  # >14 GeV (97.0% coverage)
        trigger = (y_discr > 0.8557)
      elif xml_pt > self.discr_pt_cut_med: # 8-14 GeV (97.0% coverage)
        trigger = (y_discr > 0.6640)
      elif xml_pt > self.discr_pt_cut_low: # 4-8 GeV (97.0% coverage)
        trigger = (y_discr > 0.2000)
      else:
        trigger = (y_discr >= 0.) and strg_ok
    else:
      if xml_pt > self.discr_pt_cut_high:  # >14 GeV (98.5% coverage)
        trigger = (y_discr > 0.9600)
      elif xml_pt > self.discr_pt_cut_med: # 8-14 GeV (98.5% coverage)
        trigger = (y_discr > 0.8932)
      elif xml_pt > self.discr_pt_cut_low: # 4-8 GeV (99.0% coverage)
        trigger = (y_discr > 0.2000)
      else:
        trigger = (y_discr >= 0.) and strg_ok
    return trigger

  def run(self, slim_roads, variables, predictions, x_mask_vars, x_road_vars):

    # __________________________________________________________________________
    # Extra pieces

    def get_ndof_from_x_mask(x_mask):
      assert(x_mask.shape[0] == nlayers)
      assert(x_mask.dtype == np.bool)
      valid = ~x_mask
      return valid.sum()

    def get_mode_from_x_mask(x_mask):
      assert(x_mask.shape[0] == nlayers)
      assert(x_mask.dtype == np.bool)
      valid = ~x_mask
      mode = np.int32(0)
      if np.any((valid[0], valid[1], valid[5], valid[9], valid[11])):   # ME1/1, ME1/2, RE1/2, GE1/1, ME0
        mode |= (1<<3)
      if np.any((valid[2], valid[6], valid[10])):  # ME2, RE2, GE2/1
        mode |= (1<<2)
      if np.any((valid[3], valid[7])):  # ME3, RE3
        mode |= (1<<1)
      if np.any((valid[4], valid[8])):  # ME4, RE4
        mode |= (1<<0)
      return mode

    # __________________________________________________________________________
    assert(len(slim_roads) == len(variables))
    assert(len(slim_roads) == len(predictions))
    assert(len(slim_roads) == len(x_mask_vars))
    assert(len(slim_roads) == len(x_road_vars))

    tracks = []

    for myroad, x, y, x_mask, x_road in zip(slim_roads, variables, predictions, x_mask_vars, x_road_vars):
      assert(len(x.shape) == 1)
      assert(y.shape == (1,2))
      assert(x_mask.shape == (nlayers,))
      assert(x_road.shape == (4,))

      y_pred = np.asscalar(y[...,0])
      y_discr = np.asscalar(y[...,1])
      ndof = get_ndof_from_x_mask(x_mask)
      mode = get_mode_from_x_mask(x_mask)
      strg, zone, phi_median, theta_median = x_road

      passed = self.pass_trigger(ndof, mode, strg, zone, theta_median, y_pred, y_discr)

      if passed:
        xml_pt = np.abs(1.0/y_pred)
        pt = self.get_trigger_pt(y_pred)

        trk_q = np.sign(y_pred).astype(np.int32)
        trk = Track(myroad.id, myroad.hits, mode, myroad.quality, zone, xml_pt, pt, trk_q, y_pred, y_discr, phi_median, theta_median)
        tracks.append(trk)
    return tracks


# Ghost busting module
class GhostBusting(object):
  def __init__(self):
    pass

  def run(self, tracks):
    tracks_after_gb = []

    # Sort by (zone, y_discr)
    # zone is reordered such that zone 6 has the lowest priority.
    tracks.sort(key=lambda trk: ((trk.zone+1) % 7, trk.y_discr), reverse=True)

    def get_gb_endsec(hit):
      tmp_endsec = hit.endsec
      if hit.emtf_phi < (22*60):
        if (hit.endsec == 0) or (hit.endsec == 6):
          tmp_endsec += 5
        elif (1 <= hit.endsec <= 5) or (7 <= hit.endsec <= 11):
          tmp_endsec -= 1
      return tmp_endsec

    def get_gb_emtf_phi(hit):
      tmp_emtf_phi = hit.emtf_phi
      if hit.emtf_phi < (22*60):
        tmp_emtf_phi += (60*60)
      return tmp_emtf_phi

    def get_gb_track_endsec(trk):
      return find_endsec(trk.id[0], trk.id[1])

    # Loop over the sorted tracks and remove duplicates (ghosts)
    for i in xrange(len(tracks)):
      keep = True

      # Do not share ME1/1, ME1/2, ME0, MB1, MB2
      # Need to check for neighbor sector hits
      if keep:
        track_i = tracks[i]
        hits_i = [(get_gb_endsec(hit)*100 + hit.emtf_layer, get_gb_emtf_phi(hit)) for hit in track_i.hits if hit.emtf_layer in (0,1,11,12,13)]

        for j in xrange(i):
          track_j = tracks[j]
          hits_j = [(get_gb_endsec(hit)*100 + hit.emtf_layer, get_gb_emtf_phi(hit)) for hit in track_j.hits if hit.emtf_layer in (0,1,11,12,13)]
          if set(hits_i).intersection(hits_j):  # has sharing
            keep = False
            break

      if keep:
        track_i = tracks[i]

        # Output tracks following the order of sector processors
        ind = np.searchsorted([get_gb_track_endsec(x) for x in tracks_after_gb], get_gb_track_endsec(track_i))
        tracks_after_gb.insert(ind, track_i)
    return tracks_after_gb


# Track-muon correlation module (stolen from Luca Cadamuro)
#     https://github.com/cms-l1t-offline/cmssw/blob/phase2-l1t-integration-CMSSW_10_6_0_pre4/L1Trigger/L1TTrackMatch/interface/L1TkMuCorrDynamicWindows.h
#     https://github.com/cms-l1t-offline/cmssw/blob/phase2-l1t-integration-CMSSW_10_6_0_pre4/L1Trigger/L1TTrackMatch/src/L1TkMuCorrDynamicWindows.cc
class TrackMuonCorrelation(object):
  def __init__(self):
    self.nbins = 10
    self.bounds = np.array([1.2, 1.32, 1.44, 1.56, 1.68, 1.80, 1.92, 2.04, 2.16, 2.28, 2.4])
    assert(self.nbins == len(self.bounds) - 1)
    self.wdws_theta_low = np.array([[ 4.79923815e-05,  1.11477748e+00, -4.36100776e+00],
        [ 4.76547496e-05,  1.30626682e+00, -4.47946909e+00],
        [ 3.88702599e-05,  1.07847842e-02, -2.08741238e+00],
        [ 5.50922903e-05,  1.24764682e-02, -2.52101752e+00],
        [ 4.29130452e-05,  5.31432059e-03, -2.47838073e+00],
        [ 3.73882625e-05,  2.48981434e-02, -3.76262804e+00],
        [ 3.94339036e-05,  2.25711663e-03, -2.09482927e+00],
        [ 3.53218933e-05,  3.81800213e-03, -2.42685936e+00],
        [ 3.32497695e-05,  1.48507367e-03, -1.96501655e+00],
        [ 3.08852091e-05,  1.98481558e-04, -1.52815772e+00]])
    self.wdws_theta_high = np.array([[ 0.01903233,  1.97695079, -2.36484047],
        [ 0.01966801,  0.69875118, -2.00514994],
        [ 0.01892556,  3.66943194, -3.64089329],
        [ 0.0313301 ,  2.16616408, -3.71565459],
        [ 0.01083814,  0.84843492, -2.57721512],
        [ 0.00970201,  0.88381948, -2.7597685 ],
        [ 0.00903627,  0.45407709, -2.3725132 ],
        [ 0.00861844,  0.52564379, -2.78619319],
        [ 0.00859521,  0.34854964, -2.73653931],
        [ 0.00869175,  0.15249032, -2.48836499]])
    self.wdws_phi_low = np.array([[-0.0075419 ,  1.74123017, -0.9323213 ],
        [-0.00683917,  1.53584358, -0.92535413],
        [-0.0068088 ,  1.34743112, -0.92021707],
        [-0.00567192,  1.27380128, -0.96340742],
        [-0.00495206,  1.07889352, -0.95354641],
        [-0.00425003,  0.98413841, -0.97426287],
        [-0.00351819,  0.81611635, -0.96910175],
        [-0.00327811,  0.73326577, -0.98882135],
        [-0.00207991,  0.70757257, -1.05785262],
        [-0.00195751,  0.59613818, -1.05113618]])
    self.wdws_phi_high = np.array([[ 0.00545662,  3.15822435, -1.07035832],
        [ 0.00527892,  2.77517297, -1.06546035],
        [ 0.00627204,  2.53025862, -1.08257935],
        [ 0.01319276,  2.75629254, -1.17128402],
        [ 0.00787062,  2.23080756, -1.11062574],
        [ 0.01025787,  2.26098631, -1.17457624],
        [ 0.00872535,  1.96370858, -1.15916893],
        [ 0.01002838,  1.86365365, -1.19649265],
        [ 0.00980687,  1.79091689, -1.23450375],
        [ 0.01045059,  1.76763996, -1.28661338]])
    self.safety_factor_l = 0.7  # default: 0.5
    self.safety_factor_h = 0.7  # default: 0.5
    self.initial_sf_l = 0.1  # default: 0.0
    self.initial_sf_h = 0.1  # default: 0.0
    self.pt_start = 2.0
    self.pt_end = 8.0  # default: 6.0
    self.do_relax_factor = True

  def run(self, particles, tracks):
    matched = np.zeros((len(particles), len(tracks)), dtype=np.bool)

    def sf_progressive(x, xstart, xstop, ystart, ystop):
      if (x < xstart):
        return ystart
      elif (x >= xstop):
        return ystop
      else:
        return ystart + (x-xstart)*(ystop-ystart)/(xstop-xstart)

    def get_ibin(eta):
      ieta = np.digitize((abs(eta),), self.bounds[1:])[0]  # skip lowest edge
      if ieta > self.nbins-1:
        ieta = self.nbins-1
      return ieta

    def get_theta_bound_low(part_eta, part_pt):
      # [Const]+[A]*TMath::Power(x,[B])
      f = lambda x, a, b, c: a + b * np.power(np.clip(x, 2., 100.), c)
      a, b, c = self.wdws_theta_low[get_ibin(part_eta)]
      return f(part_pt, a, b, c)

    def get_theta_bound_high(part_eta, part_pt):
      # [Const]+[A]*TMath::Power(x,[B])
      f = lambda x, a, b, c: a + b * np.power(np.clip(x, 2., 100.), c)
      a, b, c = self.wdws_theta_high[get_ibin(part_eta)]
      return f(part_pt, a, b, c)

    def get_phi_bound_low(part_eta, part_pt):
      # [Const]+[A]*TMath::Power(x,[B])
      f = lambda x, a, b, c: a + b * np.power(np.clip(x, 2., 100.), c)
      a, b, c = self.wdws_phi_low[get_ibin(part_eta)]
      return f(part_pt, a, b, c)

    def get_phi_bound_high(part_eta, part_pt):
      # [Const]+[A]*TMath::Power(x,[B])
      f = lambda x, a, b, c: a + b * np.power(np.clip(x, 2., 100.), c)
      a, b, c = self.wdws_phi_high[get_ibin(part_eta)]
      return f(part_pt, a, b, c)

    # Loop over tracking particles
    for ipart, part in enumerate(particles):
      part.invpt = np.true_divide(part.q, part.pt)
      part.d0 = calculate_d0(part.invpt, part.phi, part.vx, part.vy)

      # Skip particles that are not (BX=0, |eta|>1, |d0|<10 cm, |z0|<100 cm)
      if not ((part.bx == 0) and (np.abs(part.eta) > 1.) and (np.abs(part.d0) < 10.) and (np.abs(part.vz) < 100.)):
        continue

      # Get boundaries
      if self.do_relax_factor:
        sf_l = sf_progressive(part.pt, self.pt_start, self.pt_end, self.initial_sf_l, self.safety_factor_l)
        sf_h = sf_progressive(part.pt, self.pt_start, self.pt_end, self.initial_sf_h, self.safety_factor_h)
      else:
        sf_l = self.safety_factor_l
        sf_h = self.safety_factor_h

      theta_bound_low = (1 - sf_l) * get_theta_bound_low(part.eta, part.pt)
      theta_bound_high = (1 + sf_h) * get_theta_bound_high(part.eta, part.pt)
      phi_bound_low = (1 - sf_l) * get_phi_bound_low(part.eta, part.pt)
      phi_bound_high = (1 + sf_h) * get_phi_bound_high(part.eta, part.pt)

      # Luca uses 0.004 rad in phi & theta
      if theta_bound_low < 0.004:
        theta_bound_low = -1.  # disable check
      if phi_bound_low < 0.004:
        phi_bound_low = -1.  # disable check
      assert(theta_bound_high > 0.004 and phi_bound_high > 0.004)
      assert(theta_bound_low < theta_bound_high and phi_bound_low < phi_bound_high)

      # Loop over standalone muons
      for itrk, trk in enumerate(tracks):
        emtf_theta_glob = calc_theta_rad_from_eta(trk.eta)  # in radians
        emtf_phi_glob = np.deg2rad(trk.phi)  # in radians

        # Calculate deltas
        dtheta = np.abs(delta_theta(emtf_theta_glob, part.theta))
        tmp_dphi = delta_phi(emtf_phi_glob, part.phi)
        dphi = np.abs(tmp_dphi)
        dphi_sign = -1 if tmp_dphi < 0 else +1
        if part.pt > 100.:
          dphi_sign = -part.q  # disable check

        matched[ipart, itrk] = (
            (theta_bound_low < dtheta <= theta_bound_high) and \
            (phi_bound_low < dphi <= phi_bound_high) and \
            ((dphi_sign * part.q) < 0) and \
            ((trk.eta * part.eta) > 0)
        )

    # Make sure the matching is unique
    for ipart in xrange(matched.shape[0]):
      if matched[ipart, :].any():
        # find the first track match set to True, then set all the following track matches to False
        for itrk in xrange(matched.shape[1]):
          if matched[ipart, itrk]:
            matched[ipart, itrk+1:] = False
            break

    for itrk in xrange(matched.shape[1]):
      if matched[:, itrk].any():
        # now find the highest-pT track match, then set the other track matches to False
        highest_pt = -999999.
        highest_pt_ipart = -999
        for ipart in xrange(matched.shape[0]):
          if matched[ipart, itrk]:
            part = particles[ipart]
            if highest_pt < part.pt:
              highest_pt = part.pt
              highest_pt_ipart = ipart
        assert(highest_pt_ipart >= 0)
        matched[:, itrk] = False
        matched[highest_pt_ipart, itrk] = True
    return matched


# ______________________________________________________________________________
# Analysis: dummy

class DummyAnalysis(object):
  def run(self, omtf_input=False, run2_input=False):
    # Load tree
    if omtf_input:
      tree = load_pgun_omtf()
    else:
      tree = load_pgun()

    # Loop over events
    for ievt, evt in enumerate(tree):
      if maxEvents != -1 and ievt == maxEvents:
        break

      print("Processing event: {0}".format(ievt))

      # Hits
      for ihit, hit in enumerate(evt.hits):
        print(".. hit  {0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11}".format(ihit, hit.bx, hit.type, hit.station, hit.ring, hit.sector, hit.fr, hit.sim_phi, hit.sim_theta, hit.time, hit.sim_tp1, hit.sim_tp2))
      # Tracks
      for itrk, trk in enumerate(evt.tracks):
        print(".. trk  {0} {1} {2} {3} {4} {5} {6} {7}".format(itrk, trk.sector, trk.mode, trk.pt, trk.phi, trk.eta, trk.theta, trk.q))
      # Gen particles
      for ipart, part in enumerate(evt.particles):
        print(".. part {0} {1} {2} {3} {4} {5}".format(ipart, part.pt, part.phi, part.eta, part.theta, part.q))

    # End loop over events
    unload_tree()


# ______________________________________________________________________________
# Analysis: roads

class RoadsAnalysis(object):
  def run(self, omtf_input=False, run2_input=False):
    # Book histograms
    histograms = {}
    eff_pt_bins = (0., 0.5, 1., 1.5, 2., 3., 4., 5., 6., 7., 8., 10., 12., 14., 16., 18., 20., 22., 24., 27., 30., 34., 40., 48., 60., 80., 120.)

    for k in ("denom", "numer"):
      hname = "eff_vs_genpt_%s" % k
      histograms[hname] = Hist(eff_pt_bins, name=hname, title="; gen p_{T} [GeV]", type='F')
      histograms[hname].Sumw2()

      hname = "eff_vs_geneta_%s" % k
      histograms[hname] = Hist(70, 1.1, 2.5, name=hname, title="; gen |#eta|", type='F')
      histograms[hname].Sumw2()

      hname = "eff_vs_genphi_%s" % k
      histograms[hname] = Hist(64, -3.2, 3.2, name=hname, title="; gen #phi", type='F')
      histograms[hname].Sumw2()

    # Load tree
    if omtf_input:
      tree = load_pgun_batch_omtf(jobid)
    else:
      tree = load_pgun_batch(jobid)

    # Workers
    bank = PatternBank(bankfile)
    recog = PatternRecognition(bank, omtf_input=omtf_input, run2_input=run2_input)
    clean = RoadCleaning()
    slim = RoadSlimming(bank)
    out_particles = []
    out_roads = []
    npassed, ntotal = 0, 0

    # Event range
    n = -1

    # __________________________________________________________________________
    # Loop over events
    for ievt, evt in enumerate(tree):
      if n != -1 and ievt == n:
        break

      if len(evt.particles) == 0:
        continue

      part = evt.particles[0]  # particle gun
      part.invpt = np.true_divide(part.q, part.pt)

      roads = recog.run(evt.hits)
      clean_roads = clean.run(roads)
      slim_roads = slim.run(clean_roads)
      assert(len(clean_roads) == len(slim_roads))

      if len(slim_roads) > 0:
        mypart = Particle(part.pt, part.eta, part.phi, part.q, part.vx, part.vy, part.vz)
        out_particles.append(mypart)
        out_roads.append(slim_roads[0])

      if omtf_input:
        is_important = lambda part: (0.8 <= abs(part.eta) <= 1.24) and (part.bx == 0) and (part.pt > 5.)
        is_possible = lambda hits: (any([(hit.type == kDT and hit.station == 1) for hit in hits]) and any([(hit.type == kDT and 2 <= hit.station <= 3) for hit in hits])) or \
            (any([(hit.type == kCSC and hit.station == 1) for hit in hits]) and any([(hit.type == kDT and 1 <= hit.station <= 2) for hit in hits])) or \
            (any([(hit.type == kCSC and hit.station == 1) for hit in hits]) and any([(hit.type == kCSC and 2 <= hit.station <= 3) for hit in hits]))
      else:
        is_important = lambda part: (1.24 <= abs(part.eta) <= 2.4) and (part.bx == 0) and (part.pt > 4.)
        is_possible = lambda hits: any([((hit.type == kCSC or hit.type == kME0) and hit.station == 1) for hit in hits]) and \
            any([(hit.type == kCSC and hit.station >= 2) for hit in hits])

      if ievt < 20 or (len(clean_roads) == 0 and is_important(part) and is_possible(evt.hits)):
        print("evt {0} has {1} roads and {2} clean roads".format(ievt, len(roads), len(clean_roads)))
        print(".. part invpt: {0} pt: {1} phi: {2} eta: {3} theta: {4}".format(part.invpt, part.pt, part.phi, part.eta, part.theta))
        #part.ipt = find_pt_bin(part.invpt)
        #part.ieta = find_eta_bin(part.eta)
        #part.exphi = emtf_extrapolation(part)
        #part.sector = find_sector(part.exphi)
        #part.endcap = find_endcap(part.eta)
        #part.emtf_phi = calc_phi_loc_int(np.rad2deg(part.exphi), part.sector)
        #part.emtf_theta = calc_theta_int(calc_theta_deg_from_eta(part.eta), part.endcap)
        #part_road_id = (part.endcap, part.sector, part.ipt, part.ieta, (part.emtf_phi+16)//32)
        #part_nhits = sum([1 for hit in evt.hits if hit.endcap == part.endcap and hit.sector == part.sector])
        #print(".. part road id: {0} nhits: {1} exphi: {2} emtf_phi: {3}".format(part_road_id, part_nhits, part.exphi, part.emtf_phi))
        for ihit, hit in enumerate(evt.hits):
          hit_id = (hit.type, hit.station, hit.ring, find_endsec(hit.endcap, hit.sector), hit.fr, hit.bx)
          hit_sim_tp = hit.sim_tp1
          print(".. hit {0} id: {1} lay: {2} ph: {3} ({4}) th: {5} bd: {6} ql: {7} tp: {8}".format(ihit, hit_id, find_emtf_layer(hit), hit.emtf_phi, find_pattern_x(hit.emtf_phi), hit.emtf_theta, find_emtf_bend(hit), find_emtf_qual(hit), hit_sim_tp))
        for iroad, myroad in enumerate(roads):
          print(".. road {0} id: {1} nhits: {2} mode: {3} qual: {4} sort: {5}".format(iroad, myroad.id, len(myroad.hits), myroad.mode, myroad.quality, myroad.sort_code))
        for iroad, myroad in enumerate(clean_roads):
          print(".. croad {0} id: {1} nhits: {2} mode: {3} qual: {4} sort: {5}".format(iroad, myroad.id, len(myroad.hits), myroad.mode, myroad.quality, myroad.sort_code))
          for ihit, myhit in enumerate(myroad.hits):
            print(".. .. hit {0} id: {1} lay: {2} ph: {3} th: {4} tp: {5}".format(ihit, myhit.id, myhit.emtf_layer, myhit.emtf_phi, myhit.emtf_theta, myhit.sim_tp))
        for iroad, myroad in enumerate(slim_roads):
          print(".. sroad {0} id: {1} nhits: {2} mode: {3} qual: {4} sort: {5}".format(iroad, myroad.id, len(myroad.hits), myroad.mode, myroad.quality, myroad.sort_code))
          for ihit, myhit in enumerate(myroad.hits):
            print(".. .. hit {0} id: {1} lay: {2} ph: {3} th: {4} tp: {5}".format(ihit, myhit.id, myhit.emtf_layer, myhit.emtf_phi, myhit.emtf_theta, myhit.sim_tp))

      # Quick efficiency
      if is_important(part):
        trigger = len(clean_roads) > 0
        ntotal += 1
        if trigger:
          npassed += 1
        #else:
        #  print("evt {0} FAIL".format(ievt))

        hname = "eff_vs_genpt_denom"
        histograms[hname].fill(part.pt)
        if trigger:
          hname = "eff_vs_genpt_numer"
          histograms[hname].fill(part.pt)

        if part.pt > 20.:
          hname = "eff_vs_geneta_denom"
          histograms[hname].fill(abs(part.eta))
          if trigger:
            hname = "eff_vs_geneta_numer"
            histograms[hname].fill(abs(part.eta))
          hname = "eff_vs_genphi_denom"
          histograms[hname].fill(part.phi)
          if trigger:
            hname = "eff_vs_genphi_numer"
            histograms[hname].fill(part.phi)

    # End loop over events
    unload_tree()

    print('[INFO] npassed/ntotal: %i/%i = %f' % (npassed, ntotal, float(npassed)/ntotal))

    # __________________________________________________________________________
    # Save objects
    outfile = 'histos_tba.npz'
    if use_condor:
      outfile = 'histos_tba_%i.npz' % jobid
    print('[INFO] Creating file: %s' % outfile)
    if True:
      assert(len(out_particles) == len(out_roads))
      parameters = particles_to_parameters(out_particles)
      variables = roads_to_variables(out_roads)
      np.savez_compressed(outfile, parameters=parameters, variables=variables)


# ______________________________________________________________________________
# Analysis: rates

class RatesAnalysis(object):
  def run(self, omtf_input=False, run2_input=False, pileup=200):
    # Book histograms
    histograms = {}
    hname = "nevents"
    histograms[hname] = Hist(5, 0, 5, name=hname, title="; count", type='F')
    for m in ("emtf", "emtf2026"):
      hname = "highest_%s_absEtaMin0.8_absEtaMax2.4_qmin12_pt" % m
      histograms[hname] = Hist(100, 0., 100., name=hname, title="; p_{T} [GeV]; entries", type='F')
      hname = "highest_%s_absEtaMin1.24_absEtaMax2.4_qmin12_pt" % m
      histograms[hname] = Hist(100, 0., 100., name=hname, title="; p_{T} [GeV]; entries", type='F')
      hname = "highest_%s_absEtaMin0.8_absEtaMax1.24_qmin12_pt" % m
      histograms[hname] = Hist(100, 0., 100., name=hname, title="; p_{T} [GeV]; entries", type='F')
      hname = "highest_%s_absEtaMin1.24_absEtaMax1.65_qmin12_pt" % m
      histograms[hname] = Hist(100, 0., 100., name=hname, title="; p_{T} [GeV]; entries", type='F')
      hname = "highest_%s_absEtaMin1.65_absEtaMax2.15_qmin12_pt" % m
      histograms[hname] = Hist(100, 0., 100., name=hname, title="; p_{T} [GeV]; entries", type='F')
      hname = "highest_%s_absEtaMin2.15_absEtaMax2.4_qmin12_pt" % m
      histograms[hname] = Hist(100, 0., 100., name=hname, title="; p_{T} [GeV]; entries", type='F')

      hname = "highest_%s_absEtaMin0.8_absEtaMax2.4_matched_qmin12_pt" % m
      histograms[hname] = Hist(100, 0., 100., name=hname, title="; p_{T} [GeV]; entries", type='F')

      for l in xrange(14,22+1):
        hname = "%s_ptmin%i_qmin12_eta" % (m,l)
        histograms[hname] = Hist(18, 0.75, 2.55, name=hname, title="; |#eta|; entries", type='F')

    # Load tree
    tree = load_minbias_batch(jobid, pileup=pileup)

    # Workers
    bank = PatternBank(bankfile)
    recog = PatternRecognition(bank, omtf_input=omtf_input, run2_input=run2_input)
    clean = RoadCleaning()
    slim = RoadSlimming(bank)
    ptassig1, ptassig2 = PtAssignment(kerasfile, omtf_input=False, run2_input=run2_input), PtAssignment(kerasfile, omtf_input=True, run2_input=run2_input)
    trkprod1, trkprod2 = TrackProducer(omtf_input=False, run2_input=run2_input), TrackProducer(omtf_input=True, run2_input=run2_input)
    ghost = GhostBusting()
    mucorr = TrackMuonCorrelation()

    # Event range
    n = -1

    # __________________________________________________________________________
    # Loop over events
    for ievt, evt in enumerate(tree):
      if n != -1 and ievt == n:
        break

      roads = recog.run(evt.hits)
      clean_roads = clean.run(roads)
      slim_roads = slim.run(clean_roads)

      # EMTF mode
      slim_roads1 = [road for road in slim_roads if road.zone != 6]  # ignore zone 6
      variables1 = roads_to_variables(slim_roads1)
      variables1, predictions1, x_mask_vars1, x_road_vars1 = ptassig1.run(variables1)
      tracks1 = trkprod1.run(slim_roads1, variables1, predictions1, x_mask_vars1, x_road_vars1)

      # OMTF mode
      slim_roads2 = [road for road in slim_roads if road.zone == 6]  # only zone 6
      variables2 = roads_to_variables(slim_roads2)
      variables2, predictions2, x_mask_vars2, x_road_vars2 = ptassig2.run(variables2)
      tracks2 = trkprod2.run(slim_roads2, variables2, predictions2, x_mask_vars2, x_road_vars2)

      # Ghost busting & muon correlator
      emtf2026_tracks = ghost.run(tracks1 + tracks2)
      emtf2026_matched = mucorr.run(evt.particles, emtf2026_tracks)

      found_high_pt_tracks = any(map(lambda trk: trk.pt > 20., emtf2026_tracks))

      if found_high_pt_tracks:
        print("evt {0} has {1} roads, {2} clean roads, {3} old tracks, {4} new tracks".format(ievt, len(roads), len(clean_roads), len(evt.tracks), len(emtf2026_tracks)))
        for ipart, part in enumerate(evt.particles):
          if part.pt > 5.:
            part.invpt = np.true_divide(part.q, part.pt)
            print(".. part invpt: {0} pt: {1} phi: {2} eta: {3} theta: {4}".format(part.invpt, part.pt, part.phi, part.eta, part.theta))
        for iroad, myroad in enumerate(clean_roads):
          print(".. croad {0} id: {1} nhits: {2} mode: {3} qual: {4} sort: {5}".format(iroad, myroad.id, len(myroad.hits), myroad.mode, myroad.quality, myroad.sort_code))
          #for ihit, myhit in enumerate(myroad.hits):
          #  print(".. .. hit {0} id: {1} lay: {2} ph: {3} th: {4} tp: {5}".format(ihit, myhit.id, myhit.emtf_layer, myhit.emtf_phi, myhit.emtf_theta, myhit.sim_tp))
        for itrk, mytrk in enumerate(emtf2026_tracks):
          print(".. trk {0} id: {1} nhits: {2} mode: {3} pt: {4} y_pred: {5} y_discr: {6}".format(itrk, mytrk.id, len(mytrk.hits), mytrk.mode, mytrk.pt, mytrk.y_pred, mytrk.y_discr))
          for ihit, myhit in enumerate(mytrk.hits):
            print(".. .. hit {0} id: {1} lay: {2} ph: {3} th: {4} tp: {5}".format(ihit, myhit.id, myhit.emtf_layer, myhit.emtf_phi, myhit.emtf_theta, myhit.sim_tp))
        for itrk, mytrk in enumerate(evt.tracks):
          nhits = sum([bool(mytrk.mode & (1<<3)), bool(mytrk.mode & (1<<2)), bool(mytrk.mode & (1<<1)), bool(mytrk.mode & (1<<0))])
          print(".. otrk {0} id: {1} nhits: {2} mode: {3} pt: {4}".format(itrk, (mytrk.endcap, mytrk.sector), nhits, mytrk.mode, mytrk.pt))

      # ________________________________________________________________________
      # Fill histograms
      histograms["nevents"].fill(1.0)

      def fill_highest_pt():
        highest_pt = -999999.
        for itrk, trk in enumerate(tracks):
          if select(trk):
            if highest_pt < trk.pt:  # using scaled pT
              highest_pt = trk.pt
        if highest_pt > 0.:
          highest_pt = min(100.-1e-4, highest_pt)
          histograms[hname].fill(highest_pt)

      def fill_eta():
        h = histograms[hname]
        eta_bins = [False] * (h.GetNbinsX()+2)
        for itrk, trk in enumerate(tracks):
          if select(trk):  # using scaled pT
            b = h.FindFixBin(abs(trk.eta))
            eta_bins[b] = True
        for b in xrange(len(eta_bins)):
          if eta_bins[b]:
            h.fill(h.GetBinCenter(b))

      tracks = evt.tracks
      select = lambda trk: trk and (0.8 <= abs(trk.eta) <= 2.4) and (trk.bx == 0) and (trk.mode in (11,13,14,15))
      hname = "highest_emtf_absEtaMin0.8_absEtaMax2.4_qmin12_pt"
      fill_highest_pt()
      select = lambda trk: trk and (1.24 <= abs(trk.eta) <= 2.4) and (trk.bx == 0) and (trk.mode in (11,13,14,15))
      hname = "highest_emtf_absEtaMin1.24_absEtaMax2.4_qmin12_pt"
      fill_highest_pt()
      select = lambda trk: trk and (0.8 <= abs(trk.eta) < 1.24) and (trk.bx == 0) and (trk.mode in (11,13,14,15))
      hname = "highest_emtf_absEtaMin0.8_absEtaMax1.24_qmin12_pt"
      fill_highest_pt()
      select = lambda trk: trk and (1.24 <= abs(trk.eta) < 1.65) and (trk.bx == 0) and (trk.mode in (11,13,14,15))
      hname = "highest_emtf_absEtaMin1.24_absEtaMax1.65_qmin12_pt"
      fill_highest_pt()
      select = lambda trk: trk and (1.65 <= abs(trk.eta) < 2.15) and (trk.bx == 0) and (trk.mode in (11,13,14,15))
      hname = "highest_emtf_absEtaMin1.65_absEtaMax2.15_qmin12_pt"
      fill_highest_pt()
      select = lambda trk: trk and (2.15 <= abs(trk.eta) <= 2.4) and (trk.bx == 0) and (trk.mode in (11,13,14,15))
      hname = "highest_emtf_absEtaMin2.15_absEtaMax2.4_qmin12_pt"
      fill_highest_pt()
      for l in xrange(14,22+1):
        select = lambda trk: trk and (0 <= abs(trk.eta) <= 9.9) and (trk.bx == 0) and (trk.mode in (11,13,14,15)) and (trk.pt > float(l))
        hname = "emtf_ptmin%i_qmin12_eta" % (l)
        fill_eta()

      tracks = emtf2026_tracks
      select = lambda trk: trk and (0.8 <= abs(trk.eta) <= 2.4) and trk.zone in (0,1,2,3,4,5,6)
      hname = "highest_emtf2026_absEtaMin0.8_absEtaMax2.4_qmin12_pt"
      fill_highest_pt()
      select = lambda trk: trk and (1.24 <= abs(trk.eta) <= 2.4) and trk.zone in (0,1,2,3,4,5)
      hname = "highest_emtf2026_absEtaMin1.24_absEtaMax2.4_qmin12_pt"
      fill_highest_pt()
      select = lambda trk: trk and (0.8 <= abs(trk.eta) <= 1.24) and trk.zone in (6,)
      hname = "highest_emtf2026_absEtaMin0.8_absEtaMax1.24_qmin12_pt"
      fill_highest_pt()
      select = lambda trk: trk and (1.24 <= abs(trk.eta) <= 1.65) and trk.zone in (0,1,2,3,4,5)
      hname = "highest_emtf2026_absEtaMin1.24_absEtaMax1.65_qmin12_pt"
      fill_highest_pt()
      select = lambda trk: trk and (1.65 <= abs(trk.eta) <= 2.15) and trk.zone in (0,1,2,3,4,5)
      hname = "highest_emtf2026_absEtaMin1.65_absEtaMax2.15_qmin12_pt"
      fill_highest_pt()
      select = lambda trk: trk and (2.15 <= abs(trk.eta) <= 2.4) and trk.zone in (0,1,2,3,4,5)
      hname = "highest_emtf2026_absEtaMin2.15_absEtaMax2.4_qmin12_pt"
      fill_highest_pt()
      for l in xrange(14,22+1):
        select = lambda trk: trk and (0 <= abs(trk.eta) <= 9.9) and (trk.pt > float(l))
        hname = "emtf2026_ptmin%i_qmin12_eta" % (l)
        fill_eta()

      # For fake rate plot
      for itrk, mytrk in enumerate(emtf2026_tracks):
        m = emtf2026_matched[:, itrk]
        mytrk.matched = m.any()

      tracks = emtf2026_tracks
      select = lambda trk: trk and (0.8 <= abs(trk.eta) <= 2.4) and trk.zone in (0,1,2,3,4,5,6) and trk.matched
      hname = "highest_emtf2026_absEtaMin0.8_absEtaMax2.4_matched_qmin12_pt"
      fill_highest_pt()

    # End loop over events
    unload_tree()

    # __________________________________________________________________________
    # Save histograms
    outfile = 'histos_tbb.root'
    if use_condor:
      outfile = 'histos_tbb_%i.root' % jobid
    print('[INFO] Creating file: %s' % outfile)
    with root_open(outfile, 'recreate') as f:
      hnames = []
      hname = "nevents"
      hnames.append("nevents")
      for m in ("emtf", "emtf2026"):
        hname = "highest_%s_absEtaMin0.8_absEtaMax2.4_qmin12_pt" % m
        hnames.append(hname)
        hname = "highest_%s_absEtaMin1.24_absEtaMax2.4_qmin12_pt" % m
        hnames.append(hname)
        hname = "highest_%s_absEtaMin0.8_absEtaMax1.24_qmin12_pt" % m
        hnames.append(hname)
        hname = "highest_%s_absEtaMin1.24_absEtaMax1.65_qmin12_pt" % m
        hnames.append(hname)
        hname = "highest_%s_absEtaMin1.65_absEtaMax2.15_qmin12_pt" % m
        hnames.append(hname)
        hname = "highest_%s_absEtaMin2.15_absEtaMax2.4_qmin12_pt" % m
        hnames.append(hname)
        hname = "highest_%s_absEtaMin0.8_absEtaMax2.4_matched_qmin12_pt" % m
        hnames.append(hname)
        for l in xrange(14,22+1):
          hname = "%s_ptmin%i_qmin12_eta" % (m,l)
          hnames.append(hname)
      for hname in hnames:
        h = histograms[hname]
        h.Write()


# ______________________________________________________________________________
# Analysis: effie

class EffieAnalysis(object):
  def run(self, omtf_input=False, run2_input=False):
    # Book histograms
    histograms = {}
    eff_pt_bins = (0., 0.5, 1., 1.5, 2., 3., 4., 5., 6., 7., 8., 10., 12., 14., 16., 18., 20., 22., 24., 26., 28., 30., 34., 40., 48., 60., 80., 120.)
    eff_highpt_bins = (2., 2.5, 3., 3.5, 4., 4.5, 5., 6., 7., 8., 10., 12., 14., 16., 18., 20., 22., 24., 26., 28., 30., 34., 40., 48., 60., 80., 100., 120., 250., 500., 1000.)

    for m in ("emtf", "emtf2026"):
      for l in (0, 10, 15, 20, 30, 40, 50):
        for k in ("denom", "numer"):
          hname = "%s_eff_vs_genpt_l1pt%i_%s" % (m,l,k)
          histograms[hname] = Hist(eff_pt_bins, name=hname, title="; gen p_{T} [GeV]", type='F')
          hname = "%s_eff_vs_genpt_allzones_l1pt%i_%s" % (m,l,k)
          histograms[hname] = Hist(eff_pt_bins, name=hname, title="; gen p_{T} [GeV]", type='F')
          hname = "%s_eff_vs_genpt_highpt_l1pt%i_%s" % (m,l,k)
          histograms[hname] = Hist(eff_highpt_bins, name=hname, title="; gen p_{T} [GeV]", type='F')
          hname = "%s_eff_vs_genpt_allzones_highpt_l1pt%i_%s" % (m,l,k)
          histograms[hname] = Hist(eff_highpt_bins, name=hname, title="; gen p_{T} [GeV]", type='F')
          hname = "%s_eff_vs_genphi_l1pt%i_%s" % (m,l,k)
          histograms[hname] = Hist(76, -190, 190, name=hname, title="; gen #phi {gen p_{T} > 20 GeV}", type='F')
          hname = "%s_eff_vs_genphi_allzones_l1pt%i_%s" % (m,l,k)
          histograms[hname] = Hist(76, -190, 190, name=hname, title="; gen #phi {gen p_{T} > 20 GeV}", type='F')
          hname = "%s_eff_vs_geneta_l1pt%i_%s" % (m,l,k)
          histograms[hname] = Hist(85, 0.8, 2.5, name=hname, title="; gen |#eta| {gen p_{T} > 20 GeV}", type='F')
          hname = "%s_eff_vs_geneta_allzones_l1pt%i_%s" % (m,l,k)
          histograms[hname] = Hist(85, 0.8, 2.5, name=hname, title="; gen |#eta| {gen p_{T} > 20 GeV}", type='F')
          hname = "%s_eff_vs_gend0_l1pt%i_%s" % (m,l,k)
          histograms[hname] = Hist(80, 0, 120, name=hname, title="; gen |d_{0}| {gen p_{T} > 20 GeV}", type='F')
          hname = "%s_eff_vs_gend0_allzones_l1pt%i_%s" % (m,l,k)
          histograms[hname] = Hist(80, 0, 120, name=hname, title="; gen |d_{0}| {gen p_{T} > 20 GeV}", type='F')

      hname = "%s_l1pt_vs_genpt" % m
      histograms[hname] = Hist2D(100, -0.5, 0.5, 300, -0.5, 0.5, name=hname, title="; gen q/p_{T} [1/GeV]; q/p_{T} [1/GeV]", type='F')
      hname = "%s_l1ptres_vs_genpt" % m
      histograms[hname] = Hist2D(100, -0.5, 0.5, 300, -1, 2, name=hname, title="; gen q/p_{T} [1/GeV]; #Delta(p_{T})/p_{T}", type='F')

    # Load tree
    if omtf_input:
      tree = load_pgun_batch_omtf(jobid)
    else:
      tree = load_pgun_batch(jobid)

    # Workers
    bank = PatternBank(bankfile)
    recog = PatternRecognition(bank, omtf_input=omtf_input, run2_input=run2_input)
    clean = RoadCleaning()
    slim = RoadSlimming(bank)
    ptassig1, ptassig2 = PtAssignment(kerasfile, omtf_input=False, run2_input=run2_input), PtAssignment(kerasfile, omtf_input=True, run2_input=run2_input)
    trkprod1, trkprod2 = TrackProducer(omtf_input=False, run2_input=run2_input), TrackProducer(omtf_input=True, run2_input=run2_input)
    ghost = GhostBusting()
    mucorr = TrackMuonCorrelation()

    # Event range
    n = -1

    # __________________________________________________________________________
    # Loop over events
    for ievt, evt in enumerate(tree):
      if n != -1 and ievt == n:
        break

      if len(evt.particles) == 0:
        continue

      part = evt.particles[0]  # particle gun
      part.invpt = np.true_divide(part.q, part.pt)
      part.d0 = calculate_d0(part.invpt, part.phi, part.vx, part.vy)

      roads = recog.run(evt.hits)
      clean_roads = clean.run(roads)
      slim_roads = slim.run(clean_roads)

      # EMTF mode
      slim_roads1 = [road for road in slim_roads if road.zone != 6]  # ignore zone 6
      variables1 = roads_to_variables(slim_roads1)
      variables1, predictions1, x_mask_vars1, x_road_vars1 = ptassig1.run(variables1)
      tracks1 = trkprod1.run(slim_roads1, variables1, predictions1, x_mask_vars1, x_road_vars1)

      # OMTF mode
      slim_roads2 = [road for road in slim_roads if road.zone == 6]  # only zone 6
      variables2 = roads_to_variables(slim_roads2)
      variables2, predictions2, x_mask_vars2, x_road_vars2 = ptassig2.run(variables2)
      tracks2 = trkprod2.run(slim_roads2, variables2, predictions2, x_mask_vars2, x_road_vars2)

      # Ghost busting & muon correlator
      emtf2026_tracks = ghost.run(tracks1 + tracks2)
      emtf2026_matched = mucorr.run(evt.particles, emtf2026_tracks)

      if ievt < 20 and False:
        print("evt {0} has {1} roads, {2} clean roads, {3} old tracks, {4} new tracks".format(ievt, len(roads), len(clean_roads), len(evt.tracks), len(emtf2026_tracks)))
        print(".. part invpt: {0} pt: {1} phi: {2} eta: {3} theta: {4}".format(part.invpt, part.pt, part.phi, part.eta, part.theta))
        for itrk, mytrk in enumerate(emtf2026_tracks):
          y_true = np.true_divide(part.q, part.pt)
          y_pred = np.true_divide(mytrk.q, mytrk.xml_pt)
          m = emtf2026_matched[:, itrk]
          print(".. y_true: {0} y_pred: {1} m: {2}".format(y_true, y_pred, m.any()))

      # ________________________________________________________________________
      # Fill histograms
      def fill_efficiency_pt():
        if select_part(part):
          trigger = any([select_track(trk) for trk in tracks])  # using scaled pT
          denom = histograms[hname + "_denom"]
          numer = histograms[hname + "_numer"]
          denom.fill(part.pt)
          if trigger:
            numer.fill(part.pt)
          #
          trigger_allzones = any([select_track_allzones(trk) for trk in tracks])  # using scaled pT
          denom = histograms[hname_allzones + "_denom"]
          numer = histograms[hname_allzones + "_numer"]
          denom.fill(part.pt)
          if trigger_allzones:
            numer.fill(part.pt)

      def fill_efficiency_phi():
        if select_part(part):
          trigger = any([select_track(trk) for trk in tracks])  # using scaled pT
          denom = histograms[hname + "_denom"]
          numer = histograms[hname + "_numer"]
          denom.fill(np.rad2deg(part.phi))
          if trigger:
            numer.fill(np.rad2deg(part.phi))
          #
          trigger_allzones = any([select_track_allzones(trk) for trk in tracks])  # using scaled pT
          denom = histograms[hname_allzones + "_denom"]
          numer = histograms[hname_allzones + "_numer"]
          denom.fill(np.rad2deg(part.phi))
          if trigger_allzones:
            numer.fill(np.rad2deg(part.phi))

      def fill_efficiency_eta():
        if (part.bx == 0):
          trigger = any([select_track(trk) for trk in tracks])  # using scaled pT
          denom = histograms[hname + "_denom"]
          numer = histograms[hname + "_numer"]
          denom.fill(abs(part.eta))
          if trigger:
            numer.fill(abs(part.eta))
          #
          trigger_allzones = any([select_track_allzones(trk) for trk in tracks])  # using scaled pT
          denom = histograms[hname_allzones + "_denom"]
          numer = histograms[hname_allzones + "_numer"]
          denom.fill(abs(part.eta))
          if trigger_allzones:
            numer.fill(abs(part.eta))

      def fill_efficiency_d0():
        if select_part(part):
          trigger = any([select_track(trk) for trk in tracks])  # using scaled pT
          denom = histograms[hname + "_denom"]
          numer = histograms[hname + "_numer"]
          denom.fill(abs(part.d0))
          if trigger:
            numer.fill(abs(part.d0))
          #
          trigger_allzones = any([select_track_allzones(trk) for trk in tracks])  # using scaled pT
          denom = histograms[hname_allzones + "_denom"]
          numer = histograms[hname_allzones + "_numer"]
          denom.fill(abs(part.d0))
          if trigger_allzones:
            numer.fill(abs(part.d0))

      def fill_resolution():
        if (part.bx == 0):
          trigger = any([select_track(trk) for trk in tracks])  # using scaled pT
          if trigger:
            trk = tracks[0]
            trk.invpt = np.true_divide(trk.q, trk.xml_pt)  # using unscaled pT
            histograms[hname1].fill(part.invpt, trk.invpt)
            histograms[hname2].fill(abs(part.invpt), abs(part.invpt/trk.invpt) - 1)

      for l in (0, 10, 15, 20, 30, 40, 50):
        tracks = evt.tracks
        select_part = lambda part: (1.24 <= abs(part.eta) <= 2.4) and (part.bx == 0)
        select_track = lambda trk: trk and (1.24 <= abs(trk.eta) <= 2.4) and (trk.bx == 0) and (trk.mode in (11,13,14,15)) and (trk.pt > float(l))
        select_track_allzones = select_track
        #
        hname = "emtf_eff_vs_genpt_l1pt%i" % (l)
        hname_allzones = "emtf_eff_vs_genpt_allzones_l1pt%i" % (l)
        fill_efficiency_pt()
        hname = "emtf_eff_vs_genpt_highpt_l1pt%i" % (l)
        hname_allzones = "emtf_eff_vs_genpt_allzones_highpt_l1pt%i" % (l)
        fill_efficiency_pt()
        if part.pt > 20.:
          hname = "emtf_eff_vs_genphi_l1pt%i" % (l)
          hname_allzones = "emtf_eff_vs_genphi_allzones_l1pt%i" % (l)
          fill_efficiency_phi()
          hname = "emtf_eff_vs_geneta_l1pt%i" % (l)
          hname_allzones = "emtf_eff_vs_geneta_allzones_l1pt%i" % (l)
          fill_efficiency_eta()
          hname = "emtf_eff_vs_gend0_l1pt%i" % (l)
          hname_allzones = "emtf_eff_vs_gend0_allzones_l1pt%i" % (l)
          fill_efficiency_d0()
        if l == 0:
          hname1 = "emtf_l1pt_vs_genpt"
          hname2 = "emtf_l1ptres_vs_genpt"
          fill_resolution()

        tracks = emtf2026_tracks
        if omtf_input:
          select_part = lambda part: (0.8 <= abs(part.eta) <= 1.24) and (part.bx == 0)
          #select_track = lambda trk: trk and (0.8 <= abs(trk.eta) <= 1.24) and trk.zone in (6,) and (trk.pt > float(l))
          select_track = lambda trk: trk and (0.75 <= abs(trk.eta) <= 1.4) and trk.zone in (6,) and (trk.pt > float(l))
          select_track_allzones = lambda trk: trk and (0.75 <= abs(trk.eta) <= 1.4) and (trk.pt > float(l))
        else:
          select_part = lambda part: (1.24 <= abs(part.eta) <= 2.4) and (part.bx == 0)
          #select_track = lambda trk: trk and (1.24 <= abs(trk.eta) <= 2.4) and trk.zone in (0,1,2,3,4,5) and (trk.pt > float(l))
          select_track = lambda trk: trk and (1.1 <= abs(trk.eta) <= 2.4) and trk.zone in (0,1,2,3,4,5) and (trk.pt > float(l))
          select_track_allzones = lambda trk: trk and (1.1 <= abs(trk.eta) <= 2.4) and (trk.pt > float(l))
        #
        hname = "emtf2026_eff_vs_genpt_l1pt%i" % (l)
        hname_allzones = "emtf2026_eff_vs_genpt_allzones_l1pt%i" % (l)
        fill_efficiency_pt()
        hname = "emtf2026_eff_vs_genpt_highpt_l1pt%i" % (l)
        hname_allzones = "emtf2026_eff_vs_genpt_allzones_highpt_l1pt%i" % (l)
        fill_efficiency_pt()
        if part.pt > 20.:
          hname = "emtf2026_eff_vs_genphi_l1pt%i" % (l)
          hname_allzones = "emtf2026_eff_vs_genphi_allzones_l1pt%i" % (l)
          fill_efficiency_phi()
          hname = "emtf2026_eff_vs_geneta_l1pt%i" % (l)
          hname_allzones = "emtf2026_eff_vs_geneta_allzones_l1pt%i" % (l)
          fill_efficiency_eta()
          hname = "emtf2026_eff_vs_gend0_l1pt%i" % (l)
          hname_allzones = "emtf2026_eff_vs_gend0_allzones_l1pt%i" % (l)
          fill_efficiency_d0()
        if l == 0:
          hname1 = "emtf2026_l1pt_vs_genpt"
          hname2 = "emtf2026_l1ptres_vs_genpt"
          fill_resolution()

    # End loop over events
    unload_tree()

    # Quick efficiency
    for l in (20, 30, 40, 50):
      for k in ("denom", "numer"):
        m = 'emtf2026'
        hname = "%s_eff_vs_genpt_l1pt%i_%s" % (m,l,k)
        h = histograms[hname]
        if k == 'numer' :
          npassed = h.GetBinContent(h.FindBin(l))
        else:
          ntotal = h.GetBinContent(h.FindBin(l))
      print('[INFO] @%i GeV npassed/ntotal: %i/%i = %f' % (l, npassed, ntotal, float(npassed)/ntotal))

    # __________________________________________________________________________
    # Save histograms
    outfile = 'histos_tbc.root'
    if use_condor:
      outfile = 'histos_tbc_%i.root' % jobid
    print('[INFO] Creating file: %s' % outfile)
    with root_open(outfile, 'recreate') as f:
      hnames = []
      for m in ("emtf", "emtf2026"):
        for l in (0, 10, 15, 20, 30, 40, 50):
          for k in ("denom", "numer"):
            hname = "%s_eff_vs_genpt_l1pt%i_%s" % (m,l,k)
            hnames.append(hname)
            hname = "%s_eff_vs_genpt_allzones_l1pt%i_%s" % (m,l,k)
            hnames.append(hname)
            hname = "%s_eff_vs_genpt_highpt_l1pt%i_%s" % (m,l,k)
            hnames.append(hname)
            hname = "%s_eff_vs_genpt_allzones_highpt_l1pt%i_%s" % (m,l,k)
            hnames.append(hname)
            hname = "%s_eff_vs_genphi_l1pt%i_%s" % (m,l,k)
            hnames.append(hname)
            hname = "%s_eff_vs_genphi_allzones_l1pt%i_%s" % (m,l,k)
            hnames.append(hname)
            hname = "%s_eff_vs_geneta_l1pt%i_%s" % (m,l,k)
            hnames.append(hname)
            hname = "%s_eff_vs_geneta_allzones_l1pt%i_%s" % (m,l,k)
            hnames.append(hname)
            hname = "%s_eff_vs_gend0_l1pt%i_%s" % (m,l,k)
            hnames.append(hname)
            hname = "%s_eff_vs_gend0_allzones_l1pt%i_%s" % (m,l,k)
            hnames.append(hname)
        hname = "%s_l1pt_vs_genpt" % m
        hnames.append(hname)
        hname = "%s_l1ptres_vs_genpt" % m
        hnames.append(hname)
      for hname in hnames:
        h = histograms[hname]
        h.Write()


# ______________________________________________________________________________
# Analysis: mixing

class MixingAnalysis(object):
  def run(self, omtf_input=False, run2_input=False, test_job=159):
    tree = load_minbias_batch_for_mixing(jobid)

    # Workers
    bank = PatternBank(bankfile)
    recog = PatternRecognition(bank, omtf_input=omtf_input, run2_input=run2_input)
    clean = RoadCleaning()
    slim = RoadSlimming(bank)
    ghost = GhostBusting()
    mucorr = TrackMuonCorrelation()
    out_particles = []
    out_roads = []
    out_aux = []

    training_phase = (jobid < test_job)
    if training_phase:
      #bx_shifts = [+2, +1, 0, -1, -2]  # makes the training worse. not sure why.
      bx_shifts = [0]
    else:
      bx_shifts = [0]

    def keep_old_bx(hits, particles):
      for hit in hits:
        hit.old_bx = hit.bx
      for part in particles:
        part.old_bx = part.bx

    def manipulate_bx(hits, particles, bx_shift):
      for hit in hits:
        hit.bx = hit.old_bx + bx_shift
      for part in particles:
        part.bx = part.old_bx + bx_shift

    def make_tracks_without_pt(roads):
      tracks = []
      for myroad in roads:
        mode = myroad.mode
        zone = myroad.id[3]
        phi_median = myroad.phi_median
        theta_median = myroad.theta_median
        xml_pt, pt, trk_q, y_pred, y_discr = 0, 0, 0, 0, 0
        trk = Track(myroad.id, myroad.hits, mode, myroad.quality, zone, xml_pt, pt, trk_q, y_pred, y_discr, phi_median, theta_median)
        trk.myroad = myroad
        tracks.append(trk)
      return tracks

    # Event range
    n = -1

    # __________________________________________________________________________
    # Loop over events
    for ievt, evt in enumerate(tree):
      if n != -1 and ievt == n:
        break

      # Remember the BX
      keep_old_bx(evt.hits, evt.particles)

      # Manipulate hit BX multiple times
      for bx_shift in bx_shifts:

        # Manipulate hit BX
        manipulate_bx(evt.hits, evt.particles, bx_shift=bx_shift)

        roads = recog.run(evt.hits)
        clean_roads = clean.run(roads)
        slim_roads = slim.run(clean_roads)
        assert(len(clean_roads) == len(slim_roads))

        tracks_without_pt = make_tracks_without_pt(slim_roads)
        emtf2026_tracks = ghost.run(tracks_without_pt)
        emtf2026_matched = mucorr.run(evt.particles, emtf2026_tracks)

        def find_highest_part_pt():
          highest_pt = -999999.
          for ipart, part in enumerate(evt.particles):
            if select_part(part):
              if highest_pt < part.pt:
                highest_pt = part.pt
          if highest_pt > 0.:
            highest_pt = min(100.-1e-4, highest_pt)
          return highest_pt

        def find_highest_track_pt():
          highest_pt = -999999.
          for itrk, trk in enumerate(evt.tracks):
            if select_track(trk):
              if highest_pt < trk.pt:  # using scaled pT
                highest_pt = trk.pt
          if highest_pt > 0.:
            highest_pt = min(100.-1e-4, highest_pt)
          return highest_pt

        if omtf_input:
          select_part = lambda part: (0.8 <= abs(part.eta) <= 1.24) and (part.bx == 0)
          select_track = lambda trk: trk and (0.8 <= abs(trk.eta) <= 1.24) and (trk.bx == 0) and (trk.mode > 0)
        else:
          select_part = lambda part: (1.24 <= abs(part.eta) <= 2.4) and (part.bx == 0)
          select_track = lambda trk: trk and (1.24 <= abs(trk.eta) <= 2.4) and (trk.bx == 0) and (trk.mode in (11,13,14,15))

        #highest_part_pt = find_highest_part_pt()
        highest_track_pt = find_highest_track_pt()

        for itrk, mytrk in enumerate(emtf2026_tracks):
          m = emtf2026_matched[:, itrk]
          if m.any():
            assert(np.squeeze(m.nonzero()).ndim == 0)
            m_ipart = np.asscalar(np.squeeze(m.nonzero()))
            m_part = evt.particles[m_ipart]
            mypart = Particle(m_part.pt, m_part.eta, m_part.phi, m_part.q, m_part.vx, m_part.vy, m_part.vz)
            highest_part_pt = m_part.pt
          else:
            mypart = Particle(0., 0., 0., -1, 0., 0., 0.)
            highest_part_pt = -999999.
          aux = (jobid, ievt, highest_part_pt, highest_track_pt)
          out_particles.append(mypart)
          out_roads.append(mytrk.myroad)
          out_aux.append(aux)

        debug_event_list = set([2826, 2937, 3675, 4581, 4838, 5379, 7640])

        if ievt < 20 or ((ievt in debug_event_list) and not training_phase):
          print("evt {0} has {1} roads, {2} clean roads, {3} old tracks, {4} new tracks".format(ievt, len(roads), len(clean_roads), len(evt.tracks), '?'))
          for iroad, myroad in enumerate(clean_roads):
            print(".. croad {0} id: {1} nhits: {2} mode: {3} qual: {4} sort: {5}".format(iroad, myroad.id, len(myroad.hits), myroad.mode, myroad.quality, myroad.sort_code))
            for ihit, myhit in enumerate(myroad.hits):
              print(".. .. hit {0} id: {1} lay: {2} ph: {3} th: {4} tp: {5}".format(ihit, myhit.id, myhit.emtf_layer, myhit.emtf_phi, myhit.emtf_theta, myhit.sim_tp))

    # End loop over events
    unload_tree()

    # __________________________________________________________________________
    # Save objects
    outfile = 'histos_tbd.npz'
    if use_condor:
      outfile = 'histos_tbd_%i.npz' % jobid
    print('[INFO] Creating file: %s' % outfile)
    if True:
      assert(len(out_particles) == len(out_roads))
      assert(len(out_particles) == len(out_aux))
      parameters = particles_to_parameters(out_particles)
      variables = roads_to_variables(out_roads)
      out_aux = np.array(out_aux, dtype=np.float32)
      np.savez_compressed(outfile, parameters=parameters, variables=variables, aux=out_aux)


# ______________________________________________________________________________
# Analysis: collusion

class CollusionAnalysis(object):
  def run(self, omtf_input=False, run2_input=False):
    # Load tree
    tree = load_minbias_batch_for_collusion(jobid)

    # Workers
    bank = PatternBank(bankfile)
    recog = PatternRecognition(bank, omtf_input=omtf_input, run2_input=run2_input)
    clean = RoadCleaning()
    slim = RoadSlimming(bank)
    out_particles = []
    out_roads = []

    # Event range
    n = -1

    # __________________________________________________________________________
    # Loop over events
    for ievt, evt in enumerate(tree):
      if n != -1 and ievt == n:
        break

      if len(evt.particles) == 0:
        continue

      myparticles = []
      myparticles_sim_tp = []

      # Select genParticles
      select_part = lambda part: (part.status == 1) and (part.bx == 0) and (np.abs(part.eta) > 1.)

      for ipart, part in enumerate(evt.particles):
        if select_part(part):
          mypart = Particle(part.pt, part.eta, part.phi, part.q, part.vx, part.vy, part.vz)
          myparticles.append(mypart)
          myparticles_sim_tp.append(ipart)

      if len(myparticles) == 0:
        continue

      # Sanity check
      assert(len(myparticles) == 2)
      assert(myparticles_sim_tp[0] == 0 and myparticles_sim_tp[1] == 1)

      roads = recog.run(evt.hits)
      clean_roads = clean.run(roads)
      slim_roads = slim.run(clean_roads)
      assert(len(clean_roads) == len(slim_roads))

      myroad_0, myroad_1 = None, None

      # Match to genParticles
      for iroad, myroad in enumerate(slim_roads):
        for ihit, myhit in enumerate(myroad.hits):
          if (myhit.bx == 0) and (myhit.station == 1):
            if (not myroad_0) and (myhit.sim_tp == myparticles_sim_tp[0]):
              myroad_0 = myroad
            elif (not myroad_1) and (myhit.sim_tp == myparticles_sim_tp[1]):
              myroad_1 = myroad

      if (myroad_0 is None) or (myroad_1 is None):
        for iroad, myroad in enumerate(slim_roads):
          myroad_sim_tp_set = set()
          for ihit, myhit in enumerate(myroad.hits):
            if (myhit.bx == 0):
              if (myhit.sim_tp != -1):
                myroad_sim_tp_set.add(myhit.sim_tp)
          if len(myroad_sim_tp_set) == 1:
            myroad_sim_tp = myroad_sim_tp_set.pop()
            if (not myroad_0) and (myroad_sim_tp == myparticles_sim_tp[0]):
              myroad_0 = myroad
            elif (not myroad_1) and (myroad_sim_tp == myparticles_sim_tp[1]):
              myroad_1 = myroad

      if not ((myroad_0 is None) or (myroad_1 is None)):
        out_particles.append(myparticles[0])
        out_particles.append(myparticles[1])
        out_roads.append(myroad_0)
        out_roads.append(myroad_1)

      if ievt < 20:
        print("evt {0} has {1} roads and {2} clean roads".format(ievt, len(roads), len(clean_roads)))
        for ipart, part in enumerate(evt.particles):
          if select_part(part):
            part.invpt = np.true_divide(part.q, part.pt)
            print(".. part invpt: {0} pt: {1} phi: {2} eta: {3} theta: {4}".format(part.invpt, part.pt, part.phi, part.eta, part.theta))
        for iroad, myroad in enumerate(slim_roads):
          print(".. sroad {0} id: {1} nhits: {2} mode: {3} qual: {4} sort: {5}".format(iroad, myroad.id, len(myroad.hits), myroad.mode, myroad.quality, myroad.sort_code))
          for ihit, myhit in enumerate(myroad.hits):
            print(".. .. hit {0} id: {1} lay: {2} ph: {3} th: {4} tp: {5}".format(ihit, myhit.id, myhit.emtf_layer, myhit.emtf_phi, myhit.emtf_theta, myhit.sim_tp))
        if not ((myroad_0 is None) or (myroad_1 is None)):
          for iroad, myroad in enumerate([myroad_0, myroad_1]):
            print(".. sroad {0} id: {1} nhits: {2} mode: {3} qual: {4} sort: {5}".format('+' if iroad==0 else '-', myroad.id, len(myroad.hits), myroad.mode, myroad.quality, myroad.sort_code))

    # End loop over events
    unload_tree()

    # __________________________________________________________________________
    # Save objects
    outfile = 'histos_tbe.npz'
    if use_condor:
      outfile = 'histos_tbe_%i.npz' % jobid
    print('[INFO] Creating file: %s' % outfile)
    if True:
      assert(len(out_particles) == len(out_roads))
      parameters = particles_to_parameters(out_particles)
      variables = roads_to_variables(out_roads)
      np.savez_compressed(outfile, parameters=parameters, variables=variables)


# ______________________________________________________________________________
# Analysis: images

class ImagesAnalysis(object):
  def run(self, omtf_input=False, run2_input=False):
    # Load tree
    if omtf_input:
      tree = load_pgun_omtf()
    else:
      tree = load_pgun()

    out_part = []
    out_hits = []

    # __________________________________________________________________________
    # Loop over events
    for ievt, evt in enumerate(tree):
      if maxEvents != -1 and ievt == maxEvents:
        break

      # Skip events with very few hits
      if omtf_input:
        if not len(evt.hits) >= 2:
          continue
      else:
        if not len(evt.hits) >= 3:
          continue

      # Skip events without ME1 hits
      has_ME1 = False
      for ihit, hit in enumerate(evt.hits):
        if hit.sim_tp1 == 0:
          if hit.type == kCSC and hit.station == 1:
            has_ME1 = True
            break
          elif hit.type == kME0 and hit.station == 1:
            has_ME1 = True
            break
          elif hit.type == kDT and (hit.station == 1 or hit.station == 2):
            has_ME1 = True
            break
      if not has_ME1:
        continue

      part = evt.particles[0]  # particle gun
      part.invpt = np.true_divide(part.q, part.pt)
      part.d0 = calculate_d0(part.invpt, part.phi, part.vx, part.vy)

      images_hits = filter(is_emtf_images_hit, evt.hits)

      # Find the best sector (using csc-only 'mode')
      sector_mode_array = np.zeros((12,), dtype=np.int32)
      sector_hits_array = np.empty((12,), dtype=np.object)
      for ind in np.ndindex(sector_hits_array.shape):
        sector_hits_array[ind] = []

      # Loop over hits
      for ihit, hit in enumerate(images_hits):
        #assert(hit.emtf_phi < 5040)  # 84*60
        assert(hit.emtf_phi < 5400)  # 90*60

        endsec = find_endsec(hit.endcap, hit.sector)
        sector_hits_array[endsec].append(hit)

        if hit.sim_tp1 == 0:
          if hit.type == kCSC:
            sector_mode_array[endsec] |= (1 << (4 - hit.station))
          elif hit.type == kME0:
            sector_mode_array[endsec] |= (1 << (4 - 1))
          elif hit.type == kDT:
            sector_mode_array[endsec] |= (1 << (4 - 1))

      # Get the best sector
      #best_sector = np.argmax(sector_mode_array)
      best_sector = np.argmax(sector_mode_array * 100 + [len(x) for x in sector_hits_array])
      mode = sector_mode_array[best_sector]

      # Skip events without station 1
      if not is_emtf_singlehit(mode):
        continue

      # Get the hits
      sector_hits = sector_hits_array[best_sector]

      amap = {}  # zone -> hits

      # Loop over sector hits
      for ihit, hit in enumerate(sector_hits):
        hit.emtf_layer = find_emtf_layer(hit)
        assert(hit.emtf_layer != -99)

        zones = find_emtf_zones(hit)
        for z in zones:
          amap.setdefault(np.asscalar(z), []).append(hit)
        continue  # end loop over sector_hits

      # Loop over map of zone -> hits
      ievt_part = []
      ievt_hits = []

      for k, v in amap.iteritems():
        zone = k
        hits = v

        # Skip zones with very few hits
        if not ((zone in (0,1,2,3,4) and len(hits) >= 3) or (zone in (5,6) and len(hits) >= 2)):
          continue

        zone_mode = 0
        for ihit, hit in enumerate(hits):
          if hit.sim_tp1 == 0:
            if hit.type == kCSC:
              zone_mode |= (1 << (4 - hit.station))
            elif hit.type == kME0:
              zone_mode |= (1 << (4 - 1))
            elif hit.type == kDT:
              zone_mode |= (1 << (4 - 1))

        # Skip zones without station 1
        if not is_emtf_singlehit(zone_mode):
          continue

        zone_layers = np.zeros(16, dtype=np.bool)
        for ihit, hit in enumerate(hits):
          zone_layers[hit.emtf_layer] = True

        # Skip zones without at least 2 stations
        if not (zone_layers.sum() >= 2):
          continue

        # Particle info
        part_info = np.array([part.invpt, part.eta, part.phi, part.d0, zone, best_sector])

        # Hits info
        #get_hit_info = lambda hit: (hit.emtf_layer, hit.emtf_phi)
        get_hit_info = lambda hit: (hit.emtf_layer, find_emtf_phi(hit))
        hits_info = np.array([get_hit_info(hit) for hit in hits])

        ievt_part.append(part_info)
        ievt_hits.append(hits_info)
        continue  # end loop over map of zone -> hits

      if ievt < 20:
        ievt_nhits = [len(x) for x in ievt_hits]
        print ievt, part.pt, ievt_part, ievt_nhits

      # Output
      out_part += ievt_part
      out_hits += ievt_hits
      continue  # end loop over events

    # End loop over events
    unload_tree()

    # __________________________________________________________________________
    # Save objects
    outfile = 'histos_tbf.npz'
    if use_condor:
      outfile = 'histos_tbf_%i.npz' % jobid
    print('[INFO] Creating file: %s' % outfile)
    if True:
      assert(len(out_part) == len(out_hits))
      out_part = np.asarray(out_part, dtype=np.float32)
      out_hits = create_ragged_array(out_hits)
      print out_part.shape, out_hits.shape
      np.savez_compressed(outfile, out_part=out_part, out_hits_values=out_hits.values, out_hits_row_splits=out_hits.row_splits)


# ______________________________________________________________________________
# Settings

# Get number of events
#maxEvents = -1
#maxEvents = 4000000
maxEvents = 2000000
#maxEvents = 1000

# Condor or not
use_condor = ('CONDOR_EXEC' in os.environ)
if use_condor:
  os.environ['ROOTPY_GRIDMODE'] = 'true'

# Algorithm (pick one)
algo = 'default'  # phase 2
#algo = 'run3'
#algo = 'omtf'
if use_condor:
  algo = sys.argv[1]

# Analysis mode (pick one)
#analysis = 'dummy'
#analysis = 'roads'
#analysis = 'rates'
analysis = 'effie'
#analysis = 'mixing'
#analysis = 'collusion'
#analysis = 'images'
if use_condor:
  analysis = sys.argv[2]

# Job id
jobid = 0
if use_condor:
  jobid = int(sys.argv[3])


# Input files
bankfile = 'pattern_bank_18patt.27.npz'

kerasfile = ['model.27.json', 'model_weights.27.h5',
             'model_run3.27.json', 'model_run3_weights.27.h5',
             'model_omtf.27.json', 'model_omtf_weights.27.h5',]

infile_r = None  # input file handle

def purge_bad_files(infiles):
  good_files = []
  for infile in infiles:
    try:
      _ = TreeChain('ntupler/tree', infile)
      good_files.append(infile)
    except:
      pass
  return good_files

def define_collections(tree):
  tree.define_collection(name='hits', prefix='vh_', size='vh_size')
  tree.define_collection(name='tracks', prefix='vt_', size='vt_size')
  tree.define_collection(name='particles', prefix='vp_', size='vp_size')
  tree.define_collection(name='evt_info', prefix='ve_', size='ve_size')
  return

def load_tree_single(infile):
  print('[INFO] Opening file: %s' % infile)
  global infile_r
  infile_r = root_open(infile)
  tree = infile_r.ntupler.tree
  define_collections(tree)
  return tree

def load_tree_multiple(infiles):
  print('[INFO] Opening file: %s' % ' '.join(infiles))
  tree = TreeChain('ntupler/tree', infiles)
  define_collections(tree)
  return tree

def unload_tree():
  global infile_r
  try:
    infile_r.close()
  except:
    pass

# ______________________________________________________________________________
def load_pgun():
  infile = 'ntuple_SingleMuon_Endcap_2GeV_add.5.root'
  return load_tree_single(infile)

def load_pgun_batch(k):
  jj = np.split(np.arange(100), 100)[k]
  infiles = []
  for j in jj:
    infiles.append('root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_4_0/ntuple_SingleMuon_Endcap_2GeV/ParticleGuns/CRAB3/190416_194707/%04i/ntuple_SingleMuon_Endcap_%i.root' % ((j+1)/1000, (j+1)))
    infiles.append('root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_4_0/ntuple_SingleMuon_Endcap2_2GeV/ParticleGuns/CRAB3/190416_194826/%04i/ntuple_SingleMuon_Endcap2_%i.root' % ((j+1)/1000, (j+1)))
  return load_tree_multiple(infiles)

def load_pgun_omtf():
  infile = 'ntuple_SingleMuon_Overlap_3GeV_add.5.root'
  return load_tree_single(infile)

def load_pgun_batch_omtf(k):
  jj = np.split(np.arange(50), 50)[k]
  infiles = []
  for j in jj:
    infiles.append('root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_4_0/ntuple_SingleMuon_Overlap_3GeV/ParticleGuns/CRAB3/190416_194944/%04i/ntuple_SingleMuon_Overlap_%i.root' % ((j+1)/1000, (j+1)))
    infiles.append('root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_4_0/ntuple_SingleMuon_Overlap2_3GeV/ParticleGuns/CRAB3/190416_195101/%04i/ntuple_SingleMuon_Overlap2_%i.root' % ((j+1)/1000, (j+1)))
  #infiles = purge_bad_files(infiles)
  return load_tree_multiple(infiles)

def load_pgun_batch_displ(k):
  jj = np.split(np.arange(2000), 200)[k]
  infiles = []
  if j < 100:
    for j in jj:
      infiles.append('root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_4_0/SingleMuon_Displaced_2GeV/ParticleGuns/CRAB3/190405_195607/%04i/ntuple_SingleMuon_Displaced_%i.root' % ((j+1)/1000, (j+1)))
      infiles.append('root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_4_0/SingleMuon_Displaced2_2GeV/ParticleGuns/CRAB3/190405_195725/%04i/ntuple_SingleMuon_Displaced2_%i.root' % ((j+1)/1000, (j+1)))
  else:
    for j in jj:
      infiles.append('root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_4_0/SingleMuon_Displaced_FlatPhi_2GeV/ParticleGuns/CRAB3/190405_201844/%04i/ntuple_SingleMuon_Displaced_FlatPhi_%i.root' % ((j-1000+1)/1000, (j-1000+1)))
      infiles.append('root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_4_0/SingleMuon_Displaced2_FlatPhi_2GeV/ParticleGuns/CRAB3/190405_202001/%04i/ntuple_SingleMuon_Displaced2_FlatPhi_%i.root' % ((j-1000+1)/1000, (j-1000+1)))
  return load_tree_multiple(infiles)

def load_minbias_batch(k, pileup=200):
  if pileup == 140:
    pufiles = ['root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_4_0/ntuple_SingleNeutrino_PU140/SingleNeutrino/CRAB3/190416_160046/0000/ntuple_SingleNeutrino_PU140_%i.root' % (i+1) for i in xrange(56)]
  elif pileup == 200:
    pufiles = ['root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_4_0/ntuple_SingleNeutrino_PU200/SingleNeutrino/CRAB3/190416_160207/0000/ntuple_SingleNeutrino_PU200_%i.root' % (i+1) for i in xrange(63)]
  elif pileup == 250:
    pufiles = ['root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_4_0/ntuple_SingleNeutrino_PU250/SingleNeutrino/CRAB3/190416_160323/0000/ntuple_SingleNeutrino_PU250_%i.root' % (i+1) for i in xrange(50)]
  elif pileup == 300:
    pufiles = ['root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_4_0/ntuple_SingleNeutrino_PU300/SingleNeutrino/CRAB3/190416_160441/0000/ntuple_SingleNeutrino_PU300_%i.root' % (i+1) for i in xrange(53)]
  else:
    raise RunTimeError('Cannot recognize pileup: {0}'.format(pileup))
  #
  infile = pufiles[k]
  return load_tree_single(infile)

def load_minbias_batch_for_mixing(k):
  pufiles = []
  # For training purposes
  pufiles += ['root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_4_0/ntuple_SingleNeutrino_PU140/SingleNeutrino/CRAB3/190416_160046/0000/ntuple_SingleNeutrino_PU140_%i.root' % (i+1) for i in xrange(20)]  # up to 20/56
  pufiles += ['root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_4_0/ntuple_SingleNeutrino_PU200/SingleNeutrino/CRAB3/190416_160207/0000/ntuple_SingleNeutrino_PU200_%i.root' % (i+1) for i in xrange(30)]  # up to 30/63
  #pufiles += ['root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_4_0/ntuple_SingleNeutrino_PU250/SingleNeutrino/CRAB3/190416_160323/0000/ntuple_SingleNeutrino_PU250_%i.root' % (i+1) for i in xrange(20)]  # up to 20/50
  #pufiles += ['root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_4_0/ntuple_SingleNeutrino_PU300/SingleNeutrino/CRAB3/190416_160441/0000/ntuple_SingleNeutrino_PU300_%i.root' % (i+1) for i in xrange(20)]  # up to 20/53
  pufiles += ['root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_4_0/ntuple_SingleElectron_PU140/SingleE_FlatPt-2to100/CRAB3/190416_160602/0000/ntuple_SingleElectron_PU140_%i.root' % (i+1) for i in xrange(28)]
  pufiles += ['root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_4_0/ntuple_SingleElectron_PU200/SingleE_FlatPt-2to100/CRAB3/190416_160719/0000/ntuple_SingleElectron_PU200_%i.root' % (i+1) for i in xrange(27)]
  #pufiles += ['root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_4_0/ntuple_SingleMuon_PU140/SingleMu_FlatPt-2to100/CRAB3/190416_160834/0000/ntuple_SingleMuon_PU140_%i.root' % (i+1) for i in xrange(25)]
  #pufiles += ['root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_4_0/ntuple_SingleMuon_PU200/SingleMu_FlatPt-2to100/CRAB3/190416_160951/0000/ntuple_SingleMuon_PU200_%i.root' % (i+1) for i in xrange(26)]
  pufiles += ['root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_4_0/ntuple_SinglePhoton_PU140/SinglePhoton_FlatPt-8to150/CRAB3/190416_161116/0000/ntuple_SinglePhoton_PU140_%i.root' % (i+1) for i in xrange(27)]
  pufiles += ['root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_4_0/ntuple_SinglePhoton_PU200/SinglePhoton_FlatPt-8to150/CRAB3/190416_161239/0000/ntuple_SinglePhoton_PU200_%i.root' % (i+1) for i in xrange(27)]
  # For testing purposes (SingleNeutrino, PU200)
  pufiles += ['root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_4_0/ntuple_SingleNeutrino_PU200/SingleNeutrino/CRAB3/190416_160207/0000/ntuple_SingleNeutrino_PU200_%i.root' % (i+1) for i in xrange(30,63)]  # from 30/63
  #
  infile = pufiles[k]
  return load_tree_single(infile)

def load_minbias_batch_for_collusion(k):
  pufiles = []
  pufiles += ['root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_4_0/ntuple_SingleMuon_PU140/SingleMu_FlatPt-2to100/CRAB3/190416_160834/0000/ntuple_SingleMuon_PU140_%i.root' % (i+1) for i in xrange(25)]
  pufiles += ['root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_4_0/ntuple_SingleMuon_PU200/SingleMu_FlatPt-2to100/CRAB3/190416_160951/0000/ntuple_SingleMuon_PU200_%i.root' % (i+1) for i in xrange(26)]
  #
  infile = pufiles[k]
  return load_tree_single(infile)


# ______________________________________________________________________________
# Main

if __name__ == "__main__":
  print('[INFO] Current time    : {0}'.format(datetime.datetime.now()))
  print('[INFO] Using cmssw     : {0}'.format(os.environ['CMSSW_VERSION']))
  print('[INFO] Using condor    : {0}'.format(use_condor))
  print('[INFO] Using max events: {0}'.format(maxEvents))
  print('[INFO] Using algo      : {0}'.format(algo))
  print('[INFO] Using analysis  : {0}'.format(analysis))
  print('[INFO] Using job id    : {0}'.format(jobid))

  if algo == 'run3':
    run2_input = True
  else:
    run2_input = False

  if algo == 'omtf':
    omtf_input = True
  else:
    omtf_input = False

  if analysis == 'dummy':
    analysis = DummyAnalysis()
    analysis.run(omtf_input=omtf_input, run2_input=run2_input)

  elif analysis == 'roads':
    analysis = RoadsAnalysis()
    analysis.run(omtf_input=omtf_input, run2_input=run2_input)

  elif analysis == 'rates':
    analysis = RatesAnalysis()
    analysis.run(omtf_input=omtf_input, run2_input=run2_input, pileup=200)
  elif analysis == 'rates140':
    analysis = RatesAnalysis()
    analysis.run(omtf_input=omtf_input, run2_input=run2_input, pileup=140)
  elif analysis == 'rates250':
    analysis = RatesAnalysis()
    analysis.run(omtf_input=omtf_input, run2_input=run2_input, pileup=250)
  elif analysis == 'rates300':
    analysis = RatesAnalysis()
    analysis.run(omtf_input=omtf_input, run2_input=run2_input, pileup=300)

  elif analysis == 'effie':
    analysis = EffieAnalysis()
    analysis.run(omtf_input=omtf_input, run2_input=run2_input)

  elif analysis == 'mixing':
    analysis = MixingAnalysis()
    analysis.run(omtf_input=omtf_input, run2_input=run2_input)

  elif analysis == 'collusion':
    analysis = CollusionAnalysis()
    analysis.run(omtf_input=omtf_input, run2_input=run2_input)

  elif analysis == 'images':
    analysis = ImagesAnalysis()
    analysis.run(omtf_input=omtf_input, run2_input=run2_input)

  else:
    raise RunTimeError('Cannot recognize analysis: {0}'.format(analysis))
