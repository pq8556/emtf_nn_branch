import numpy as np

nlayers = 16  # 5 (CSC) + 4 (RPC) + 3 (GEM) + 4 (DT)

nvariables = 36  # 20 (CSC) + 8 (RPC) + 4 (GEM) + 4 (ME0)

nvariables_input = (nlayers * (9+1)) + 4

nparameters_input = 6


# ______________________________________________________________________________
class Encoder(object):

  def __init__(self, x, y, reg_pt_scale=1.0, reg_dxy_scale=1.0,
               drop_ge11=True, drop_ge21=True, drop_me0=True,
               drop_irpc=True, drop_dt=True):

    if x is None or y is None:
      raise Exception('Invalid input x or y')

    assert(x.shape[1] == nvariables_input)
    if y.shape[1] == 1:
      y = np.zeros((y.shape[0], nparameters_input), dtype=np.float32)
    else:
      assert(y.shape[1] == nparameters_input)
    assert(x.shape[0] == y.shape[0])

    self.nentries = x.shape[0]
    self.x_orig  = x
    self.y_orig  = y
    self.x_copy  = x.copy()
    self.y_copy  = y.copy()

    # ________________________________________________________________________
    # Get views

    # Each layer can have 0 or 1 hit. If there is a hit, each hit has several
    # features (phi, theta, bend, quality, time, etc). Currently there are 9.
    # Each layer also has a 'mask' variable that indicates if there is a hit
    # (0: has hit, 1: no hit).
    # Additionally, each road has 4 features (straightness, zone, phi_median,
    # theta_median).
    # Note that some inputs are not actually used.
    self.x_phi       = self.x_copy[:, nlayers*0:nlayers*1]
    self.x_theta     = self.x_copy[:, nlayers*1:nlayers*2]
    self.x_bend      = self.x_copy[:, nlayers*2:nlayers*3]
    self.x_qual      = self.x_copy[:, nlayers*3:nlayers*4]
    self.x_time      = self.x_copy[:, nlayers*4:nlayers*5]
    self.x_ring      = self.x_copy[:, nlayers*5:nlayers*6]
    self.x_fr        = self.x_copy[:, nlayers*6:nlayers*7]
    self.x_old_phi   = self.x_copy[:, nlayers*7:nlayers*8]
    self.x_old_bend  = self.x_copy[:, nlayers*8:nlayers*9]
    self.x_mask      = self.x_copy[:, nlayers*9:nlayers*10].astype(np.bool)  # this makes a copy
    self.x_road      = self.x_copy[:, nlayers*10:nlayers*11]
    self.y_pt        = self.y_copy[:, 0]  # q/pT
    self.y_phi       = self.y_copy[:, 1]
    self.y_eta       = self.y_copy[:, 2]
    self.y_vx        = self.y_copy[:, 3]
    self.y_vy        = self.y_copy[:, 4]
    self.y_vz        = self.y_copy[:, 5]

    # Find d0 (dxy is a misnomer)
    #self.y_dxy  = self.y_vx * np.sin(self.y_phi) - self.y_vy * np.cos(self.y_phi)  # valid for a straight track
    _invPt = self.y_pt.astype(np.float64, copy=True)  # needs double precision
    _invPt = np.where(np.abs(_invPt) < 1./10000, np.sign(_invPt+1e-15) * 1./10000, _invPt)
    _R = -1.0 / (0.003 * 3.811 * _invPt)           # R = -pT/(0.003 q B)  [cm]
    _xc = self.y_vx - (_R * np.sin(self.y_phi))    # xc = xv - R sin(phi)
    _yc = self.y_vy + (_R * np.cos(self.y_phi))    # yc = yv + R cos(phi)
    _d0 = _R - (np.sign(_R) * np.hypot(_xc, _yc))  # d0 = R - sqrt(xc^2 + yc^2) * sign(R)
    self.y_dxy = _d0.astype(np.float32, casting='same_kind')

    # Scale q/pT for training
    self.y_pt  *= reg_pt_scale

    # Scale dxy for training
    self.y_dxy *= reg_dxy_scale

    # ________________________________________________________________________
    # Drop detectors
    x_dropit = self.x_mask.copy()
    if drop_ge11:
      x_dropit[:, 9] = 1  # 9: GE1/1
    if drop_ge21:
      x_dropit[:, 10] = 1 # 10: GE2/1
    if drop_me0:
      x_dropit[:, 11] = 1 # 11: ME0
    if drop_irpc:
      x_ring_tmp = self.x_ring.astype(np.int32)
      x_ring_tmp = (x_ring_tmp == 2) | (x_ring_tmp == 3)
      x_dropit[~x_ring_tmp[:,7], 7] = 1  # 7: RE3, neither ring2 nor ring3
      x_dropit[~x_ring_tmp[:,8], 8] = 1  # 8: RE4, neither ring2 nor ring3
    if drop_dt:
      x_dropit[:, 12:16] = 1 # 12,13,14,15: MB1,2,3,4

    self.x_phi      [x_dropit] = np.nan
    self.x_theta    [x_dropit] = np.nan
    self.x_bend     [x_dropit] = np.nan
    self.x_qual     [x_dropit] = np.nan
    self.x_time     [x_dropit] = np.nan
    self.x_ring     [x_dropit] = np.nan
    self.x_fr       [x_dropit] = np.nan
    self.x_old_phi  [x_dropit] = np.nan
    self.x_old_bend [x_dropit] = np.nan
    self.x_mask     [x_dropit] = 1

    # ________________________________________________________________________
    # Straightness & zone
    self.x_straightness  = self.x_road[:, 0][:, np.newaxis]
    self.x_zone          = self.x_road[:, 1][:, np.newaxis]

    # Subtract median phi from hit phis
    self.x_phi_median    = self.x_road[:, 2][:, np.newaxis]
    self.x_phi          -= self.x_phi_median
    self.x_old_phi      -= self.x_phi_median

    # Subtract median theta from hit thetas
    self.x_theta_median  = self.x_road[:, 3][:, np.newaxis]
    #self.x_theta        -= self.x_theta_median

    # Modify ring and F/R definitions
    x_ring_tmp = self.x_ring.astype(np.int32)
    self.x_ring[(x_ring_tmp == 2) | (x_ring_tmp == 3)] = +1 # ring 2,3 -> +1
    self.x_ring[(x_ring_tmp == 1) | (x_ring_tmp == 4)] = -1 # ring 1,4 -> -1
    x_fr_tmp = self.x_fr.astype(np.int32)
    self.x_fr[(x_fr_tmp == 1)] = +1  # front chamber -> +1
    self.x_fr[(x_fr_tmp == 0)] = -1  # rear chamber  -> -1

    # ________________________________________________________________________
    # Add dedicated GEM-CSC bend
    # Need to account for ME1/1 f or r
    #self.x_gem_csc_bend = (self.x_orig[:,9] - self.x_orig[:,0])         # 9: GE1/1, 0: ME1/1
    #self.x_gem_csc_bend[(self.x_mask[:,9] | self.x_mask[:,0])] = np.nan # 9: GE1/1, 0: ME1/1
    #self.x_gem_csc_bend = np.hstack((self.x_gem_csc_bend[:,np.newaxis], self.x_gem_csc_bend[:,np.newaxis]))
    #self.x_gem_csc_bend[(self.x_fr[:,0]!=0),0] = np.nan  # for ME1/1r bend, set ME1/1f to nan
    #self.x_gem_csc_bend[(self.x_fr[:,0]!=1),1] = np.nan  # for ME1/1f bend, set ME1/1r to nan

    # ________________________________________________________________________
    # Remove NaN
    self._handle_nan_in_x(self.x_copy)
    #self._handle_nan_in_x(self.x_gem_csc_bend)
    return

  # Copied from scikit-learn
  def _handle_zero_in_scale(self, scale):
    scale[scale == 0.0] = 1.0
    return scale

  def _handle_nan_in_x(self, x):
    x[np.isnan(x)] = 0.0
    return x

  def get_x(self, drop_columns_of_zeroes=True, drop_columns_emtf=True, drop_columns_omtf=False):
    #x_new = np.hstack((self.x_phi, self.x_theta, self.x_bend, self.x_qual, self.x_time))
    x_new = np.hstack((self.x_old_phi, self.x_theta, self.x_old_bend, self.x_fr, self.x_time))

    # Drop input nodes
    if drop_columns_of_zeroes:
      drop_phi    = [nlayers*0 + x for x in xrange(0,0)]   # keep everyone
      drop_theta  = [nlayers*1 + x for x in xrange(0,0)]   # keep everyone
      drop_bend   = [nlayers*2 + x for x in xrange(5,11)]  # no bend for RPC, GEM
      drop_qual   = [nlayers*3 + x for x in xrange(5,11)]  # no qual for RPC, GEM
      drop_time   = [nlayers*4 + x for x in xrange(0,16)]  # no time for everyone

      x_dropit = np.zeros(x_new.shape[1], dtype=np.bool)
      for i in drop_phi + drop_theta + drop_bend + drop_qual + drop_time:
        x_dropit[i] = True
      x_new = x_new[:, ~x_dropit]

    # Drop more input nodes (in EMTF mode)
    if drop_columns_emtf:
      drop_phi    = [nlayers*0 + x for x in [12,13,14,15]]  # drop MB1, MB2, MB3, MB4
      drop_theta  = [nlayers*1 + x for x in [12,13,14,15]]  # drop MB1, MB2, MB3, MB4
      drop_bend   = [nlayers*2 + x for x in [6,7,8,9]]      # drop MB1, MB2, MB3, MB4
      drop_qual   = [nlayers*2 + x for x in [16,17,18,19]]  # drop MB1, MB2, MB3, MB4
      drop_time   = [nlayers*2 + x for x in []]             # drop nothing
      #
      x_dropit = np.zeros(x_new.shape[1], dtype=np.bool)
      for i in drop_phi + drop_theta + drop_bend + drop_qual + drop_time:
        x_dropit[i] = True
      x_new = x_new[:, ~x_dropit]

    # Drop more input nodes (in OMTF mode)
    if drop_columns_omtf:
      drop_phi    = [nlayers*0 + x for x in [0,4,8,9,10,11,15]] # drop ME1/1, ME4, RE4, GE1/1, GE2/1, ME0, MB4
      drop_theta  = [nlayers*1 + x for x in [0,4,8,9,10,11,15]] # drop ME1/1, ME4, RE4, GE1/1, GE2/1, ME0, MB4
      drop_bend   = [nlayers*2 + x for x in [0,4,5,9]]          # drop ME1/1, ME4, ME0, MB4
      drop_qual   = [nlayers*2 + x for x in [10,14,15,19]]      # drop ME1/1, ME4, ME0, MB4
      drop_time   = [nlayers*2 + x for x in []]                 # drop nothing
      #
      x_dropit = np.zeros(x_new.shape[1], dtype=np.bool)
      for i in drop_phi + drop_theta + drop_bend + drop_qual + drop_time:
        x_dropit[i] = True
      x_new = x_new[:, ~x_dropit]
      #
      x_rsvd = np.zeros((x_new.shape[0],6), dtype=np.float32)
      x_new = np.hstack((x_new, x_rsvd))
    return x_new

  def get_x_mask(self):
    x_mask = self.x_mask.copy()
    return x_mask

  def get_x_road(self):
    x_road = self.x_road.copy()
    return x_road

  def get_y(self):
    y_new = self.y_pt.copy()
    return y_new

  def get_y_corrected_for_eta(self):
    y_new = self.y_pt * (np.sinh(1.8587) / np.sinh(np.abs(self.y_eta)))
    return y_new

  def get_dxy(self):
    dxy_new = self.y_dxy.copy()
    return dxy_new

  def get_dz(self):
    dz_new = self.y_vz.copy()
    return dz_new

  def get_w(self):
    w_new = np.ones_like(self.y_pt)
    return w_new


# ______________________________________________________________________________
def create_encoder(x, y=None, reg_pt_scale=100., reg_dxy_scale=0.4):
  if y is None:
    y = np.zeros((x.shape[0], 1), dtype=np.float32)
  encoder = Encoder(x, y, reg_pt_scale, reg_dxy_scale)
  return encoder
