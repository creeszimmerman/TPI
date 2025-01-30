# -*- coding: utf-8 -*-
"""
Parameters file - imported into the scripts. periodic (non-periodic commented out)
"""

import numpy as np

# data loading
newpath = 'MC_simulation_data' # folder in which data are stored
nssfile = 'newsnapshotstarts.npy' # npy file containing index of start of each new snapshot
readfile = 'Positions' # hdf file within newpath folder containing positions
newpath_matrices = 'Matrices Folder' 
midpoints_file = 'midpoints.hdf'
gr_file = 'periodic_BC_gr.hdf' # periodic
# gr_file = 'nonperiodic_BC_gr.hdf' # nonperiodic

# for distance-histogram
sigma = 1 # diameter
bins = 500
cutoff = sigma*5 # cutoff up to which we want to calculate u(r)
bin_width = cutoff/bins
kT = 1
beta = 1/kT
small_number = 10**-20

# for test-particle insertion
num_testparticles = 10**4
root = int(np.sqrt(num_testparticles))
num_testparticles = root**2 # 2D code
multiplier = 1

# for inversion
optimisations = 250
