# -*- coding: utf-8 -*-
"""
matrices. periodic (non-periodic commented out). 2D
"""

import pandas as pd
import numpy as np
np.seterr(all='ignore')
from scipy.spatial import KDTree
import time
from scipy import sparse
import os

#importing parameters
from Parameters import newpath, nssfile, readfile, newpath_matrices, cutoff, root, bins

if not os.path.exists(newpath_matrices): #creating the folder if it doesn't exist 
    os.makedirs(newpath_matrices)

#timing the script
start = time.time()

# 1. LOAD FILES

xlength = np.load('{}/xlength.npy'.format(newpath))
ylength = np.load('{}/ylength.npy'.format(newpath))

#read entire simulation output into pandas dataframe.
readpositions = pd.read_hdf('{}/{}.hdf'.format(newpath, readfile)) #reading the positions file

#convert pandas dataframe to faster numpy n-d array
arraypositions = readpositions.to_numpy()

#determines where one snapshot ends and another begins
newsnapshotstarts = np.load('{}/{}'.format(newpath, nssfile))


# 2. DEFINITIONS

#generate 10**4 lattice points
def generate_test_particle_positions(): 
    xpositions = np.linspace(start=0, stop = xlength, endpoint = False, num = root) # periodic
    ypositions = np.linspace(start=0, stop = ylength, endpoint = False, num = root) # periodic
  #  xpositions = np.linspace(start=cutoff, stop = xlength-cutoff, endpoint = False, num = root) # non-periodic
  #  ypositions = np.linspace(start=cutoff, stop= ylength-cutoff, endpoint = False, num = root) # non-periodic
    xx, yy = np.meshgrid(xpositions, ypositions)
    testpositions = np.column_stack((xx.flatten(), yy.flatten()))
    return testpositions
testpositions = generate_test_particle_positions()

def create_matrices(i):
    comparison_matrix = [] #list to store the comparison matrix

    firstindex = newsnapshotstarts[i-1]
    finalindex = newsnapshotstarts[i]

    #obtains coordinates of all particles in a particular snapshot 
    snapshotpositions = arraypositions[firstindex:finalindex]

    #ensures the snapshots are of double precision
    snapshotpositions = snapshotpositions.astype(dtype=np.float64)
    snapshotpositions = snapshotpositions%[xlength, ylength]  # deals with issue of being outside box due to numerical rounding of xlength, ylength
       
    #generates a KDTree of all particle coordinates for snapshot of interest.
    snapshotKD = KDTree(snapshotpositions, boxsize = [xlength, ylength]) # periodic
   # snapshotKD = KDTree(snapshotpositions) # non-periodic
    #loop to construct the comparison matrices for a particular snapshot
    for testparticle in testpositions:
        noindices = snapshotKD.query_ball_point(testparticle, r=cutoff, return_length=True)
        if noindices == 0:
            distances = np.array([])
        else:
            distances = np.asarray(snapshotKD.query(testparticle, distance_upper_bound=cutoff, k=int(noindices)))
        no_histogram = np.histogram(distances, bins = bins, range=(0,cutoff), density=False)[0]
        comparison_matrix.append(list(no_histogram))
    #cast to int16 data type to reduce memory usage.
    comparison_matrix = np.array(comparison_matrix, dtype=np.int16) #convert list to numpy array
    comparison_matrix = sparse.csr_matrix(comparison_matrix) #convert to numpy array to sparse CSR array
    #saves matrices for a given snapshot, with the file names indicating the corresponding snapshot number.
    sparse.save_npz('{}/Comparison_Matrix_{}'.format(newpath_matrices, i), comparison_matrix)
    return "Snapshot {}".format(i) 


# 3. CREATE MATRICES

#runs the create matrices definition 
for index in range(1, len(newsnapshotstarts)):
    result = create_matrices(index)
    print(result)

end = time.time()
time = end - start
print('Matrices script runtime: ',time)
