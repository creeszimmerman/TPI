# -*- coding: utf-8 -*-
"""
distance-histogram script. periodic. 2D.
"""

#1. IMPORTING PACKAGES, PARAMETERS, POSITIONS COORDINATES

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from scipy.spatial import cKDTree as KDTree
from scipy import constants


from Parameters import newpath, nssfile, readfile, midpoints_file, bins, cutoff, bin_width
    
#1. FILE PATHS
positionspath = '{}/{}.hdf'.format(newpath, readfile) # where positions are stored
newsnapshotstarts = np.load('{}/{}'.format(newpath, nssfile))
xlength = np.load('{}/xlength.npy'.format(newpath))
ylength = np.load('{}/ylength.npy'.format(newpath))
V = xlength*ylength

    
#graph formatting
mpl.rc('figure',  figsize=(10, 5))
mpl.rc('image', cmap='gray')


#2. DEFINITIONS

#returns a histogram of snapshot distances starting at 0 and ending at cutoff. Has no. bins = to bins variable.
def pair_distribution_function(distances): 
    return np.histogram(distances, bins = bins, range=(0,cutoff), density=False)

#parallel histogram generation
def generate_midpoints():
    for i in range(1, 2):

        #records the index of the first particle in the snapshot and the last particle in the snapshot
        firstindex = newsnapshotstarts[i-1]
        finalindex = newsnapshotstarts[i]

        #obtains coordinates of all particles in a particular snapshot by taking slice of array containing data every snapshot.
        snapshotpositions = arraypositions[firstindex:finalindex]
        snapshotpositions = snapshotpositions.astype(dtype=np.float64)
        snapshotpositions = snapshotpositions%[xlength, ylength]  # deals with issue of being outside box due to numerical rounding of xlength, ylength
        
        #generates a KDTree of all particle coordinates for snapshot of interest.
        snapshotKD = KDTree(snapshotpositions, boxsize = [xlength,ylength])

        for selectedparticle in snapshotpositions[[0]]: # only need to use one particle, as just want to generate midpoints
            #returns both distances < cutoff and indices of the corresponding particles
            noindices = snapshotKD.query_ball_point(selectedparticle, r=cutoff, return_length=True)
            distances = np.asarray(snapshotKD.query(selectedparticle, k=noindices, distance_upper_bound=cutoff))
            #these distances must be used to calculate the minimum distance.
            cleaned_distances = distances[0][distances[0] != 0]

            #calls function to generate histogram from our snapshot distances
            histogram = pair_distribution_function(cleaned_distances)

            midpoints = np.array([])
            histogrambins = histogram[1]
            for index in range(0,len(histogrambins)-1):
                midpoints = np.append(midpoints, (histogrambins[index] + histogrambins[index+1])/2)

        return midpoints
        
#calculating the average g(r) in a snapshot
def calculate_histogram(i):

    #defines an array of zeros that is used to keep track of the cumulative g(r) for ALL PARTICLES in a particular snapshot.
    sum_g_r_particles = np.zeros(bins)

    #records the index of the first particle in the snapshot and the last particle in the snapshot
    firstindex = newsnapshotstarts[i-1]
    finalindex = newsnapshotstarts[i]

    #obtains coordinates of all particles in a particular snapshot by taking slice of array containing data every snapshot.
    snapshotpositions = arraypositions[firstindex:finalindex]
    snapshotpositions = snapshotpositions.astype(dtype=np.float64)
    snapshotpositions = snapshotpositions%[xlength, ylength]  # deals with issue of being outside box due to numerical rounding of xlength, ylength
    
    #obtains the number of particles for a particular snapshot
    N = len(snapshotpositions)

    #generates a KDTree of all particle coordinates for snapshot of interest.
    snapshotKD = KDTree(snapshotpositions, boxsize = [xlength,ylength])


    for selectedparticle in snapshotpositions:

        #returns both distances < cutoff and indices of the corresponding particles

        noindices = snapshotKD.query_ball_point(selectedparticle, r=cutoff, return_length=True)
        distances = np.asarray(snapshotKD.query(selectedparticle, k=noindices, distance_upper_bound=cutoff))
        #these distances must be used to calculate the minimum distance.
        cleaned_distances = distances[0][distances[0] != 0]

        #calls function to generate histogram from our snapshot distances
        histogram = pair_distribution_function(cleaned_distances)
        
        #calculates a normalised g_r for a particular reference particle
        g_r_particle = np.divide(histogram[0],2*constants.pi*midpoints*bin_width*(N-1)/V) # 2D equation

        #adds the normalised g_r for particle to cumulative g(r) for snapshot
        sum_g_r_particles = np.add(sum_g_r_particles, g_r_particle)
    
    #calculates average g(r) across all N particles in snapshot
    average_g_r_particle_in_snap = np.divide(sum_g_r_particles, N)

    #return average_g_r_particle_in_snap
    return average_g_r_particle_in_snap, N


#3 DISTANCE HISTOGRAM METHOD

#3.1 Periodic Calculation

hist_start = time.time()

#reads entire simulation output hdf file into a pandas dataframe
readpositions = pd.read_hdf(positionspath)

#convert pandas dataframe to faster numpy n-d array
arraypositions = readpositions.to_numpy()

#calculating midpoints and saving as an hdf file
midpoints = generate_midpoints()
midpoints_dataframe = pd.DataFrame(midpoints)
midpoints_dataframe.to_hdf(midpoints_file, key='midpoints')


#setting up arrays
sum_g_r_snapshots = np.zeros((bins))#array of zeros that is used to store the sum of the average g(r) values for ALL snapshots
N_per_snapshot = np.array([]) #number of particles per snapshot
all_g_r = [] #g(r) values for all snapshots

#calculating g(r) values and number of particles per snapshot
starts = [index for index in range(1, len(newsnapshotstarts))]

for i in range(1, len(starts)+1):
    histogram, N = calculate_histogram(i)
    all_g_r.append(histogram)
    N_per_snapshot = np.append(N_per_snapshot, N)

total_snapshots = len(starts)
for g_r in all_g_r:
    sum_g_r_snapshots = np.add(sum_g_r_snapshots, g_r)

average_DH_g_r_periodic = sum_g_r_snapshots/total_snapshots

periodic_g_r_dataframe = pd.DataFrame(average_DH_g_r_periodic)
periodic_g_r_dataframe.to_hdf('periodic_BC_gr.hdf', key = 'periodic_g_r_dataframe', mode='w') 

# plot graph
plt.plot(midpoints,average_DH_g_r_periodic, 'm')
plt.xlabel('r')
plt.ylabel('g(r)')
plt.title("Periodic boundary conditions DH g(r)", y=1.2)
plt.savefig('Periodic_BC_DH_g(r).png', transparent=True, bbox_inches = 'tight', pad_inches = 0.4)
plt.show()

#calculating the time
hist_end = time.time()
hist_time = hist_end - hist_start

print("Periodic Histogram Runtime: ", hist_time)
