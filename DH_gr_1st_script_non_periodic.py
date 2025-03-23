# -*- coding: utf-8 -*-
"""
distance-histogram script. non-periodic. 2D.
"""

#1. IMPORTING PACKAGES, PARAMETERS, POSITIONS COORDINATES

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from scipy.spatial import cKDTree as KDTree
from scipy import constants

from Parameters import newpath, nssfile, readfile, gr_file, midpoints_file, bins, cutoff, bin_width, small_number

#1. FILE PATHS
positionspath = '{}/{}.hdf'.format(newpath,readfile) # where positions are stored
newsnapshotstarts = np.load('{}/{}'.format(newpath, nssfile))
xlength = np.load('{}/xlength.npy'.format(newpath))
ylength = np.load('{}/ylength.npy'.format(newpath))
V = xlength*ylength


#can play around with the way that images are plotted
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

        #generates a KDTree of all particle coordinates for snapshot of interest.
        snapshotKD = KDTree(snapshotpositions)

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

    #obtains the number of particles for a particular snapshot
    N = len(snapshotpositions)

    #generates a KDTree of all particle coordinates for snapshot of interest.
    snapshotKD = KDTree(snapshotpositions)

    for selectedparticle in snapshotpositions:

        #returns both distances < cutoff and indices of the corresponding particles

        noindices = snapshotKD.query_ball_point(selectedparticle, r=cutoff, return_length=True)
        distances = np.asarray(snapshotKD.query(selectedparticle, k=noindices, distance_upper_bound=cutoff))
        #these distances must be used to calculate the minimum distance.
        cleaned_distances = distances[0][distances[0] != 0]

        #calls function to generate histogram from our snapshot distances
        histogram = pair_distribution_function(cleaned_distances)
        
        #calculates a normalised g_r for a particular reference particle
        g_r_particle = np.divide(histogram[0],2*constants.pi*midpoints*bin_width*(N-1)/V)

        #adds the normalised g_r for particle to cumulative g(r) for snapshot
        sum_g_r_particles = np.add(sum_g_r_particles, g_r_particle)
    
    #calculates average g(r) across all N particles in snapshot
    average_g_r_particle_in_snap = np.divide(sum_g_r_particles, N)

    #return average_g_r_particle_in_snap
    return average_g_r_particle_in_snap, N

#generating ideal gas particles - can overlap as take no volume
def ideal_gas_positions(N,xlength,ylength):
    idealgas_positions = np.random.rand(N,2)*[xlength, ylength]
    idealgas_positionsframe = pd.DataFrame(np.array(idealgas_positions), columns=['x-coordinates', 'y-coordinates'])
    idealgas_positionsframe.to_hdf('idealgas_positions.hdf', format='table', key='idealgas_positionsframe', append = 'true') #saving the positions
    return idealgas_positions

#ideal gas histogram
def calculate_idealgas_histogram(i):

    #defines an array of zeros that is used to keep track of the cumulative g(r) for ALL PARTICLES in a particular snapshot.
    sum_g_r_particles = np.zeros(bins)

    #records the index of the first particle in the snapshot and the last particle in the snapshot
    firstindex = newsnapshotstarts[i-1]
    finalindex = newsnapshotstarts[i]

    #obtains coordinates of all particles in a particular snapshot by taking slice of array containing data every snapshot.
    snapshotpositions = idealgas_arraypositions[firstindex:finalindex]
    snapshotpositions = snapshotpositions.astype(dtype=np.float64)

    #obtains the number of particles for a particular snapshot
    N = len(snapshotpositions)

    #generates a KDTree of all particle coordinates for snapshot of interest.
    snapshotKD = KDTree(snapshotpositions)

    for selectedparticle in snapshotpositions:

        #returns both distances < cutoff and indices of the corresponding particles

        noindices = snapshotKD.query_ball_point(selectedparticle, r=cutoff, return_length=True)
        distances = np.asarray(snapshotKD.query(selectedparticle, k=noindices, distance_upper_bound=cutoff))
        #these distances must be used to calculate the minimum distance.
        cleaned_distances = distances[0][distances[0] != 0]

        #calls function to generate histogram from our snapshot distances
        histogram = pair_distribution_function(cleaned_distances)
        
        #calculates a normalised g_r for a particular reference particle
        g_r_particle = np.divide(histogram[0],2*constants.pi*midpoints*bin_width*(N-1)/V)

        #adds the normalised g_r for particle to cumulative g(r) for snapshot
        sum_g_r_particles = np.add(sum_g_r_particles, g_r_particle)
    
    #calculates average g(r) across all N particles in snapshot
    average_idealgas_g_r_particle_in_snap = np.divide(sum_g_r_particles, N)

    return average_idealgas_g_r_particle_in_snap, N


#3 DISTANCE HISTOGRAM METHOD

#3.1 Unadjusted Calculation

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
sum_g_r_snapshots = np.zeros((bins)) #array of zeros that is used to store the sum of the average g(r) values for ALL snapshots
N_per_snapshot = np.array([]) #number of particles per snapshot
all_g_r = [] #g(r) values for all snpashots


#calculating g(r) values and number of particles per snapshot
starts = [index for index in range(1, len(newsnapshotstarts))]
for i in range(1, len(starts)+1):
    histogram, N = calculate_histogram(i)
    all_g_r.append(histogram)
    N_per_snapshot = np.append(N_per_snapshot, N)

total_snapshots = len(starts)
sum_g_r_snapshots = np.zeros((bins))
for g_r in all_g_r:
    sum_g_r_snapshots = np.add(sum_g_r_snapshots, g_r)

average_DH_g_r_unadjusted = sum_g_r_snapshots/total_snapshots

unadjusted_g_r_dataframe = pd.DataFrame(average_DH_g_r_unadjusted)


# plot graph of unadjusted g(r)
plt.plot(midpoints, average_DH_g_r_unadjusted, 'g')
plt.xlabel('r')
plt.ylabel('g(r)')
plt.title("Distance Histogram g(r) - unadjusted non-periodic BCs", y=1.2)
plt.savefig('DH g(r) - unadjusted non-periodic BCs.png', transparent=True)
plt.show()

#3.2 Ideal gas calculation

totalN = 0

#making ideal gas snapshots with same number of particles as experiment
for i in range (0,len(N_per_snapshot)):
    N = int(N_per_snapshot[i])
    totalN = totalN + N
    idealgas_positions = ideal_gas_positions(N,xlength, ylength)

#reads entire simulation output hdf file into a pandas dataframe
read_idealgas_positions = pd.read_hdf('idealgas_positions.hdf')

#convert pandas dataframe to faster numpy n-d array
idealgas_arraypositions = read_idealgas_positions.to_numpy()
idealgas_arraypositions = idealgas_arraypositions.astype(dtype=np.float64)

all_idealgas_g_r = []

#calculating g(r) for all ideal gas snapshots
starts = [index for index in range(1, len(newsnapshotstarts))]
for i in range(1, len(starts)+1):
    histogram, N = calculate_idealgas_histogram(i)
    all_idealgas_g_r.append(histogram)


sum_g_r_snapshots_idealgas = np.zeros((bins))
for g_r in all_idealgas_g_r:
    sum_g_r_snapshots_idealgas = np.add(sum_g_r_snapshots_idealgas, g_r)
average_DH_g_r_IG = sum_g_r_snapshots_idealgas/total_snapshots
average_DH_g_r_IG = np.where(average_DH_g_r_IG == 0, small_number, average_DH_g_r_IG)

# plot graph of IG g(r)
plt.plot(midpoints, average_DH_g_r_IG, 'b')
plt.xlabel('r')
plt.ylabel('g(r)')
plt.title("Distance Histogram g(r) - ideal gas", y=1.2)
plt.savefig('DH g(r) - ideal gas.png', transparent=True, bbox_inches = 'tight', pad_inches = 0.4)
plt.show()


#3.3 Non-periodic boundary conditions calculation

nonperiodic_BC_DH_g_r = average_DH_g_r_unadjusted / average_DH_g_r_IG
nonperiodic_BC_DH_g_r_dataframe = pd.DataFrame(nonperiodic_BC_DH_g_r)
nonperiodic_BC_DH_g_r_dataframe.to_hdf(gr_file, key = 'nonperiodic_BC_DH_g_r', mode='w')

# plot graph of adjusted g(r)
plt.plot(midpoints, nonperiodic_BC_DH_g_r, 'm')
plt.xlabel('r')
plt.ylabel('g(r)')
plt.title(" Non-periodic boundary conditions DH g(r)", y=1.2)
plt.savefig('Nonperiodic_BC_DH_g(r).png', transparent=True, bbox_inches = 'tight', pad_inches = 0.4)
plt.show()

#calculating the time
hist_end = time.time()
hist_time = hist_end - hist_start

print("Non-periodic Histogram Runtime: ", hist_time)
