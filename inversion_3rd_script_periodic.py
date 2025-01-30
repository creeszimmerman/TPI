# -*- coding: utf-8 -*-
"""
inversion. same code for periodic or nonperiodic. Nothing in this script is specific to 2D.
"""

#1. IMPORTING PACKAGES + PARAMETERS, CREATING FOLDERS

#1.1 Importing packages

import pandas as pd
import numpy as np
np.seterr(all='ignore')
import matplotlib.pyplot as plt
import time
from scipy import sparse
import os

import warnings
warnings.filterwarnings('ignore')

#start timing the script
start = time.time()

#graph formatting 
plt.rcParams.update({'font.size':14}) #font size change
plt.rcParams["figure.figsize"] = (8,6) #figure size change

#1.2 Importing parameters
from Parameters import newpath, nssfile, readfile, gr_file, midpoints_file, newpath_matrices, kT, bins, multiplier, small_number, optimisations

max_energy = -kT*np.log(small_number)

#1.3 Creating folders
inversepath = 'Inverse Path'
if not os.path.exists(inversepath):
    os.makedirs(inversepath)

#creating folders within inverse path
distributions_path = '{}/Distributions'.format(inversepath)
potentials_path = '{}/Potentials'.format(inversepath)
if not os.path.exists(distributions_path):
    os.makedirs(distributions_path)
if not os.path.exists(potentials_path):
    os.makedirs(potentials_path)

#2. DEFINITIONS
        
def import_RDF():
    read_midpoints = pd.read_hdf(midpoints_file)
    read_histogram = pd.read_hdf(gr_file)
    midpoints_array = read_midpoints.to_numpy()
    midpoints_array = np.transpose(midpoints_array) #transposes the matrix from a column to a row or vice versa
    histogram_array = read_histogram.to_numpy()
    histogram_array = np.transpose(histogram_array)
    return histogram_array[0], midpoints_array[0]

#average difference between target and generated RDF g(r)
def get_avg_difference(target, generated, experiment): 
    generated = np.nan_to_num(generated)
    subtracted = np.subtract(target, generated)
    subtracted_squared = subtracted**2
    avg_difference = np.sum(subtracted_squared)
    return avg_difference

def calculate_g_r_for_trial_v(index):
    comparison_matrix = all_comparison_matrices[index] #retrieve relevant comparison matrix
    insertion_energy = comparison_matrix @ column_v #matrix multiplication
    exponential = np.exp(-insertion_energy/kT) #matrix exponential
    bulk_average = np.mean(exponential) #mean of exponential column vector
    matrix_product = np.transpose(comparison_matrix)@exponential #matrix multiplication
    array_comparison = comparison_matrix.toarray() #conversion of sparse matrix to numpy array
    C_sums = array_comparison.sum(axis=0)
    local_averages = np.divide(np.transpose(matrix_product), C_sums)
    g_r_guess = np.divide(np.transpose(local_averages), bulk_average)
    g_r_guess = np.nan_to_num(g_r_guess)
    return g_r_guess

def update_trial_potential(avg_g_r_experiment, trial_potential, multiplier):
    trial_potential = np.nan_to_num(trial_potential, posinf=max_energy, neginf = max_energy) #replace NaN values with max_energy
    ratio_write = np.divide(target_RDF, avg_g_r_experiment)
    ratio_write = np.nan_to_num(ratio_write)
    ratio_write = np.where(ratio_write < small_number, small_number, ratio_write) #clean the ratio of target RDF to generated one
    log_ratio_write = np.log(ratio_write)
    trial_potential = trial_potential - multiplier*kT*log_ratio_write #corrector on trial potential
    return trial_potential

#3.INVERSION CALCULATION

#3.1 Calculating g(r)

target_RDF, midpoints = import_RDF() 
#replace any values less than small_number with small_number (to aid numerical stability)
target_RDF = np.where(target_RDF<small_number, small_number, target_RDF) #finding where it is less than small_number and replacing by this value

#3.2 Reading positions and newsnapshotstarts

#read entire simulation output into pandas dataframe.
readpositions = pd.read_hdf('{}/{}.hdf'.format(newpath, readfile)) #reading the positions file

#convert pandas dataframe to faster numpy n-d array
arraypositions = readpositions.to_numpy()

#determines where one snapshot ends and another begins
newsnapshotstarts = np.load('{}/{}'.format(newpath, nssfile))

#3.3 Loading matrices

#load in the comparison sparse matrices from the relevant files
loadtime_s = time.time() #probably take a long time so want to check how long it takes
all_comparison_matrices = []
iterations = len(newsnapshotstarts)

for i in range(1, iterations):
    comparison_matrix = sparse.load_npz('{}/Comparison_Matrix_{}.npz'.format(newpath_matrices,i))
    all_comparison_matrices.append(comparison_matrix)
loadtime_f = time.time()
loadtime = loadtime_f - loadtime_s
print("loadtime: ", loadtime)

#3.4 Inversion

times = np.array([])
times = np.append(times, loadtime)

avg_differences = np.array([])
chi_array = np.array([])
optimisation_array = np.array([])

time1 = time.time()

for experiment in range(0, optimisations+1): #number of optimisations done
    optimisation_t0 = time.time() #timing the optimisations
    total_experiment_g_r = [] #store the total g(r)
    if experiment == 0: #0: initial guess 
        initial_guess = -kT*np.log(target_RDF) #formula for trial potential using known g(r)
        trial_potential = initial_guess
        trial_potential = np.nan_to_num(trial_potential, posinf=max_energy, neginf = max_energy)
        #plot graph + save it to potentials folder
        plt.plot(midpoints, trial_potential, label='Generated u(r)', color='cornflowerblue', markerfacecolor='none', marker='D', alpha=0.8)
        plt.locator_params(axis='x', nbins=5)
        plt.xlabel('r')
        plt.ylabel('u(r)')
        plt.xlim(0.5, max(midpoints+0.005))
        plt.ylim(-4, 5)
        plt.legend()
        plt.title('Initial Guess', y=1.2)  
        plt.savefig('{}/Initial_Guess'.format(potentials_path), dpi=600, bbox_inches = 'tight', pad_inches = 0.4)
        plt.show()
        plt.clf() #clear current figure
        plt.cla() #clear current axes 
        plt.close()   
        potential_dataframe = pd.DataFrame(np.transpose(trial_potential)) 
        potential_dataframe.to_hdf('{}/Potentials_Backup.hdf'.format(inversepath), key ='potential_dataframe', mode='w', append = True) #saves the initial guess potential inverse folder
    
    column_v = np.transpose(trial_potential) #ensures correct dimensionality for subsequent linear algebra

    #single core.
    starts = [index for index in range(1, len(all_comparison_matrices)+1)] #list of snapshot numbers 
    total_snapshots = len(starts)
    
    for i in range(0, total_snapshots):
        g_r = calculate_g_r_for_trial_v(i)
        total_experiment_g_r.append(g_r)
    sum_g_r_snapshots = np.zeros((bins)) #zero array to store the snapshot sums
    #element-wise addition of g_r from each snapshot to calculate a total
    for g_r in total_experiment_g_r:
        sum_g_r_snapshots = np.add(sum_g_r_snapshots, g_r)
    #calculation of average g(r) and cleaning 
    avg_g_r_experiment = sum_g_r_snapshots/total_snapshots
    avg_g_r_experiment = np.nan_to_num(avg_g_r_experiment)
    avg_g_r_experiment = np.where(avg_g_r_experiment<small_number, small_number, avg_g_r_experiment)

   
    if experiment == 0: #guess values in first column 
        #create new file to hold all g(r) values
        g_r_dataframe = pd.DataFrame(np.transpose(avg_g_r_experiment))
        g_r_dataframe.to_hdf('{}/RDFs_Backup.hdf'.format(inversepath), key = 'g_r_dataframe', mode='w')
    else:
        #append to g(r) values file with guess of g(r)
        read_rdf_hdf = pd.read_hdf('{}/RDFs_Backup.hdf'.format(inversepath)) 
        g_r_dataframe = pd.DataFrame(np.transpose(avg_g_r_experiment))
        read_rdf_hdf['{}'.format(experiment)] = g_r_dataframe
        read_rdf_hdf.to_hdf('{}/RDFs_Backup.hdf'.format(inversepath), key = 'g_r_dataframe', mode = 'a')
    #calculation of difference between target and generated potentials and RDFs
    avg_difference = get_avg_difference(target_RDF, avg_g_r_experiment, experiment)
    avg_differences = np.append(avg_differences, avg_difference)

    #only print the difference for every 5th optimisation
    if experiment % 5 == 0:
        print("Optimisation {}, Sum of Differences: ".format(experiment), avg_difference)

    #update trial potential, then append the new potential to file containing potential for each optimisation
    trial_potential = update_trial_potential(avg_g_r_experiment, trial_potential, multiplier) 
    read_potential_hdf = pd.read_hdf('{}/Potentials_Backup.hdf'.format(inversepath), key = 'potential_dataframe')
    potential_dataframe = pd.DataFrame(np.transpose(trial_potential))
    read_potential_hdf['{}'.format(experiment)] = potential_dataframe
    read_potential_hdf.to_hdf('{}/Potentials_Backup.hdf'.format(inversepath), key='potential_dataframe', mode = 'a')
    
    #comparing graph differences 

    chi = (avg_g_r_experiment[0] - target_RDF[0])**2
    for i in range (1, len(avg_g_r_experiment)):
        chi = chi + (avg_g_r_experiment[i] - target_RDF[i])**2
    chi_array = np.append(chi_array, chi)
    
    #for first ten optimisations and for every 10th optimisation after, output and save comparison graphs
    if experiment < 10 or experiment%10 == 0:
        plt.plot(midpoints, avg_g_r_experiment, label='Generated g(r)', color='cornflowerblue', markerfacecolor='none', marker='D', alpha=0.8)
        plt.plot(midpoints, target_RDF, label = 'Target g(r)', color='black', alpha=1, linewidth=1.5)
        plt.xlabel('r')
        plt.ylabel('g(r)')
        plt.xlim(0,max(midpoints)+0.05)
        plt.legend()
        plt.text(2.75, 2,f"\u03C7²= {chi}")
        plt.title('Optimisation {}'.format(experiment), y=1.2)
        plt.savefig('{}/DC_{}'.format(distributions_path, experiment), dpi=600) # savefig does not work if path length too long
        plt.show()
        plt.clf()
        plt.cla()
        plt.close()
       
        plt.plot(midpoints, trial_potential, label='Generated u(r)', color='cornflowerblue', markerfacecolor='none', marker='D', alpha=0.8)
        plt.locator_params(axis='x', nbins=5)
        plt.xlabel('r')
        plt.ylabel('u(r)')
        plt.xlim(0.5, max(midpoints+0.005))
        plt.ylim(-4, 5)
        plt.legend()
        plt.title('Optimisation {}'.format(experiment), y=1.2)
        plt.savefig('{}/PotentialComparison_{}'.format(potentials_path, experiment), dpi=600)
        plt.show()
        plt.clf()
        plt.cla()
        plt.close()
        
    optimisation_array = np.append(optimisation_array, experiment)
    
    optimisation_t1 = time.time()

    times = np.append(times, optimisation_t1-optimisation_t0) #time is how long it takes to load the matrices in row 0), and the rest up to 21 is the difference in time between the optimisations (final time - initial time) which is appended to the file
    
time2 = time.time()
print("Insertion time for all experiments: ", time2-time1)
print("Average time per optimisation: ", np.average(times))


averages_dataframe = pd.DataFrame(np.transpose(avg_differences))
averages_dataframe.to_hdf('{}/Averages.hdf'.format(inversepath), key='averages_dataframe')

#output the final generated pair potential and midpoints to a separate file (for easy interpolation if we want to use it in simulation)
generated_dataframe = pd.DataFrame({'r':midpoints, 'generated':trial_potential})
generated_dataframe.to_hdf('{}/GeneratedPotential.hdf'.format(inversepath), key='generated_dataframe')

#4. PLOTTING CONVERGENCE OF G(R) THROUGHOUT OPTIMISATIONS
plt.loglog(optimisation_array, chi_array)
plt.xlabel('Optimisation Number')
plt.ylabel('\u03C7²')
plt.title('\u03C7² vs Optimisation Number')
plt.savefig('{}/\u03C7² vs Optimisation Number'.format(inversepath), bbox_inches = 'tight', pad_inches = 0.4)
plt.show()

end = time.time()
print("Runtime: ", end-start)
