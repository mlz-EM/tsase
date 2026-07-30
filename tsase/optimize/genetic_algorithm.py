#!/usr/bin/env python

import numpy
import sys
import matplotlib
matplotlib.use("agg") # Change to "agg" to run on FRI
from pylab import *
import tsase
import ase
#from basin import *
from tsase.optimize.minima_basin_ga3 import Hopping
import random

np.set_printoptions(threshold=np.inf)

##initializes population array
n=5
sample = random.sample(range(100),n)
aGA = np.zeros(n,dtype = object)
for i in range(n):    
    aGA[i] = tsase.io.read_con('./lj38-clusters/'+str(sample[i])+'.con')

p = tsase.io.read_con('cluster_38.con')
lj = tsase.calculators.lj(cutoff=35.0)
p.center(100.0)
p.set_calculator(lj)

displacement = 0.3
 
opt = Hopping(  atoms = p, 
                     temperature = 8000 * ase.units.kB,  # temperture for metropolis MC run 
                     fmax = 0.01,                        # convergence criteria for local optimization
                     dr = displacement,                  # initial MC step size
                     adjust_cm = True,                   # you do not need to chang this flag
                     minenergy = -173.928 + .01,#-9.103852,              # stops BH when you are below this energy (REPLACE THIS VALUE WITH GLOBAL MINIMA FROM PAPER plus 0.01 eV) 
                     distribution = 'uniform',
                     #logfile = "idk",
                     optimizer_logfile = "ll_out", # uncomment this line if you do not want the information from each optimization to print 
                     #history_num = 5,
                     #use_geometry = True,

                    #GA parameters
                     move_type = 2,
                     atoms_GA = aGA,
                     pop_size = 5, #this doesnt do anything
                     mut_prob = 1.0,
                     nparents = 1,
                     mut_type = 'displacement',
                     mut_factor = 1.0,
                     fertility = .25,
                     fit_func = 'energy',
                     fit_accept_criteria = 'deterministic',
                     mut_n_atoms = 'all',
                     )
####################################################
# The function below, opt.run, runs Basin Hopping ###

opt.run(10,gens = 1)  # The number to the left is the MAXIMUM NUBMER OF MONTE CARLO STEPS

#f = open('out','a+')
#print('force calls',lj.force_calls,'pot energy',p.get_potential_energy()) # prints the number of force calls used in this script
#for i in [1,5,10,20,30,38]:
#    forcecalls = np.zeros(30)
#    poten = np.zeros(30)
#    name = 'BH-mut-%d' %i
#    lname = name +'-energy'
#    f = open(name,'w+')
#    l = open(lname,'w+')
#    for j in range(30):    
#        #BH code
#        p = tsase.io.read_con('cluster_38.con')
#        lj = tsase.calculators.lj(cutoff=35.0)
#        p.center(100.0)
#        p.set_calculator(lj)
#        displacement = 0.1
#        opt = Hopping(atoms = p, temperature = 8000 * ase.units.kB, fmax = 0.01,dr = displacement,adjust_cm = True, minenergy = -173.928 + .05, distribution = 'uniform',optimizer_logfile = "ll_out", move_type = 2,mut_n_atoms = i)
#        opt.run(5000)

#        forcecalls[j] = lj.force_calls
#        poten[j] = p.get_potential_energy()
#    failedruns = 0
#    for k in poten:
#        if k > -173.928 + .01:
#            failedruns += 1
#    f.write(name + '\navg force calls: ' + str(np.mean(forcecalls)) +'\nstdev force calls: ' + str(np.std(forcecalls)) + '\nfailed runs: ' + str(failedruns))
#    l.write(str(poten))
#    print(name + '\navg force calls: ' + str(np.mean(forcecalls)) +'\nstdev force calls: ' + str(np.std(forcecalls)) + '\nfailed runs: ' + str(failedruns) + ' \n' + str(poten))
        #ase.io.write('global_optimum_5',p,format = "vasp") # writes out file with the structure of the global minimum
    

sys.exit()













