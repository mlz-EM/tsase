import numpy as np
import os
from ase import io, units
from ase.optimize.optimize import Dynamics
from ase.optimize.fire import FIRE
from ase.optimize import QuasiNewton
from tsase.optimize.sdlbfgs import SDLBFGS
from ase.units import kB
from ase.parallel import world
from ase.io.trajectory import PickleTrajectory
from ase.md import VelocityVerlet
from ase.md import MDLogger
import tsase
import sys
from tsase.optimize import atoms_operator as geometry
from distutils.dist import Distribution
import queue 

##GZG don't know how much of this we'll need, got from https://wiki.fysik.dtu.dk/ase/dev/tutorials/ga/ga_optimize.html
import random # probably should us np, but this was easier for line 609
# for initialize function
from ase.ga.data import PrepareDB
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import closest_distances_generator
from ase.ga.utilities import get_all_atom_types
#for main function
from ase.ga.data import DataConnection
from ase.ga.population import Population
from ase.ga.standard_comparators import InteratomicDistanceComparator
from ase.ga.cutandsplicepairing import CutAndSplicePairing
from ase.ga.utilities import closest_distances_generator
from ase.ga.utilities import get_all_atom_types
from ase.ga.offspring_creator import OperationSelector
from ase.ga.standardmutations import MirrorMutation
from ase.ga.standardmutations import RattleMutation
from ase.ga.standardmutations import PermutationMutation

class Hopping(Dynamics):
    """Basin hopping algorithm.

    After Wales and Doye, J. Phys. Chem. A, vol 101 (1997) 5111-5116

    and 

    David J. Wales and Harold A. Scheraga, Science, Vol. 285, 1368 (1999)
    """

    def __init__(self,

                 # General GO parameters
                 atoms, # ASE atoms object defining the PES
                 temperature = 500, # K, initial temperature
                 optimizer = 'FIRE', # local optimizer
                 fmax = 0.01, # magnitude of the L2 norm used as the convergence criteria for local optimization
                 adjust_cm = True, # fix the center of mass (True or False)
                 mss = 0.1, # maximum step size for the local optimizer
                 minenergy = None, # the GO algorithm stops when a configuration is found with a lower potential energy than this value
                 pushapart = 0.1, # push atoms apart until all atoms are no closer than this distance
                 keep_minima_arrays = True, # create global_minima and local_minima arrays of length maximum number of Monte Carlo steps (True or False)
                 minima_threshold = 4,  # round potential energies to how many decimal places

                 calculator = None, # the potential calculator (i.e. LJ); not required (used for force call break downs)
                 seed = None, # the random seed for RNG
                 keep_movie = False, # keep a trajectory of accepted moves over time
                 check_bonds = True, # check that no bonds broke after every trial move iteration
                 max_localops = True, # quit run when total # of local optimixations > steps defined in def run()

                 # Logging parameters
                 logfile = None,
                 trajectory = None,
                 optimizer_logfile = '-',
                 #local_minima_trajectory = 'local_minima.con',
                 local_minima_trajectory = None,

                 # Selecting move type
##GZG changed 'move_type = True' to 'move_type = 1', this shouldn't mess up BH and MH code and will let us make 'move_type = 2' map to GA
                 move_type = 1, # True = basin hopping trial move; False = minima hopping
                 distribution = 'uniform', # The distribution to use in trial move. Make "molecular_dynamics" the distribution for MH trial move

                 # Random move parameters
                 dr = 0.45, # maximum displacement in each degree of freedom for Monte Carlo trial moves
                 adjust_step_size = 10, # adjust dr after this many Monte Carlo steps (Default: None; does not adjust dr)
                 target_ratio = 0.5, # specified ratio of Monte Carlo steps
                 adjust_fraction = 0.05, # fraction by which to adjust dr by in order to meet target_ratio
                 significant_structure = True,  # displace from minimum at each move (True or False)

                 # Dynamic move parameters (Molecular Dynamics)
                 timestep = 0.1, # fs, molecular dynamics time step
                 mdmin = 2, # number of minima to pass in MD before stopping
                 dimer_method = True, # uses an iterative dimer method before molecular dynamics (True or False)
                 dimer_a = 0.001, # scalar for forces in optimization
                 dimer_d = 0.01, # distance between two images in the dimer
                 dimer_steps = 14, # number of dimer iterations

                 # Occasional jumping parameters
                 jump_distribution = 'uniform', # The distribution to use in OJ move. Same options as distribution flag
                 jumpmax = None, # number of previously rejected MC steps before taking an OJ move; None = no OJ move
                 jmp = 7, # number of consecutive accepted moves taken in OJ
                 global_jump = None, # number of times to visit the same PE before we take an OJ; None = no global jump
                 global_reset = False, # True = reset all of the history counts after a jump (basically delete the history and start fresh)

                 # Select acceptance criteria
##GZG changed 'true' to '1', same as move_type
                 acceptance_criteria = 1, # BH = True; MH = False

                 # BH acceptance parameters
                 accept_temp = 8000, # K; separate temperature to use for BH acceptance (None: use temperature parameter instead)
                 adjust_temp = False, # dynamically adjust the temperature in BH acceptance (True or False)
                 history_weight = 0.0, # the weight factor of BH history >= 0 (0.0: no history comparison in BH acceptance)
                 adjust_step_size_in_hw = False,
                 history_num = 0, # limit of previously accepted minima to keep track of for BH history (set to 0 to keep track of all minima)
                 a = 1.0,
                 my_func = lambda a: a,

                 # MH acceptance criteria
                 beta1 = 1.04, # temperature adjustment parameter
                 beta2 = 1.04, # temperature adjustment parameter
                 beta3 = 1.0/1.04, # temperature adjustment parameter
                 Ediff0 = 0.5,  # eV, initial energy acceptance threshold
                 alpha1 = 0.98,  # energy threshold adjustment parameter
                 alpha2 = 1./0.98,  # energy threshold adjustment parameter
                 minimaHopping_history = True, # use history in MH trial move (True or False)

                 # Geometry comparison parameters
                 use_geometry = False, # True = compare geometry of systems when they have the same PE when determining if they are the same atoms configuration
                 eps_r = 0.1, # positional difference to consider atoms in the same location
                 use_get_mapping = True, # from atoms_operator.py use get_mapping if true or rot_match if false to compare geometry
                 neighbor_cutoff = 2.0, # parameter for get_mapping only
##GZG                 
                 # GA parameters
                 mut_prob = 1.0, # probability of mutation of cluster. This is fine for nparents = 1, but should be changed for nparents > 1
                 nparents = 1, # number of parents, 1 = asexual, will need to implement crossover before having more than 1
                 mut_type = 'displacement', # how the cluster is changed in mutation, random is effectively the same as BH random displacement, will add other operators from Rondina et al 2013
                 mut_factor = 1.0, # scales the severity of mutations, not sure how this will work with operators other than 'random'
                 fertility = .25, # portion of the population that reproduces on average
                 fit_func = 'energy', # how fitness is scored, we'll just use energy for now but measuring 'uniqueness' of a cluster may be useful
                 fit_accept_criteria = 'exponential', # how clusters will be accepted according to their fitness score, will also include 'linear' and 'hyperbolic' from R.L. Johnston 2003
                 gamma1 = 1.05, # if known configuration is landed on multiply mut_factor or mut_prob by this amount
                 gamma2 = 1.05, # if current cluster is more fit than best known cluster multiply mut_factor or mut_prob by this amount (I may have that backwards)
                 gamma3 = .95, # opposite of gamma2
                 mut_n_atoms = 'all' # int number of atoms displaced in a mutation or 'all'

                 ):
        
        Dynamics.__init__(self, atoms, logfile, trajectory)
        self.temperature = temperature
        self.optimizer = optimizer
        self.fmax = fmax
        self.dr = dr
        if adjust_cm:
            self.cm = atoms.get_center_of_mass()
        else:
            self.cm = None
        self.mss = mss ##GZG added this
        self.optimizer_logfile = optimizer_logfile
        self.lm_trajectory = local_minima_trajectory
        if isinstance(local_minima_trajectory, str):
            tsase.io.write_con(self.lm_trajectory,atoms,w='w')
        self.minenergy = minenergy
        self.move_type = move_type
        self.distribution = distribution
        self.adjust_step = adjust_step_size
        self.adjust_step_in_hw = adjust_step_size_in_hw
        self.target_ratio = target_ratio
        self.adjust_fraction = adjust_fraction
        self.significant_structure = significant_structure
        self.pushapart = pushapart
        self.jumpmax = jumpmax
        self.jmp = jmp
        self.jump_distribution = jump_distribution
        self.global_jump = global_jump
        self.num_gj = 0
        self.num_oj = 0
        self.jump_step = []
        self.global_reset = global_reset
        self.dimer_method = dimer_method
        self.dimer_a = dimer_a
        self.dimer_d = dimer_d
        self.dimer_steps = dimer_steps
        self.timestep = timestep
        self.mdmin = mdmin
        self.my_func = my_func
        self.a = a
        self.w = history_weight
        self.history_num = history_num
        self.adjust_temp = adjust_temp
        self.accept_temp = accept_temp
        self.accept_criteria = acceptance_criteria
        self.mh_history = minimaHopping_history
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.Ediff = Ediff0
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.minima_threshold = minima_threshold
        self.use_geo = use_geometry
        self.eps_r = eps_r
        self.use_get_mapping = use_get_mapping
        self.neighbor_cutoff = neighbor_cutoff
        self.keep_minima_arrays = keep_minima_arrays
        self.global_minima = [] # an array of the current global minimum for every MC step
        self.local_minima = [] # an array of the current local minimum for every MC step
        self.allTemps = []
        self.saveEdiff = []
        self.trial_loop = []
        self.num_localops = 0
        self.num_geo_compare = []
        self.minima = {}
##GZG        
        self.mut_prob = mut_prob
        self.nparents = nparents
        self.mut_type = mut_type
        self.mut_factor = mut_factor
        self.fertility = fertility
        self.fit_func = fit_func 
        #self.fit_accept = fit_accept
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.mut_n_atoms = mut_n_atoms

        # list with fixed size that will store history_num previous minima
        self.temp_minima = [0] * self.history_num
        self.temp_posIndex = [0] * self.history_num

     	# for geometry comparison
        self.positionsMatrix = []
        # count of unique geometries we have accepted so far
        self.numPositions = 0
        # dictionary for geometry comparison
        # keys = approximate potential energy
        # values = list of indexes in positionsMatrix with the same PE
        self.geometries = {}
        # dictionary with keys = index in positionsMatrix
        # values = # accepted visits
        self.geo_history = {}
        self.current_index = None
        self.last_index = None
        # number of Monte Carlo moves that have been accepted in the run
        self.num_accepted_moves = 0.0
        self.num_rejected_moves = 0.0
        self.calculator = calculator
        self.fc_trial_move = 0.0
        self.fc_dimer = 0.0
        self.fc_md = 0.0
        self.fc_local_ops = 0.0
        self.fc_jump = 0.0
        self.seed = seed
        self.keep_movie = keep_movie
        self.check_bonds = check_bonds
        self.max_localops = max_localops
        self.initialize()

    def initialize(self):
        self.positions = 0.0 * self.atoms.get_positions()
        self.Emin = self.get_energy(self.atoms.get_positions()) or 1.e32 
        self.rmin = self.atoms.get_positions()
        self.positions = self.atoms.get_positions()
        self.call_observers()
        self.log(-1, self.Emin, self.Emin,self.dr)
        
    def log(self, step, En, Emin,dr):
        if self.logfile is None:
            return
        name = self.__class__.__name__
        self.logfile.write('%s: step %d, energy %15.6f, emin %15.6f'
                           % (name, step, En, Emin))
	#if not self.molecular_dynamics:
        if self.distribution != 'molecular_dynamics':
            self.logfile.write(', dr %15.6f'
                               % (dr))
##GZG this seems to be broken, maybe not now?
        self.logfile.write(', Temperature %12.4f' % (self.temperature))
        if not self.accept_criteria:
            self.logfile.write(', Ediff %12.4f\n' % (self.Ediff))
        else:
            self.logfile.write(', w %12.4f\n' % (self.w))
       	                    #% (self.my_func(self.a)))
                            #% (self.w))
        self.logfile.flush()

    def check_connected(self):
        '''
        Use BFS to determine if all atoms in p are connected to one another
        '''
        # get the neighbor list of every atom
        intersect = geometry.sweep_and_prune(self.atoms, self.neighbor_cutoff)
        visited = [0]*len(self.atoms.get_positions())
        q = queue.Queue()
        q.put(0)
        # BFS to find connected atoms
        while not q.empty():
            curr_atom = q.get()
            visited[curr_atom] = 1
            for neighbor in intersect[curr_atom]:
                if visited[neighbor] == 0:
                    q.put(neighbor)
        if 0 in visited:
            return False
        return True

    def find_energy_match(self, En, Eo):
        """ determines if En is the same PE as any previously visited minima."""
        approxEn = round(En,self.minima_threshold)
        approxEo = round(Eo,self.minima_threshold)
        try:
            countEo = self.minima[approxEo]
        except KeyError:
            # approxEo isn't in the dict of pervious local min
            # this should only be a possiblity at step 0 of the run
            countEo = 0
        # check if approxEn is a previously visited minimum
        if approxEn in self.minima:
            countEn = self.minima[approxEn]
            if approxEn == approxEo:
                return 1, countEn, countEo
            return 2, countEn, countEo
        # approxEn is a new local minimum
        else:
            return None, 0, countEo

    def update_minima(self, En, Eo):
        """Update the dictionary of local minima to include this new location
           and return True if En is a new local minima."""
        approxEn = round(En,self.minima_threshold)
        approxEo = round(Eo,self.minima_threshold)
        if approxEn in self.minima:
            self.minima[approxEn] += 1
            return False, approxEn, approxEo
        else:
            self.minima[approxEn] = 1
            return True, approxEn, approxEo

    def update_geometries(self, En):
        '''Update the dictionaries to include this newly accepted geometry
            and return the index in self.positionsMatrix[] that this geometry is stored at'''
        approxEn = round(En, self.minima_threshold)
        index = -1
        if approxEn in self.geometries:
            if self.current_index is not None:
                self.geo_history[self.current_index] += 1
                self.last_index = self.current_index
            else:
                self.geo_history[self.numPositions] = 1
                self.last_index = self.numPositions
                self.geometries[approxEn].append(self.numPositions)
                self.positionsMatrix[self.numPositions] = self.atoms.get_positions()
                self.numPositions += 1
        else:
            self.geometries[approxEn] = [self.numPositions]
            self.geo_history[self.numPositions] = 1
            self.last_index = self.numPositions
            self.positionsMatrix[self.numPositions] = self.atoms.get_positions()
            self.numPositions += 1

    def find_match(self, En, Eo, positionsOld):
        """ determines if atoms is the same geometry as any previously visited minima."""
        approxEn = round(En, self.minima_threshold)
        approxEo = round(Eo, self.minima_threshold)
        if approxEn in self.geometries:
            #print "energy match", approxEn
            # check if we are back in the current local minimum
            same = False
            if approxEn == approxEo:
                if self.use_get_mapping:
                    same = geometry.get_mappings(self.atoms, positionsOld, self.eps_r, self.neighbor_cutoff)
                    self.num_geo_compare[self.steps] += 1
                else:
                    same = geometry.rot_match(self.atoms, positionsOld, self.eps_r)
            if same:
                self.current_index = self.last_index
                return 1, positionsOld
            for index in self.geometries[approxEn]:
                positions = self.positionsMatrix[index]
       	        if self.use_get_mapping:
       	       	    same = geometry.get_mappings(self.atoms, positions, self.eps_r, self.neighbor_cutoff)
                    self.num_geo_compare[self.steps] += 1
       	        else:
                    same = geometry.rot_match(self.atoms, positions, self.eps_r)
                if same:
                    self.current_index = index
                    return 2, positions
            self.current_index = None
            return None, self.atoms.get_positions()
        return ValueError

    def _maxwellboltzmanndistribution(self,masses,N,temp,communicator=world):
        # For parallel GPAW simulations, the random velocities should be
        # distributed.  Uses gpaw world communicator as default, but allow
        # option of specifying other communicator (for ensemble runs)
        xi = np.random.standard_normal((len(masses), 3))
        if N is not None:
            xi = N
        momenta = xi * np.sqrt(masses * temp)[:, np.newaxis]
        communicator.broadcast(xi, 0)
        return momenta

    def MaxwellBoltzmannDistribution(self,N,temp,communicator=world,
                                     force_temp=False):
        """Sets the momenta to a Maxwell-Boltzmann distribution. temp should be
        fed in energy units; i.e., for 300 K use temp=300.*units.kB. If
        force_temp is set to True, it scales the random momenta such that the
        temperature request is precise.
        """
        momenta = self._maxwellboltzmanndistribution(self.atoms.get_masses(),N,temp,communicator)
        self.atoms.set_momenta(momenta)
        if force_temp:
            temp0 = self.atoms.get_kinetic_energy() / len(self.atoms) / 1.5
            gamma = temp / temp0
            self.atoms.set_momenta(self.atoms.get_momenta() * np.sqrt(gamma))

    def _molecular_dynamics(self, step, N):
        """Performs a molecular dynamics simulation, until mdmin is
        exceeded. If resuming, the file number (md%05i) is expected."""
        mincount = 0
        energies, oldpositions = [], []
        thermalized = False
        if not thermalized:
            self.MaxwellBoltzmannDistribution(N,
                                         temp=self.temperature * kB,
                                         force_temp=True)
        dyn = VelocityVerlet(self.atoms, dt=self.timestep * units.fs)
        while mincount < self.mdmin:
            dyn.run(1)
            energies.append(self.atoms.get_potential_energy())
            # safe guard conservation of energy
            if len(energies) > 1: # and energies[-1] - energies[-2] > 10.0:
                energydiff = energies[-1] - energies[-2]
                if energydiff > 10.0:
                    self.atoms.set_positions(oldpositions[-1])
                    dyn = VelocityVerlet(self.atoms, dt=self.timestep * 0.1 * units.fs)
                    mincount -= 1
            passedmin = self.passedminimum(energies)
            if passedmin:
                mincount += 1
            oldpositions.append(self.atoms.positions.copy())
        # Reset atoms to minimum point.
        self.atoms.positions = oldpositions[passedmin[0]]

    def get_minimum(self):
        """Return minimal energy and configuration."""
        self.atoms.set_positions(self.rmin)
        print('get_minimum',self.Emin)
        return self.Emin

    def get_energy(self, positions):
        """Return the energy of the nearest local minimum."""
        fc_before = 0.0
        if self.calculator is not None:
            fc_before = self.calculator.force_calls
        if np.sometrue(self.positions != positions):
            self.positions = positions
            self.atoms.set_positions(positions)
            if self.optimizer == "LBFGS": ##GZG changed QuasiNewton to "LBFGS"
                    opt = QuasiNewton(self.atoms, #changed self.optimizer to QuasiNewton
                    logfile=self.optimizer_logfile)
            elif self.optimizer == "FIRE":
                    opt = FIRE(self.atoms, #changed self.optimizer to FIRE
                                        maxmove = self.mss,
                                        dt = 0.2, dtmax = 1.0,
                                        logfile=self.optimizer_logfile)
            else: ##GZG this is broken, think it should set default optimizer
                opt = self.optimizer(self.atoms,
                                     logfile=self.optimizer_logfile,
                                     maxstep=self.mss)
            opt.run(fmax=self.fmax)
            self.num_localops += 1
            self.local_min_pos = self.atoms.get_positions()
            #except:
                # Something went wrong.
                # In GPAW the atoms are probably to near to each other.
                #return None
                #print("3",self.atoms.get_potential_energy())
                #return None
        temp = self.atoms.get_potential_energy()
        if self.calculator is not None:
            self.fc_local_ops += self.calculator.force_calls - fc_before
        return temp
##GZG WORK IN PROGRESS
#    def get_fitness(self, positions, fit_func):
#        """Return fitness for the nearest local minimum"""
#        if fit_func == 'energy':
#            fitness = 
    def push_apart(self,positions):
        """ Push atoms' positions apart until all atoms are no closer than self.pushapart distance from one another."""
        movea = np.zeros(np.shape(positions))
        alpha = 0.025
        for w in range(500):
            moved = 0
            movea = np.zeros(np.shape(positions))
            for i in range(len(positions)):
                for j in range(i+1,len(positions)):
                    d = positions[i] - positions[j]
                    magd = np.sqrt(np.vdot(d,d))
                    if magd < self.pushapart:
                        moved += 1
                        vec = d/magd
                        movea[i] += alpha *vec
                        movea[j] -= alpha *vec
            positions += movea
            if moved == 0:
                break
        return positions

    def trial_move(self, step, ro, distribution):
        """ Take a BH or MH trial move using a random move or
        molecular dynamics depending on distribution."""
        rn = None
        # take a BH trial move
        if self.move_type:
            rn = self.move(step, ro, distribution)
        # take a MH trial move
        else:
            old_timestep = self.timestep
            old_last = self.last_index
            old_curr = self.current_index
            positionsOld = self.atoms.get_positions()
            atomsOld = self.atoms.copy()
            Eo = self.atoms.get_potential_energy()
            # todo: remove these logs later
            rn = self.move(step,ro, self.distribution)
            En = self.get_energy(rn)
            self.log(step, En, self.Emin,self.dr)
            match, countEn, countEo = self.find_energy_match(En, Eo)
            if self.use_geo and (match is not None):
                match, position = self.find_match(En, Eo, positionsOld)
            loop_count = 1
            old_dr = self.dr
            while (match is not None):
                loop_count += 1
                if match == 1:
                # re-found last minimum
                    self.temperature *= self.beta1
                elif self.mh_history:
                # re-found previously found minimum
                    self.temperature *= self.beta2
                    self.atoms.set_positions(positionsOld)
                rn = self.move(step,ro, self.distribution)
                En = self.get_energy(rn)
                if not self.check_connected():
                    self.timestep = self.timestep * 0.1
                    self.atoms.set_positions(ro)
                    rn = ro
                    En = Eo
                match, countEn, countEo = self.find_energy_match(En, Eo)
                if self.use_geo and (match is not None):
                    match, position = self.find_match(En, Eo, positionsOld)
                    #print "match",match
                # todo: remove these logs later
                self.log(step, En, self.Emin,self.dr)
                # found a minimum that is different than ro, so stop the
                # loop if not using history
                if not self.mh_history and match != 1:
                    #self.temperature *= self.beta3
                    break
                if (self.adjust_step is not None) and loop_count >= self.adjust_step:
                    self.dr = self.dr * (1+self.adjust_fraction)
            #else:
            # must have found a new minimum
            #self.dr = old_dr
            self.temperature *= self.beta3
            #print "loop_count: ",loop_count
            self.trial_loop = np.append(self.trial_loop, loop_count)
            self.current_index = old_curr
            self.last_index = old_last
            self.timestep = old_timestep
        return rn

    def move(self, step, ro, distribution):
        """Move atoms by a random step or MD"""
        # random move
        if distribution != 'molecular_dynamics':
##GZG
            if distribution == 'uniform':
                disp = np.random.uniform(-self.dr, self.dr, (len(self.atoms), 3))

            elif distribution == 'gaussian':
                disp = np.random.normal(0,self.dr,size=(len(self.atoms), 3))

            elif distribution == 'linear':
                distgeo = self.get_dist_geo_center()
                disp = np.zeros(np.shape(self.atoms.get_positions()))
                for i in range(len(disp)):
                    maxdist = self.dr*distgeo[i]
                    disp[i] = np.random.uniform(-maxdist,maxdist,3)

            elif distribution == 'quadratic':
                distgeo = self.get_dist_geo_center()
                disp = np.zeros(np.shape(self.atoms.get_positions()))
                for i in range(len(disp)):
                    maxdist = self.dr*distgeo[i]*distgeo[i]
                    disp[i] = np.random.uniform(-maxdist,maxdist,3)

            # there is a typo in distribution or jump distribution
            else:
                print('distribution flag has typo')
                sys.exit()
                disp = np.random.uniform(-1*self.dr, self.dr, (len(self.atoms), 3))
##GZG
            # update atoms positions by the random move
            if self.move_type == 1 or self.mut_n_atoms == 'all':# moved original code to be under if statement, shouldn't affect BH if move_type flag is set
                rn = ro + disp
                rn = self.push_apart(rn)
                self.atoms.set_positions(rn)

                # refix the center of mass after random move
                if self.cm is not None:
                    cm = self.atoms.get_center_of_mass()
                    self.atoms.translate(self.cm - cm)

            elif self.move_type == 2 and self.mut_n_atoms <= len(ro):# randomly selects which atoms move and sets displacements for all others to 0
                zeros = random.sample(range(len(ro)),(len(ro)-self.mut_n_atoms))# randomly selects which atoms DO NOT move, may be a good idea to make this np later on
                for i in zeros:
                    disp[i] = [0,0,0]

                rn = ro + disp
                rn = self.push_apart(rn)
                self.atoms.set_positions(rn)

                # refix the center of mass after random move
                if self.cm is not None:
                    cm = self.atoms.get_center_of_mass()
                    self.atoms.translate(self.cm - cm)

        # Molecular dynamics
        else :
            if self.calculator is not None:
                fc_before = self.calculator.force_calls
            N = None
            # use the dimer method to create an initial velocity vector for MD
            if self.dimer_method:
                dimer = ModifiedDimer()
                N = dimer(self.atoms, self.dimer_a, self.dimer_d, self.dimer_steps, my_seed=self.seed)
                if self.calculator is not None:
                    self.fc_dimer += self.calculator.force_calls - fc_before
            if self.calculator is not None:
                fc_before = self.calculator.force_calls
            self._molecular_dynamics(step, N)
            if self.calculator is not None:
                self.fc_md += self.calculator.force_calls - fc_before

        rn = self.atoms.get_positions()
        world.broadcast(rn, 0)
        #if not self.check_connected():
        #    tsase.io.write_con('broken_bonds.con',self.atoms,w='w')
        #    return None
        return rn

    def adjust_temperature(self, match):
        if match:
            if match == 1:
            # re-found last minimum
                self.temperature *= self.beta1
            else:
            # re-found previously found minimum
                self.temperature *= self.beta2
        else:
        # must have found a new minimum
            self.temperature *= self.beta3

    def accept_criteria_bh(self, Eo, En, positionsOld):
        #print("Eo", Eo, "sef.w", self.w,"En",En)
        if En == None:
            return False
        elif Eo >= En and self.w == 0.0:
            return True
        else:
            totalMin = len(self.minima)
            approxEn = round(En, self.minima_threshold)
            approxEo = round(Eo, self.minima_threshold)
            hn = 0
            ho = 0
            # no history if the run has not accepted any previous moves (num_accepted_moves == 0)
            if self.num_accepted_moves:
                match, countEn, countEo = self.find_energy_match(En, Eo)
                if self.use_geo and (match is not None):
                    match, position = self.find_match(En, Eo, positionsOld)
                if self.adjust_temp:
                    self.adjust_temperature(match)
                if self.history_num and not self.use_geo:
                    moves = min(self.num_accepted_moves, self.history_num)
                    hn = self.temp_minima.count(approxEn) / moves
                    ho = self.temp_minima.count(approxEo) / moves
                elif self.use_geo:
                    moves = self.num_accepted_moves
                    if self.history_num:
                        moves = min(self.num_accepted_moves, self.history_num)
                    if self.current_index is not None:
                        hn = self.geo_history[self.current_index] / float(moves)
                    if self.last_index is not None:
                        ho = self.geo_history[self.last_index] / float(moves)
                else:
                    hn = countEn / self.num_accepted_moves
                    ho = countEo / self.num_accepted_moves

            kT = self.temperature * kB
            if self.accept_temp is not None:
                kT = self.accept_temp * kB
            val = ((Eo - En) + (self.w * (ho - hn))) / kT

            # RB: try adjusting dr in hw added 7/19/2019
            # if traditional boltzmann would accept but bh+hw would reject -> increase dr
            # if traditional boltzmann would reject but bh+hw would accept -> decrease dr
            # issue - no penalty for revisiting current local min
            # OR
            # if hn != 0 increase dr
            # else decrease dr
            if self.adjust_step_in_hw and self.distribution != 'molecular_dynamics':
                if hn > 0.0:
                    self.dr = self.dr * (1+self.adjust_fraction)
                else:
                    self.dr = self.dr * (1-self.adjust_fraction)

            #self.a = self.my_func(self.a)
            if val < 1.0:
                if val < -100:
                    # RB: getting an underflow in exp??
                    return False
                return (np.exp(val) > np.random.uniform())
            else:
                # accept the new position
                return True

    def accept_criteria_mh(self, Eo, En):
        if (En < (Eo + self.Ediff)):
            self.Ediff *= self.alpha1
            return True
        else:
            self.Ediff *= self.alpha2
            return False

    def running_loop(self, steps, ro, Eo, maxtemp = None):
        if self.keep_movie:
            tsase.io.write_con('movie_MHBH.con',self.atoms,w='w')
        rejectnum = 0
        acceptnum = 0
        recentaccept = 0
        count_cons = 0
        for step in range(steps):
            if self.temperature < 6.0 and self.distribution == "molecular_dynamics":
                tsase.io.write_con('lowtemp'+str(count_cons)+'.con',self.atoms,w='w')
                count_cons += 1
                if count_cons >= 8:
                    print("temp too low")
                    break
            positionsOld = self.atoms.get_positions()
            atomsOld = self.atoms.copy()
            Eo = self.atoms.get_potential_energy()
            if self.calculator is not None:
                fc_before = self.calculator.force_calls
            rn = self.trial_move(step, ro, self.distribution)
            En = self.get_energy(rn)
            print(En)
            self.log(step, En, self.Emin,self.dr)
            accept = False
            immedietly_reject = False
            if self.calculator is not None:
                self.fc_trial_move += self.calculator.force_calls - fc_before
            if self.check_bonds:
                if not self.check_connected():
                    immedietly_reject = True
            if not immedietly_reject:
                # BH acceptance
                if self.accept_criteria:
                    accept = self.accept_criteria_bh(Eo, En, positionsOld)
                # MH acceptance
                else:
                    accept = self.accept_criteria_mh(Eo, En)
            
            if self.calculator is not None:
                fc_before = self.calculator.force_calls
            # occasional jumping
            if self.jumpmax and (rejectnum > self.jumpmax):
                #print "OJ!!!"
                self.num_oj +=1
                self.jump_step.append(step)
                temp = rn
                num_jumps = self.jmp
                searching = True
                while searching:
                    rn = temp
                    for i in range(0,num_jumps):
                        rn = self.move(step,rn, self.jump_distribution)
                    En = self.get_energy(rn)
                    #coordinations = geometry.coordination_numbers(self.atoms, self.neighbor_cutoff)
                    connected_cluster = self.check_connected()
                    if not connected_cluster:
                        # bond broken: try another jump!
                        num_jumps -= 1
                        #print "RB:try less aggressive jump"
                        if num_jumps == 0:
                            # don't take an OJ if all possibilities were too aggressive
                            En = self.get_energy(temp)
                            searching = False
                    else:
                        searching = False

                # OJ was too aggresive; take a smaller OJ instead
                #if En > -10.0:
                #    rn = temp
                #    for i in range(0,3):
                #        rn = self.move(step,rn, self.jump_distribution)
                #   En = self.get_energy(rn)
                accept = True
                if self.calculator is not None:
                    self.fc_jump += self.calculator.force_calls - fc_before

            if accept:
                if self.significant_structure == True:
                    ro = self.local_min_pos.copy()
                else:
                    ro = rn.copy()
                approxEn = round(En,self.minima_threshold)

                # update self.last_index before recording this config
                match, countEn, countEo = self.find_energy_match(En, Eo)
                if self.use_geo and (match is not None):
                    match, position = self.find_match(En, Eo, positionsOld)

                if self.calculator is not None:
                    fc_before = self.calculator.force_calls
                # take a global jump
                if self.global_jump:
                    pe_count = 0
                    try:
                        pe_count = self.minima[approxEn]
                    except KeyError:
                        # no pe matching approxEn has been accepted before
                        pass
                    if pe_count >= self.global_jump:
                        if (self.use_geo and (self.geo_history[self.last_index] >= self.global_jump)) or not self.use_geo:
                            #print "global jump taken!!"
                            self.num_gj += 1
                            self.jump_step.append(step)
                            temp = rn
                            num_jumps = self.jmp
                            searching = True
                            while searching:
                                rn = temp
                                for i in range(0,num_jumps):
                                    rn = self.move(step,rn, self.jump_distribution)
                                En = self.get_energy(rn)
                                connected_cluster = self.check_connected()
                                if not connected_cluster:
                                    # bond broken: try another jump!
                                    num_jumps -= 1
                                    #print "RB:try less aggressive jump"
                                    if num_jumps == 0:
                                        # don't take an OJ if all possibilities were too aggressive
                                        En = self.get_energy(temp)
                                        searching = False
                                else:
                                    searching = False
                            #for i in range(0,self.jmp):
                            #   rn = self.move(step,rn, self.jump_distribution)
                            #   if self.global_reset:
                               # This may cause an error. Need to test!
                            #       self.minima = {}
                            #En = self.get_energy(rn)
                            approxEn = round(En,self.minima_threshold)
                            #update self.last_index after taking a gj
                            match, countEn, countEo = self.find_energy_match(En, Eo)
                            if self.use_geo and (match is not None):
                                match, position = self.find_match(En, Eo, positionsOld)
                            if self.calculator is not None:
                                self.fc_jump += self.calculator.force_calls - fc_before

                if self.use_geo:
                    self.update_geometries(En)
                self.update_minima(En, Eo)

                # update temp_minima and history dicts
                if self.history_num:
                    oldEnergy = self.temp_minima[int(self.num_accepted_moves) % self.history_num]
                    oldPosIndex = self.temp_posIndex[int(self.num_accepted_moves) % self.history_num]
                    self.temp_minima[int(self.num_accepted_moves) % self.history_num] = approxEn
                    self.temp_posIndex[int(self.num_accepted_moves) % self.history_num] = self.last_index
                    # need to delete history from geometry dicts
                    if self.use_geo and self.num_accepted_moves >= self.history_num:
                        posIndices = self.geometries[oldEnergy]
                        i = 0
                        self.geo_history[oldPosIndex] -= 1
                        self.minima[oldEnergy] -= 1
                        if self.minima[oldEnergy] == 0:
                            del self.minima[oldEnergy]
                        if self.geo_history[oldPosIndex] == 0:
                            del self.geo_history[oldPosIndex]
                            if len(posIndices) == 1:
                                del self.geometries[oldEnergy]
                            else:
                                self.geometries[oldEnergy].remove(oldPosIndex)
                acceptnum += 1.
                recentaccept += 1.
                rejectnum = 0
                self.num_accepted_moves += 1

                if En < self.Emin:
                    self.Emin = En
                    self.rmin = self.atoms.get_positions()
                    self.call_observers()
                if self.lm_trajectory is not None:
                    tsase.io.write_con(self.lm_trajectory,self.atoms,w='a')
            else:
                self.num_rejected_moves += 1
                rejectnum += 1
                self.atoms.set_positions(positionsOld)
                #  RB: try increasing current local minimum count (for +hw part)
                if self.w:
                    En = Eo
                    approxEn = round(En,self.minima_threshold)
                    # update self.last_index before recording this config
                    match, countEn, countEo = self.find_energy_match(En, Eo)
                    if self.use_geo and (match is not None):
                        match, position = self.find_match(En, Eo, positionsOld)

                    if self.use_geo:
                        self.update_geometries(En)
                    self.update_minima(En, Eo)

                    # update temp_minima and history dicts
                    if self.history_num:
                        oldEnergy = self.temp_minima[int(self.num_accepted_moves) % self.history_num]
                        oldPosIndex = self.temp_posIndex[int(self.num_accepted_moves) % self.history_num]
                        self.temp_minima[int(self.num_accepted_moves) % self.history_num] = approxEn
                        self.temp_posIndex[int(self.num_accepted_moves) % self.history_num] = self.last_index
                        # need to delete history from geometry dicts
                        if self.use_geo and self.num_accepted_moves >= self.history_num:
                            posIndices = self.geometries[oldEnergy]
                            i = 0
                            self.geo_history[oldPosIndex] -= 1
                            self.minima[oldEnergy] -= 1
                            if self.minima[oldEnergy] == 0:
                                del self.minima[oldEnergy]
                            if self.geo_history[oldPosIndex] == 0:
                                del self.geo_history[oldPosIndex]
                                if len(posIndices) == 1:
                                    del self.geometries[oldEnergy]
                                else:
                                    self.geometries[oldEnergy].remove(oldPosIndex)
                    self.num_accepted_moves += 1
                 # RB: end weird history tracking

            if self.keep_minima_arrays:
                np.put(self.local_minima, step, self.atoms.get_potential_energy())
                np.put(self.global_minima, step, self.Emin)
                # RB: only for keeping all temperature
                np.put(self.allTemps, step, self.temperature)
                np.put(self.saveEdiff, step, self.Ediff)
            self.steps += 1
            if self.minenergy != None:
                if Eo < self.minenergy:
                    break
            if self.adjust_step is not None:
                if step % self.adjust_step == 0:
                    ratio = float(acceptnum)/float(steps)
                    ratio = float(recentaccept)/float(self.adjust_step)
                    recentaccept = 0.
                    if ratio > self.target_ratio:
                       self.dr = self.dr * (1+self.adjust_fraction)
                    elif ratio < self.target_ratio:
                        self.dr = self.dr * (1-self.adjust_fraction)
            if maxtemp and maxtemp < self.temperature:
                  print("At maxtemp", self.temperature)
                  break
            if self.max_localops and self.num_localops > (steps+1):
                print("At max local ops"), self.num_localops
                break
            if self.keep_movie:
                tsase.io.write_con('movie_MHBH.con',self.atoms,w='a')
              
    def run(self, steps, maxtemp = None):
        """Hop the basins for defined number of steps."""
        np.random.seed(seed=self.seed)
        self.steps = 0
        ro = self.positions
        Eo = self.get_energy(ro)
        if self.keep_minima_arrays:
            self.global_minima = np.zeros(steps + 1)
            self.local_minima = np.zeros(steps + 1)
            self.allTemps = np.zeros(steps + 1)
            self.saveEdiff = np.zeros(steps + 1)
        if self.use_geo:
            #self.positionsMatrix = np.zeros(steps + 1)
            # best way to get total number of atoms in the system??
            self.positionsMatrix = np.zeros((steps + 1, self.atoms.get_number_of_atoms(), 3))
            self.num_geo_compare = np.zeros(steps + 1)
        self.running_loop(steps, ro, Eo, maxtemp)
        self.get_minimum()

    def get_dist_geo_center(self):
        position = self.atoms.get_positions()
        geocenter = np.sum(position,axis=0)/float(len(position))
        distance = np.zeros(len(position))
        for i in range(len(distance)):
            vec = position[i]-geocenter
            distance[i] = np.sqrt(np.vdot(vec,vec))
        distance /= np.max(distance)  #np.sqrt(np.vdot(distance,distance))
        return distance

    def _unique_minimum_position(self):
        """Identifies if the current position of the atoms, which should be
        a local minima, has been found before."""
        unique = True
        dmax_closest = 99999.
        compare = CompareEnergies()
        #self._read_minima()
        for minimum in self._minima:
            dmax = compare(minimum, self._atoms)
            if dmax < self._minima_threshold:
                unique = False
            if dmax < dmax_closest:
                dmax_closest = dmax
        return unique, dmax_closest


class ModifiedDimer:
# Class that moves the initial velocity vector of a MD escape trial
# toward a direction with low curvature
    def __call__(self,atoms,dimer_a,dimer_d,dimer_steps, my_seed=None):
        np.random.seed(seed=my_seed)
        p = atoms.copy()
        #count = counter
        oldPos = atoms.get_positions()
        # RB: use the same calculator atoms parameter was using
        #lj = tsase.calculators.lj(cutoff=35.0)
        #p.set_calculator(lj)
        p.set_calculator(atoms.get_calculator())
        N = self.gradientDimer(p,dimer_a,dimer_d,dimer_steps)
        atoms.set_positions(oldPos)
        return N

    def perpForce(self,p, N):
        # function that calculates the perpendicular force
        force = p.get_forces()
        f = force - np.vdot(force,N) * N
        return f

    def escapeDirection(self,x, y):
        # function that calculates the escape direction vector
        diff = y - x
        return diff / np.linalg.norm(diff)

    def gradientDimer(self,p,dimer_a,dimer_d,dimer_steps):
        localOpt = tsase.optimize.SDLBFGS(p,logfile = None)
        localOpt.run()
        x = p.get_positions()
        a = dimer_a
        d = dimer_d
        # random uniform vector
        N = np.random.standard_normal((len(p), 3))
        Nmag = np.linalg.norm(N)
        N = N / np.linalg.norm(N)
        y = x + d * N
        p.set_positions(y)
        maxIteration = dimer_steps
        iteration = 0

        # After a few steps the iteration is stopped before a locally
        # optimal lowest curvature mode is found
        while iteration < maxIteration:
            f = self.perpForce(p,N)
            y = y + a * f
            N = self.escapeDirection(x,y)
            y = x + d * N
            p.set_positions(y)
            iteration += 1
        return Nmag * N


class PassedMinimum:
    """Simple routine to find if a minimum in the potential energy surface
    has been passed. In its default settings, a minimum is found if the
    sequence ends with two downward points followed by two upward points.
    Initialize with n_down and n_up, integer values of the number of up and
    down points. If it has successfully determined it passed a minimum, it
    returns the value (energy) of that minimum and the number of positions
    back it occurred, otherwise returns None."""

    def __init__(self, n_down=2, n_up=2):
        self._ndown = n_down
        self._nup = n_up

    def __call__(self, energies):
        if len(energies) < (self._nup + self._ndown + 1):
            return None
        status = True
        index = -1
        for i_up in range(self._nup):
            if energies[index] < energies[index - 1]:
                status = False
            index -= 1
        for i_down in range(self._ndown):
            if energies[index] > energies[index - 1]:
                status = False
            index -= 1
        if status:
            return (-self._nup - 1), energies[-self._nup - 1]


class CompareEnergies:
    """Class that compares the potential energy of 'M' and 'Mcurrent'
    or 'M' with all other local minima perviously found
    """
    def __call__(self, atoms1, atoms2):
        atoms1 = atoms1.copy()
        atoms2 = atoms2.copy()
        dmax = self._indistinguishable_compare(atoms1, atoms2)
        return dmax

    def _indistinguishable_compare(self, atoms1, atoms2):
        lj = tsase.calculators.lj()
        atoms1.set_calculator(lj)
        atoms2.set_calculator(lj)
        difference = atoms2.get_potential_energy() - atoms1.get_potential_energy()
        dmax = np.absolute(difference)
        return dmax
