"""
This neural-network machine-learning optimizer for NEB.
Initially contributed by Lei Li
Other contributors:
"""
import os,sys
import time, copy
import ase.io
import numpy as np
from amp import Amp
from ase.io import Trajectory
from math import sqrt
from amp.utilities import TrainingConvergenceError
from scipy.optimize import minimize
from ase.optimize.optimize import Optimizer
from ase.optimize import FIRE
from ase.calculators.eam import EAM
from ase.calculators.lj import LennardJones
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.calculator import PropertyNotImplementedError
from tsase import neb
from tsase.neb.optimize.base import minimizer_ssneb
from tsase.neb.util import vmag

def make_dir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
           raise

class DoubleCalculator(Calculator):
    implemented_properties = ['energy', 'forces']
    default_parameters = {}
    nolabel = True

    def __init__(self, calc0=None, calc1=None, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.calc0 = calc0
        self.calc1 = calc1

    def calculate(self, atoms=None,
                  properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        atoms0 =self.atoms.copy()
        atoms1 =self.atoms.copy()
        atoms0.set_calculator(self.calc0)
        atoms1.set_calculator(self.calc1)
        self.results['energy'] = atoms0.get_potential_energy()[0] + atoms1.get_potential_energy()
        self.results['forces'] = atoms0.get_forces() + atoms1.get_forces()

class pseudoCalculator(Calculator):
    implemented_properties = ['energy', 'forces']
    default_parameters = {}
    nolabel = True

    def __init__(self, energy=None, forces=None,**kwargs):
        Calculator.__init__(self, **kwargs)
        self.energy = energy
        self.forces = forces

    def calculate(self, atoms=None,
                  properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        self.results['energy'] = self.energy
        self.results['forces'] = self.forces

class NNML(minimizer_ssneb):
    """
    Optimizer that uses machine-learning methods to get a rough PES
    """

    def __init__(self, band, restart=None, logfile='-', trajectory=None,
                 ml_module=None, max_training_cycle=10, lossfunction=None, regressor=None,
                 optimizer=FIRE, optimizer_logfile='ml_opt.log', maxstep=0.1, dt=0.1, dtmax=0.2,
                 force_consistent=None
                ):
        """
           ml_module = Amp(descriptor=Gaussian(Gs=Gs,
                                               cutoff=Cosine(6.0)
                                              #cutoff=Polynomial(gamma=5, Rc=3.0)
                                               ),
                           cores=12,
                           model=NeuralNetwork(hiddenlayers=(50,), activation='sigmoid'))
        """
        minimizer_ssneb.__init__(self, band)

        #duplicated path to store amp calculator
        self.band_ml = copy.deepcopy(band)
        self.nimages = self.band.numImages
        #use a pre-defined approxPot to avoid unphysical structure
        #self.approxPot = LennardJones(epsilon=0.65, sigma=2.744)
        self.cwd = os.getcwd()
        self.approxPot = EAM(potential=self.cwd+'/ffield.eam.alloy')
        self.approxPotBand = copy.deepcopy(band)
        for i in range(self.nimages):
            self.approxPotBand.path[i].set_calculator(self.approxPot)
            self.approxPotBand.path[i].get_potential_energy()
            self.approxPotBand.path[i].get_forces()
        #self.approxPot_replica.set_calculator(self.approxPot)

        #self.logfile = open(logfile, 'w')
        #TODO: check if parameters are correctly set up
        self.ml_module   = ml_module
        if lossfunction is not None:
           self.ml_module.model.lossfunction = lossfunction
        if regressor is not None:
           self.ml_module.model.regressor = regressor

        #Optimizer used to relax geometry on ML PES
        self.optimizer = optimizer
        self.optimizer_logfile = optimizer_logfile
        self.progress_log = open('progress.log','w')
        self.maxstep = maxstep
        self.dt = dt
        self.dtmax = dtmax
        self.force_consistent = force_consistent
        self.training_set = []
        if not os.path.exists(self.cwd + '/amp_train'):
           make_dir(self.cwd + '/amp_train')
        if not os.path.exists(self.cwd + '/amp_neb'):
           make_dir(self.cwd + '/amp_neb')
        self.training_traj = Trajectory(self.cwd + '/amp_train/'+'training.traj','w')
        self.ml_e = None
        #self.ml_log = open('ml_opt.log', 'w')
     
        self.train_endpoints = True
        self.calc_endpoints = True
        self.numbTrain = 0
        self.numbRelax = 0
        self.function_calls =0
        self.force_calls = 0

    def relax_model(self, r0s):
        """
        Minimization on ml PES
        """
        dir_relax = self.cwd + '/amp_neb/0' + str(self.numbRelax)
        self.numbRelax += 1
        if not os.path.exists(dir_relax):
           make_dir(dir_relax)
        os.chdir(dir_relax)

        for i in range(self.nimages):
            # making a directory for each image, which is nessecary for vasp to read last step's WAVECAR
            # also, it is good to prevent overwriting files for parallelizaiton over images
            fdname = '0'+str(i)
            #try:
            #   os.system('rm -rf '+fdname)
            #except:
            #   pass
            os.mkdir(fdname)
            #if not os.path.exists(fdname): os.mkdir(fdname)
        # Endpoints need to be calculate since they're not initiated with a given band
        if self.calc_endpoints:
           os.chdir(dir_relax+'/00')
           self.band_ml.path[0].get_potential_energy()
           self.band_ml.path[0].get_forces()
           os.chdir(dir_relax+'/0'+str(self.nimages-1))
           self.band_ml.path[-1].get_potential_energy()
           self.band_ml.path[-1].get_forces()
           #self.calc_endpoints = False
           os.chdir(dir_relax)

        opt = neb.fire_ssneb(self.band_ml, maxmove =self.maxstep, dtmax = self.dtmax, dt=self.dt)
        self.progress_log.write("  Relax geometry with the machine-learning force field\n")
        for i in range(self.nimages-2):
           self.band_ml.path[i+1].set_positions(r0s[i])

        opt.minimize(forceConverged=0.10, maxIterations = 100)

        r1s = np.array([self.band_ml.path[i+1].get_positions() for i in range(self.nimages-2)])
       
        #if np.all((r1s-r0s) < 0.002 ):
        #   print "atoms not moved"
        os.chdir(self.cwd)
        self.progress_log.write("    Relax geometries done\n")
        return r1s

    def update(self, rs, es, fs):
        """
        training data with machine-learning module given by ml_module
        """
        self.progress_log.write("  Training the machine-learning model:\n")
        dir_train =self.cwd + '/amp_train/0'+ str(self.numbTrain)
        self.progress_log.write("    {:s}\n".format(dir_train))
        self.numbTrain += 1
        if not os.path.exists(dir_train):
           make_dir(dir_train)
        os.chdir(dir_train)

        if self.train_endpoints:
           self.progress_log.write("    End points used for training\n")
           nimages = self.nimages
           n_offset = 0
           self.train_endpoints = False
        else:
           nimages = self.nimages - 2
           n_offset = 1

        for i in range(nimages):
           if es[i] < -65.0:
              self.progress_log.write("    Error Energies: {:12.6f}\n".format(es[i]))
              cdir = '0'+str(i+n_offset)
              os.system('cp -r '+cdir+' error_'+cdir)
              continue


           self.approxPotBand.path[i+n_offset].set_positions(rs[i])
           f = fs[i] - self.approxPotBand.path[i+n_offset].get_forces()
           e = es[i] - self.approxPotBand.path[i+n_offset].get_potential_energy()
           
           self.band.path[i+n_offset].set_positions(rs[i])
           pseudoAtoms = self.band.path[i+n_offset].copy()
           pseudoAtoms.set_calculator(pseudoCalculator(energy= e, \
                                                       forces= f))

           pseudoAtoms.get_potential_energy()
           pseudoAtoms.get_forces()
           self.training_set.append(pseudoAtoms)
           self.training_traj.write(pseudoAtoms)

        #self.training_set=Trajectory('training.traj','r')
        #os.chdir(workdir)
        #if os.path.exists('amp-fingerprint-primes.ampdb'):
        #   os.system('rm -rf amp-fingerprint-primes.ampdb')
        #if os.path.exists('amp-fingerprints.ampdb'):
        #   os.system('rm -rf amp-fingerprints.ampdb')
        #if os.path.exists('amp-fingerprints.ampdb'):
        #   os.system('rm -rf amp-neighborlists.ampdb')
        #if os.path.exists('checkpoint'):
        #   os.system('rm checkpoint')
        #try:
        #   os.system('rm tfAmpNN-checkpoint')
        #   os.system('rm tfAmpNN-checkpoint.meta')
        #except:
        #   print "error on rm"
        #   pass
        #if os.path.exists('amp.amp'):
        #   os.system('rm amp.amp')
        #if os.path.exists('amp-untrained-parameters.amp'):
           #load nn model including lossfunction
        #   os.system('rm amp-untrained-parameters.amp')
        try:
           #self.progress_log.write("Train ml model\n")
           self.ml_module.train(images='../training.traj', overwrite=True)
        except TrainingConvergenceError:
           os.system('mv amp-untrained-parameters.amp amp.amp')
           pass
        #load ml model
        try:
           ml_calc = Amp.load('amp.amp')
        except:
           ml_calc = Amp.load('amp-untrained-parameters.amp')
           pass

        double_calc = DoubleCalculator(calc0=ml_calc, calc1=self.approxPot)
        for i in range(self.nimages):
            self.band_ml.path[i].set_calculator(double_calc)
        self.progress_log.write("    Training done\n")
        os.chdir(self.cwd)

    def get_path_ref(self):
        #get positions for intermidate images
        start_time = time.time()
        self.progress_log.write("  Calculating DFT potential energies and NEB forces.\n")
        self.band.forces()
        end_time = time.time()
        self.progress_log.write("    Time used: {:12.6f}\n".format(end_time-start_time))
        if self.train_endpoints:
           rs = [self.band.path[i].get_positions()\
                        for i in range(self.nimages)]
           es = [self.band.path[i].u \
                for i in range(self.nimages)]
           fs = [self.band.path[i].f \
                 for i in range(self.nimages)]
        else:
           rs = [self.band.path[i+1].get_positions()\
                        for i in range(self.nimages-2)]
           es = [self.band.path[i+1].u \
                for i in range(self.nimages-2)]
           fs = [self.band.path[i+1].f \
                 for i in range(self.nimages-2)]
        self.progress_log.write("    DFT Energies: {}\n".format(','.join([str(e) for e in es])))
        self.progress_log.write("    Energy-Force fetching time: {:12.6f}\n".format(time.time()-end_time))
        fMax = 0.
        for i in range(1, self.nimages - 1):
           fi  = vmag(self.band.path[i].totalf)
           if fi > fMax:
               fMax = fi
        return np.array(rs), np.array(es), np.array(fs), fMax

    def set_path_rs(self, rs):
        for i in range(self.nimages-2):
           self.band.path[i+1].set_positions(rs[i])

    def step(self):
        self.progress_log.write("================================================\n")
        self.progress_log.write("New step started:\n")
        r0s, e0s, f0s, f0Max = self.get_path_ref()
        #update ml model
        self.update(r0s, e0s, f0s)
        self.progress_log.flush()
     
        #relax atoms on ml-rough PES
        r1s = self.relax_model(r0s)
        self.progress_log.flush()

        self.set_path_rs(r1s)

        r1s, e1s, f1s, f1Max = self.get_path_ref()
        self.progress_log.flush()

        self.function_calls += 1
        self.force_calls += 1

        count = 0
        self.progress_log.flush()
        e0s = e0s[1:self.nimages-1]
        #while np.all(e1s >= e0s):
        while f1Max > f0Max:
            self.update(r1s, e1s, f1s)
            self.progress_log.flush()
            r1s = self.relax_model(r0s)

            self.set_path_rs(r1s)
            r1s, e1s, f1s, f1Max = self.get_path_ref()

            self.function_calls += 1
            self.force_calls += 1
            self.progress_log.write("  Opted with ML: {:3d} {:16.8f} {:16.8f}\n".format(count, f1Max, f0Max))
            self.progress_log.flush()

            count += 1
            if count == 30:
                raise RuntimeError('A descent model could not be built')
