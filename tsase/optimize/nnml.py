import os,sys
import time
import ase.io
import numpy as np
from amp import Amp
from ase.io import Trajectory
from math import sqrt
from amp.utilities import TrainingConvergenceError
from scipy.optimize import minimize
from ase.optimize.optimize import Optimizer
from ase.optimize import FIRE
from ase.calculators.lj import LennardJones

def make_dir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
           raise

#TODO: minimizer wrapper that wraps all kinds of optimizer and select a defined optimizer in a case/switch manner
#      Need to modify optimize.py in ASE or TSASE to enable run(atoms,fmax, steps):
#      def run(self, atoms=None, fmax, steps):
#          if atoms is not None:
#             self.atoms = atoms
class minimizer(object):
      def __init__(self, atoms, optimizer=None,
                   maxstep=0.1, dt=0.1, dtmax=0.2,
                   trajectory = 'geoopt.traj',
                   logfile='geoopt.log'):
          self.atoms=atoms
          self.optimizer = optimizer
          self.maxstep = maxstep
          self.dt = dt
          self.dtmax = dtmax
          self.trajectory = trajectory
          self.logfile = logfile

      def get_optimizer(self):
          opt = getattr(slef, self.optimizer.__name__, lambda:"Invalid Optimizer")
          return opt

      def neb(self,):
          """
             #TODO: neb optimizer
             nim = 7  # number of images, including end points
             band = neb.ssneb(p1, p2, numImages = nim, method = 'ci', ss=False)
         
          # to restart, uncomment the following lines which read the previous optimized images into the band
          #    for i in range(1,nim-1):
          #        filename = str(i)+'.CON'
          #        b = read(filename,format='vasp')
          #        band.path[i].set_positions(b.get_positions())
          #        band.path[i].set_cell(b.get_cell())
         
             opt = neb.qm_ssneb(band, maxmove =0.2, dt=0.1)
          #Uncomment to use fire optimization algorithm
          #    opt = neb.fire_ssneb(band, maxmove =0.05, dtmax = 0.03, dt=0.03)
             opt.minimize(forceConverged=0.02, maxIterations = 200)
          """
          return "neb optimizer"

      def dimer(self,):
          return "neb optimizer"

      def fire(self, ):
          opt = self.optimizer(atoms=self.atoms,
                               maxmove = self.maxstep,
                               dt = self.dt, dtmax = self.dtmax,
                                   trajectory = self.trajectory,
                                   logfile=self.logfile)
          return opt
      def others(self,):
          opt = self.optimizer(atoms=self.atoms,
                               logfile=self.logfile,
                               trajectory = self.trajectory,
                               maxstep=self.maxstep)
          return opt

from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.calculator import PropertyNotImplementedError


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



class NNML(Optimizer):
    """
    Optimizer that uses machine-learning methods to get a rough PES
    """

    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
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
        Optimizer.__init__(self, atoms, restart, logfile, trajectory,
                           force_consistent=force_consistent)


        self.atoms = atoms
        self.replica = atoms.copy()
        #use a pre-defined approxPot to avoid unphysical structure
        self.approxPot = LennardJones(epsilon=0.65, sigma=2.744)
        self.approxPot_replica = atoms.copy()
        self.approxPot_replica.set_calculator(self.approxPot)

        #self.logfile = open(logfile, 'w')
        #TODO: check if parameters are correctly set up
        self.ml_module   = ml_module
        self.max_training_cycle = max_training_cycle
        if lossfunction is not None:
           self.ml_module.model.lossfunction = lossfunction
        if regressor is not None:
           self.ml_module.model.regressor = regressor
#        self.ml_module.model.lossfunction = LossFunction(convergence={'energy_rmse': 0.001,
#                                                                     'force_rmse': 0.02},
#                                                         force_coefficient=0.2)
        self.optimizer = optimizer
        self.optimizer_logfile = optimizer_logfile
        self.progress_log = open('progress.log','w')
        self.maxstep = maxstep
        self.dt = dt
        self.dtmax = dtmax
        self.training_set = []
        self.training_traj = Trajectory('training.traj','w')
        self.ml_e = None
        self.ml_log = open('ml_opt.log', 'w')

        self.cwd = os.getcwd()
        self.function_calls =0
        self.force_calls = 0
        """
        if self.optimizer.__name__ == "FIRE":
           self.minimizer = self.optimizer(atoms,
                                maxmove = self.maxstep,
                                dt = self.dt, dtmax = self.dtmax,
                                    trajectory = 'ml_geoopt.traj',
                                    logfile=self.optimizer_logfile)
        else:
           self.minimizer = self.optimizer(atoms,
                                logfile=self.optimizer_logfile,
                                trajectory = 'ml_geoopt.traj',
                                maxstep=self.maxstep)
        print "  Relax geometry with the machine-learning force field"
        if memo_interval is not None:
           #traj = []
           #def traj_memo(atoms=self.atoms):
           #    traj.append(atoms)
               #epot=atoms.get_potential_energy()
               #ekin=atoms.get_kinetic_energy()
           #opt.attach(traj_memo, interval=memo_interval)
           self.minimizer.attach(self.training_traj.write, interval=memo_interval)
        """

    def relax_model(self, r0):
        """
        Minimization on ml PES
        """
#        result = minimize(self.predict_e, r0, method='L-BFGS-B', jac=self.predict_g,
#                         options= {'ftol': 1e-05,
#                                   'gtol': 1e-03,
#                                   'maxfun': 1000,
#                                   'maxiter': 200})


#        if self.optimizer.__name__ == "FIRE":
        #r =r0.reshape(-1,3)
        self.replica.set_positions(r0.reshape(-1,3))
        opt = self.optimizer(self.replica,
                             maxmove = self.maxstep,
                             dt = self.dt, dtmax = self.dtmax,
                                 trajectory = 'ml_geoopt.traj',
                                 logfile=self.optimizer_logfile)
#        else:
#           opt = self.optimizer(atoms,
#                                logfile=self.optimizer_logfile,
#                                trajectory = 'ml_geoopt.traj',
#                                maxstep=self.maxstep)
        self.progress_log.write("  Relax geometry with the machine-learning force field\n")
        opt.run(fmax=0.1, steps=100)
        #if memo_interval is not None:
           #traj = []
           #def traj_memo(atoms=self.atoms):
           #    traj.append(atoms)
               #epot=atoms.get_potential_energy()
               #ekin=atoms.get_kinetic_energy()
           #opt.attach(traj_memo, interval=memo_interval)
         #  self.minimizer.attach(self.training_traj.write, interval=memo_interval)
        #print "  After opt:", atoms.get_positions()

#        if result.success:
        return self.replica.get_positions().reshape(-1)
#        else:
#            self.dump()
#            raise RuntimeError(
#                "The minimization of the acquisition function has not converged")

    def predict_e(self, r):
        self.replica.set_positions(r.reshape(-1,3))
        self.approxPot_replica.set_positions(r.reshape(-1,3))
        self.ml_e = self.replica.get_potential_energy() + self.approxPot_replica.get_potential_energy()
        #print "ml e:", self.ml_e
        return self.ml_e

    def predict_g(self,r):
        self.replica.set_positions(r.reshape(-1,3))
        self.approxPot_replica.set_positions(r.reshape(-1,3))
        fs = self.atoms.get_forces().reshape(-1) +  self.approxPot_replica.get_forces().reshape(-1)
        self.ml_log.write("ml energy: {:12.6f} max force: {:12.6f}\n".format(self.ml_e, np.amax(np.absolute(fs))))
        return -fs

    def update(self, r, e, f):
        """
        training data with machine-learning module given by ml_module
        """
        #workdir = self.cwd+'/training'
        #if not os.path.isdir(workdir):
        #   make_dir(workdir)
        r = r.reshape(-1,3)
        self.atoms.set_positions(r)
        #self.approxPot_replica.set_positions(r)
        #f = f - self.approxPot_replica.get_forces()
        #e = e - self.approxPot_replica.get_potential_energy()
          
        pseudoAtoms = self.atoms.copy()
        pseudoAtoms.set_calculator(pseudoCalculator(energy= e, \
                                                    forces= f))
        pseudoAtoms.get_potential_energy()
        pseudoAtoms.get_forces()

        self.training_set.append(pseudoAtoms)
        self.training_traj.write(pseudoAtoms)
        #self.training_set=Trajectory('training.traj','r')
        #os.chdir(workdir)
        if os.path.exists('amp-fingerprint-primes.ampdb'):
           os.system('rm -rf amp-fingerprint-primes.ampdb')
        if os.path.exists('amp-fingerprints.ampdb'):
           os.system('rm -rf amp-fingerprints.ampdb')
        if os.path.exists('amp-fingerprints.ampdb'):
           os.system('rm -rf amp-neighborlists.ampdb')
        #if os.path.exists('amp.amp'):
        #   os.system('rm amp.amp')
        #if os.path.exists('amp-untrained-parameters.amp'):
           #load nn model including lossfunction
        #   os.system('rm amp-untrained-parameters.amp')
        try:
           self.progress_log.write("Train ml model\n")
           self.ml_module.train(images='training.traj', overwrite=True)
        except TrainingConvergenceError:
           os.system('mv amp-untrained-parameters.amp amp.amp')
           pass
        #load ml model
        try:
           self.replica.set_calculator(Amp.load('amp.amp'))
        except:
           self.replica.set_calculator(Amp.load('amp-untrained-parameters.amp'))
           pass
        #os.chdir(self.cwd)
 
    def step(self, f):
    
        atoms = self.atoms
        r0 = atoms.get_positions().reshape(-1)
        e0 = atoms.get_potential_energy(force_consistent=self.force_consistent) 

        if f is None:
           f = atoms.get_forces()
        #update ml model
        self.update(r0, e0, f)

        #relax atoms on ml-rough PES
        r1 = self.relax_model(r0)
        self.atoms.set_positions(r1.reshape(-1, 3))

        e1 = self.atoms.get_potential_energy(force_consistent=self.force_consistent)
        f1 = self.atoms.get_forces()
        
        self.function_calls += 1
        self.force_calls += 1

        count = 0
        self.progress_log.write("# New step started:\n")
        while e1 >= e0:

            self.update(r1, e1, f1)
            r1 = self.relax_model(r0)

            self.atoms.set_positions(r1.reshape(-1,3))
            e1 = self.atoms.get_potential_energy(force_consistent=self.force_consistent)
            f1 = self.atoms.get_forces()

            self.function_calls += 1
            self.force_calls += 1
            self.progress_log.write("  Opted with ML: {:3d} {:16.8f} {:16.8f}\n".format(count, e0, e1))
            self.progress_log.flush()
            if self.converged(f1):
                break

            count += 1
            if count == 30:
                raise RuntimeError('A descent model could not be built')
#        self.dump()

