import numpy
numpy.seterr(all='raise')
from tsase import io as io
from tsase import calculators
from tsase import data
from tsase import neb
from tsase import structure
from tsase import constraints
from tsase import md
from tsase import svm
from tsase import optimize
from tsase import bgsd
#import mc

def interpolate(trajectory, intermediate_frames = 8):
    interpolated = []
    for i in range(len(trajectory)-1):
        interpolated.append(trajectory[i].copy())
        for j in range(0, intermediate_frames):
            temp = trajectory[i].copy()
            temp.positions = trajectory[i].positions + (trajectory[i+1].positions-trajectory[i].positions) * ((j+1)/float(intermediate_frames+1))
            interpolated.append(temp)
    interpolated.append(trajectory[-1])
    return interpolated
    
    
