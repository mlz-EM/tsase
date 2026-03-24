from pathlib import Path

import numpy as np
from mace.calculators import MACECalculator
from tsase import neb
from ase.io import read
from ase.optimize import FIRE, BFGS
import os

ROOT = Path(__file__).resolve().parents[0]

# Input structures and model available in this repository.
STRUCTURE_PATHS = ROOT / "AFE" 

MODEL_PATH = ROOT / "example" / "MACE_model.model"



structures = sorted([path for path in os.listdir(STRUCTURE_PATHS) if path.endswith('.cif')])


ref = -980.136346-7.842693/1000
base_calc = MACECalculator(model_paths=str(MODEL_PATH), device="cuda")
for path in structures:
    atoms = read(STRUCTURE_PATHS/path) 
    atoms.set_calculator(base_calc)
    optimizer = BFGS(atoms, logfile=None)  # you can also use BFGS(atoms)
    optimizer.run(fmax=0.05)  # fmax in eV/Å
    energy = atoms.get_potential_energy()
    n_atoms = len(atoms)

    print(f"Final energy per atom of {path}: {(energy / n_atoms-ref)*1000:.6f} meV")

    atoms.write(path.replace('.cif', '_relaxed.cif'))