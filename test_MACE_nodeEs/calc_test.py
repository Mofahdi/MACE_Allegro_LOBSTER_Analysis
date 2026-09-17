import os
import numpy as np
import pandas as pd
import torch  
from ase.io import read

from mace.calculators.mace import MACECalculator
from mace.calculators import mace_mp

atoms = read("CONTCAR.gz")
device = "cuda" if torch.cuda.is_available() else "cpu"

mace_calculator = MACECalculator(model_paths='mace-mpa-0-medium.model', model_dtype='float32', device=device)
#calc = mace_mp(model="medium-mpa-0", device="cpu", default_dtype='float32')

atoms.calc = mace_calculator
#atoms.calc = calc

# Total energy
#E = atoms.get_potential_energy()

# Per-node/site energies
#atomic_energies = atoms.calc.results["node_energy"]
atomic_energies = atoms.get_potential_energies()

#print("Total:", E)
print("Node energies:", type(atomic_energies)) #node_E)
