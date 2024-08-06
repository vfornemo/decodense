import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pyscf import gto, scf
import numpy as np


# MO_BASIS = ['can', 'pm']
MO_BASIS = ['can']
POP_METHOD = ['mulliken', 'iao']

np.set_printoptions(threshold=100000)


# NH3 Molecule
mol = gto.Mole()
mol.atom = '''
N
H 1 1.008000
H 1 1.008000 2 109.47
H 1 1.008000 2 109.47 3 120
'''

mol.basis = 'cc-pvdz'
mol.build()


# mf calc
mf = scf.RHF(mol)
mf.conv_tol = 1e-14
mf.kernel()

print('\n###### NH3 ######\n')

print("mf.mo_coeff", mf.mo_coeff)
