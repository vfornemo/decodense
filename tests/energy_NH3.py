# test script for standard decodense

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pyscfad import gto, scf
import decodense
import numpy as np
import pandas as pd
from decodense.orbitals import gen_orbs

# print this script
print(open(__file__).read())
print("-------------- Log starts here --------------")

MO_BASIS = ['can', 'pm']
POP_METHOD = ['mulliken', 'iao']

np.set_printoptions(threshold=100000)
np.set_printoptions(linewidth=np.inf)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 5000)


# NH3 Molecule
mol = gto.Mole()
mol.atom = '''
N
H 1 1.008000
H 1 1.008000 2 109.47
H 1 1.008000 2 109.47 3 120
'''

mol.basis = 'aug-pcseg-1'
mol.build()


# mf calc
mf = scf.RHF(mol)
mf.conv_tol = 1e-14
mf.kernel()

print('\n###### NH3 ######\n')


# Criteria of decomposition
# decomp1 = decodense.DecompCls(part='atoms', mo_basis='can', prop='energy', verbose=0, pop_method='mulliken')
# decomp1 = decodense.DecompCls(part='atoms', mo_basis='can', prop='energy', verbose=0, pop_method='mulliken')
# decomp2 = decodense.DecompCls(part='atoms', mo_basis='can', prop='energy', verbose=0, pop_method='iao')
# decomp2 = decodense.DecompCls(part='atoms', mo_basis='can', prop='energy', verbose=0, pop_method='iao')

for mo_basis in MO_BASIS:
    for pop_method in POP_METHOD:
        decomp = decodense.DecompCls(part='atoms', mo_basis=mo_basis, prop='energy', verbose=0, pop_method=pop_method)
        
        # get orbitals and mo occupation
        mo_coeff, mo_occ = gen_orbs(mol, mf, decomp)
        res = decodense.main(mol, decomp, mf, mo_coeff=mo_coeff, mo_occ=mo_occ, AD=False)
        print("res", res)
        e_tot = np.sum(res[decodense.decomp.CompKeys.tot])
        print(f'{mo_basis}/{pop_method} Energy', e_tot)

