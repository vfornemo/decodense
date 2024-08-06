import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pyscf import gto, scf
import decodense
import numpy as np
import pandas as pd

# print this script
print(open(__file__).read())
print("-------------- Log starts here --------------")

# MO_BASIS = ['can', 'pm']
MO_BASIS = ['pm']
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
 
#  N   0.00000000    0.00000000    0.09302991
#  H   0.00000000    0.96018208   -0.21706979
#  H  -0.83154207   -0.48009104   -0.21706979
#  H   0.83154207   -0.48009104   -0.21706979

# mol.basis = 'cc-pvdz'
mol.basis = '6-311++G**'
mol.build()


# mf calc
mf = scf.RHF(mol)
mf.conv_tol = 1e-14
# mf.kernel()

E0 = np.array([.0, .0, .0])
ao_dip = mol.intor_symmetric('int1e_r', comp=3)
h1 = mf.get_hcore()
field = np.einsum('x,xij->ij', E0, ao_dip)
mf.get_hcore = lambda *args, **kwargs: h1 + field
mf.kernel()

print('\n###### NH3 ######\n')


# Criteria of decomposition
# decomp1 = decodense.DecompCls(part='atoms', mo_basis='can', prop='energy', verbose=0, pop_method='mulliken')
# decomp1 = decodense.DecompCls(part='atoms', mo_basis='can', prop='energy', verbose=0, pop_method='mulliken')
# decomp2 = decodense.DecompCls(part='atoms', mo_basis='can', prop='energy', verbose=0, pop_method='iao')
# decomp2 = decodense.DecompCls(part='atoms', mo_basis='can', prop='energy', verbose=0, pop_method='iao')

dipmom_tot = np.zeros(3)

for mo_basis in MO_BASIS:
    for pop_method in POP_METHOD:
        decomp = decodense.DecompCls(part='atoms', mo_basis=mo_basis, prop='dipole', verbose=0, pop_method=pop_method)
        res = decodense.main(mol, decomp, mf)
        print("res", res)
        for ax_idx, axis in enumerate((' (x)', ' (y)', ' (z)')):
            dipmom_tot[ax_idx] = np.sum(res[decodense.decomp.CompKeys.tot + axis])
        print(f'{mo_basis}/{pop_method} Dipole', dipmom_tot)

