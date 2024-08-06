import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pyscfad import gto, scf
from jax import numpy as jnp
import decodense
import numpy as np
import pandas as pd

# print this script
print(open(__file__).read())
print("-------------- Log starts here --------------")

MO_BASIS = ['can', 'pm']
# MO_BASIS = ['pm']
POP_METHOD = ['mulliken']

jnp.set_printoptions(threshold=100000)
jnp.set_printoptions(linewidth=np.inf)
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
# mol.build(trace_exp=False, trace_ctr_coeff=False)

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
        res = decodense.main(mol, decomp, mf)
        # res = decodense.main(mol, decomp, mf)
        print("res", res)
        e_tot = np.sum(res[decodense.decomp.CompKeys.tot])
        print(f'{mo_basis}/{pop_method} Energy', e_tot)

