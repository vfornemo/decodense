import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pyscf import gto, scf
import decodense
import numpy as np
import pandas as pd

# print this script
print(open(__file__).read())
print("-------------- Log starts here --------------")


np.set_printoptions(threshold=100000)
np.set_printoptions(linewidth=np.inf)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 5000)


nh3 = gto.Mole()
nh3.atom = '''
N
H 1 1.008000
H 1 1.008000 2 109.47
H 1 1.008000 2 109.47 3 120
'''
hf = gto.Mole()
hf.atom = '''
H
F 1 0.91
'''

h2o = gto.Mole()
h2o.atom = '''
O
H 1 0.96
H 1 0.96 2 104.52
'''

ch4 = gto.Mole()
ch4.atom = '''
C
H 1 1.087
H 1 1.087 2 109.4712206
H 1 1.087 2 109.4712206 3 120
H 1 1.087 2 109.4712206 4 120
'''

MOL = [nh3, h2o, hf, ch4]
MOL2 = ['H2O','CH4']
MOL_DICT = {'NH3': nh3, 'H2O': h2o, 'HF': hf, 'CH4': ch4}
BASIS = ['aug-pcseg-1', '6-311++G**'] #, 'sadlej-pvtz'
MO = ['can', 'pm', 'fb']
POP = ['mulliken','iao']








for mol2 in MOL2:
    for basis in BASIS:
        for mo_basis in MO:
            for pop_method in POP:
                print(f'{mol2}/{basis}/{mo_basis}/{pop_method}')
                mol = MOL_DICT[mol2]
                mf = scf.RHF(mol)
                mf.conv_tol = 1e-14
                mf.kernel()
                decomp = decodense.DecompCls(part='atoms', mo_basis=mo_basis, prop='energy', verbose=0, pop_method=pop_method)
                res = decodense.main(mol, decomp, mf)
                print("res", res)
                e_tot = np.sum(res[decodense.decomp.CompKeys.tot])
                print(f'{mo_basis}/{pop_method} Energy', e_tot)

