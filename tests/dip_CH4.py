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
MOL2 = ['CH4']
MOL_DICT = {'NH3': nh3, 'H2O': h2o, 'HF': hf, 'CH4': ch4}
BASIS = ['aug-pcseg-1', '6-311++G**'] #, 'sadlej-pvtz'
MO = ['can', 'pm', 'fb']
POP = ['mulliken','iao']

# save to csv
filename = 'dip_CH4.csv'
f = open(filename, 'a', buffering = 1)

f.write("molecule,basis set,mo_basis,pop_method,C0x,C0y,C0z,H1x,H1y,H1z,H2x,H2y,H2z,H3x,H3y,H3z,H4x,H4y,H4z\n")

for mol2 in MOL2:
    for basis in BASIS:
        for mo_basis in MO:
            for pop_method in POP:
                dipmom_tot = np.zeros(3)
                print(f'{mol2}/{basis}/{mo_basis}/{pop_method}')
                mol = MOL_DICT[mol2]
                mf = scf.RHF(mol)
                mf.conv_tol = 1e-14
                mf.kernel()
                
                E0 = np.array([.0, .0, .0])
                ao_dip = mol.intor_symmetric('int1e_r', comp=3)
                h1 = mf.get_hcore()
                field = np.einsum('x,xij->ij', E0, ao_dip)
                mf.get_hcore = lambda *args, **kwargs: h1 + field
                mf.kernel()

                decomp = decodense.DecompCls(part='atoms', mo_basis=mo_basis, prop='dipole', verbose=0, pop_method=pop_method)
                res = decodense.main(mol, decomp, mf)
                
                f.write(f"{mol2},{basis},{mo_basis},{pop_method}")
                
                for i in range(5):
                    for axis in (' (x)', ' (y)', ' (z)'):
                        f.write(',')
                        f.write(str(res[decodense.decomp.CompKeys.tot + axis][i]))
                f.write("\n")
                
                print("res", res)
                for ax_idx, axis in enumerate((' (x)', ' (y)', ' (z)')):
                    dipmom_tot[ax_idx] = np.sum(res[decodense.decomp.CompKeys.tot + axis])
                print(f'{mo_basis}/{pop_method} Dipole', dipmom_tot)

