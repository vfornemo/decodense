import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import decodense
import pandas as pd
import numpy as np
from pyscf import gto, scf, dft

np.set_printoptions(threshold=100000)
np.set_printoptions(linewidth=np.inf)

# print this script
print(open(__file__).read())
print("-------------- Log starts here --------------")

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
MOL2 = ['NH3', 'H2O', 'HF', 'CH4']
MOL_DICT = {'NH3': nh3, 'H2O': h2o, 'HF': hf, 'CH4': ch4}
BASIS = ['aug-pcseg-1', '6-311++G**']
MO = ['pm', 'can', 'fb']
POP = ['mulliken','iao']



E0 = np.array([0., 0., 0.])

    
for mol2 in MOL2:
    for basis in BASIS:
        for mo in MO:
            for pop in POP:
                print(f'{mol2} {basis} {mo} {pop}\n')
                mol = MOL_DICT[mol2]
                mol.basis = basis
                mol.build()
                
                # mf calc
                mf = scf.RHF(mol)
                mf.conv_tol = 1e-14
                mf.kernel()
                
                filename = 'weights.txt'
                f = open(filename, 'a', buffering = 1)
                # str: basis set + mo_basis + pop_method
                f.write(mol2)
                f.write('_') 
                f.close()
                
                # Criteria of decomposition
                e_decomp1 = decodense.DecompCls(part='atoms', mo_basis=mo, prop='energy', verbose=0, pop_method=pop)
                # Static external electric field

                res = decodense.main(mol, e_decomp1, mf)
                # res = decodense.main(mol, decomp, mf)
                e_tot = np.sum(res[decodense.decomp.CompKeys.tot])


                    


                