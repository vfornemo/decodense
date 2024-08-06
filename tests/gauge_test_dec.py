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


H2O_GEOM0 = '''
O  0.000000000000   0.000000000000   0.000000000000
H  1.814137079582   0.000000000000   0.000000000000 
H -0.454836704346   0.000000000000   1.756193871956 
'''

H2O_GEOM1 = '''
O  0.500000000000   0.000000000000   0.000000000000
H  2.314137079582   0.000000000000   0.000000000000 
H  0.045163295654   0.000000000000   1.756193871956 
'''

H2O_GEOM2 = '''
O  0.000000000000   0.500000000000   0.000000000000
H  1.814137079582   0.500000000000   0.000000000000 
H -0.454836704346   0.500000000000   1.756193871956 
'''

H2O_GEOM3 = '''
O  0.000000000000   0.000000000000   0.500000000000
H  1.814137079582   0.000000000000   0.500000000000 
H -0.454836704346   0.000000000000   2.256193871956 
'''

GEOMS = [H2O_GEOM0, H2O_GEOM1, H2O_GEOM2, H2O_GEOM3]
MOL = ['H2O']
BASIS = ['aug-pcseg-1']
MO = ['can', 'pm']
POP = ['mulliken','iao']

# save to csv
filename = 'dip_H2O.csv'
f = open(filename, 'a', buffering = 1)

f.write("molecule,basis set,mo_basis,pop_method,O0x,O0y,O0z,H1x,H1y,H1z,H2x,H2y,H2z,total_x,total_y,total_z\n")


for basis in BASIS:
    for mo_basis in MO:
        for pop_method in POP:
            for geom in GEOMS:
                dipmom_tot = np.zeros(3)
                print(f'H2O /{basis}/{mo_basis}/{pop_method}')
                mol = gto.M(atom=geom, basis=basis, unit='au')
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
                
                f.write(f"H2O,{basis},{mo_basis},{pop_method}")
                
                for i in range(3):
                    for axis in (' (x)', ' (y)', ' (z)'):
                        f.write(',')
                        f.write(str(res[decodense.decomp.CompKeys.tot + axis][i]))
                
                print("res", res)
                for ax_idx, axis in enumerate((' (x)', ' (y)', ' (z)')):
                    dipmom_tot[ax_idx] = np.sum(res[decodense.decomp.CompKeys.tot + axis])
                print(f'{mo_basis}/{pop_method} Dipole', dipmom_tot)
                for i in range(3):
                    f.write(',')
                    f.write(str(dipmom_tot[i]))
                f.write("\n")
                

