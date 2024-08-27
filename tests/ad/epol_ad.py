
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import jax
from jax import jacrev
from jax import numpy as jnp
from pyscfad import gto
import decodense
from decodense.ad.prop.dipole import energy
from pyscfad import config

config.update('pyscfad_scf_implicit_diff', True)

jnp.set_printoptions(threshold=100000)
jnp.set_printoptions(linewidth=jnp.inf)

# print this script
print(open(__file__).read())
print("-------------- Log starts here --------------")

# Criteria of decomposition
e_decomp1 = decodense.DecompCls(part='atoms', mo_basis='pm', prop='energy', verbose=0, pop_method='mulliken')
e_decomp2 = decodense.DecompCls(part='atoms', mo_basis='pm', prop='energy', verbose=0, pop_method='iao')

# static external electric field
E0 = jnp.array([0.] * 3)
# localization threshold
LOC_TOL = 1.e-10
# basis set
BASIS = 'aug-pcseg-1'
# orbitals and population method
MOS = ['pm']
POP_METHODS = ['iao', 'mulliken']

# test molecules and reference polarizabilities (from pyscf prop module)
H2O_GEOM = '''
O
H 1 0.96
H 1 0.96 2 104.52
'''
HF_GEOM = '''
H
F 1 0.91
'''
H2O_POL_REF = jnp.array([[ 8.85292796e+00,  3.33125639e-15, -3.56629337e-01],
                         [ 3.33125639e-15,  7.81868672e+00, -2.96946650e-15],
                         [-3.56629337e-01, -2.96946650e-15,  8.66820107e+00]])
HF_POL_REF = jnp.array([[ 5.63380397e+00,  3.10959550e-16, -5.55134039e-17],
                        [ 3.10959550e-16,  4.36007043e+00,  1.05471187e-15],
                        [-5.55134039e-17,  1.05471187e-15,  4.36007043e+00]])
MOLS = ['HF', 'H2O']
GEOMS = [HF_GEOM, H2O_GEOM]
POL_REFS = [HF_POL_REF, H2O_POL_REF]

# loop over test molecules
for mol_name, geom, pol_ref in zip(MOLS, GEOMS, POL_REFS):
    for mo in MOS:
        for pop_method in POP_METHODS:
            mol = gto.Mole()
            mol.atom = geom
            mol.basis = BASIS
            mol.verbose = 4
            mol.build(trace_coords=False, trace_exp=False, trace_ctr_coeff=False)
            # atomic polarizability
            decomp = decodense.DecompCls(part='atoms', mo_basis=mo, prop='energy', verbose=0, pop_method=pop_method)
            pol = jnp.sum(-jacrev(jacrev(energy))(E0, decomp, mol, sweep=False), axis=0)
            # assert differences
            print(f'{mol_name:} / {mo:} / {pop_method:}:')
            print('total polarizabilities:\n', pol - pol_ref)
            print('isotropic polarizabilities:\n', (jnp.trace(pol) - jnp.trace(pol_ref)) / 3.)
