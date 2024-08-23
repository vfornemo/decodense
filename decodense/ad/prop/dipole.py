import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import jax
from jax import jacrev
from jax import numpy as jnp
from pyscfad import gto, scf, lo
import decodense
from ..orbitals_ad import loc_orbs

# Core Hamiltonian modified energy
def dipole1(E, decomp, mol, sweep=False):
    
    def hf_e(E, decomp, mf, ao_dip, h1):
        field = jnp.einsum('x,xij->ij', E, ao_dip)
        mf.get_hcore = lambda *args, **kwargs: h1 + field
        mf.kernel()
        # Localize here
        
        mo_coeff, mo_occ = loc_orbs(mol, mf, decomp.mo_basis, decomp.pop_method,
                            decomp.mo_init, decomp.loc_exp, decomp.verbose)
        
        ad = True
        e_part = decodense.main(mol = mol, decomp = decomp, mf = mf, mo_coeff = mo_coeff, mo_occ = mo_occ, AD = ad)

        # dimension of the electronic part needs to be considered
        e_tot = e_part[decodense.decomp.CompKeys.el] + e_part[decodense.decomp.CompKeys.struct]
        return e_tot
    
    def nucl_dip(E, decomp, mf, ao_dip, h1):
        field = jnp.einsum('x,xij->ij', E, ao_dip)
        mf.get_hcore = lambda *args, **kwargs: h1 + field
        mf.kernel()
        # Localize here
        mo_coeff, mo_occ = loc_orbs(mol, mf, decomp.mo_basis, decomp.pop_method,
                    decomp.mo_init, decomp.loc_exp, decomp.verbose)
        ad = True
        e_part = decodense.main(mol = mol, decomp = decomp, mf = mf, mo_coeff = mo_coeff, mo_occ = mo_occ, AD = ad)
        nuc_dip = e_part[decodense.decomp.CompKeys.nuc_dip]
        return nuc_dip
    
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    mf.kernel()
    ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    h1 = mf.get_hcore()
    el_dip = -jacrev(hf_e)(E, decomp, mf, ao_dip, h1)
    nuc_dip = nucl_dip(E, decomp, mf, ao_dip, h1)
    return el_dip + nuc_dip

def dipole2(E, decomp, mol, sweep=False):
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    mf.kernel()
    ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    h1 = mf.get_hcore()
    field = jnp.einsum('x,xij->ij', E, ao_dip)
    mf.get_hcore = lambda *args, **kwargs: h1 + field
    mf.kernel()
    # Localize here
    mo_coeff, mo_occ = loc_orbs(mol, mf, decomp.mo_basis, decomp.pop_method,
                        decomp.mo_init, decomp.loc_exp, decomp.verbose)
    ad = True
    dip_part = decodense.main(mol = mol, decomp = decomp, mf = mf, mo_coeff = mo_coeff, mo_occ = mo_occ, AD = ad)
    dip_tot = dip_part[decodense.decomp.CompKeys.el] + dip_part[decodense.decomp.CompKeys.struct]
    return dip_tot

# Core Hamiltonian modified energy
def energy(E, decomp, mol, sweep=False):
    mf = scf.RHF(mol)
    ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    h1 = mf.get_hcore()
    field = jnp.einsum('x,xij->ij', E, ao_dip)
    mf.get_hcore = lambda *args, **kwargs: h1 + field
    mf.kernel()
    assert mf.converged, 'mf not converged'
    
    # Localize here
    mo_coeff, mo_occ = loc_orbs(mol, mf, decomp.mo_basis, decomp.pop_method,
                        decomp.mo_init, decomp.loc_exp, decomp.verbose, sweep)
    
    ad = True
    e_part = decodense.main(mol = mol, decomp = decomp, mf = mf, mo_coeff = mo_coeff, mo_occ = mo_occ, AD = ad, ext = field)

    # dimension of the electronic part needs to be considered
    e_tot = e_part[decodense.decomp.CompKeys.el] + e_part[decodense.decomp.CompKeys.struct]
    return e_tot
