from pyscfad import gto, scf, dft, lo
from pyscfad.lo import pipek, iao, orth, boys
from jax import numpy as jnp
from typing import Dict, Tuple, List, Union, Optional, Any
import numpy as np
from pyscfad.ops import stop_grad
from functools import reduce

LOC_TOL = 1.e-10


def loc_orbs(mol: gto.Mole, mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT], \
             mo_basis: str, pop_method: str, mo_init: str, loc_exp: int, \
             verbose: int, sweep: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        this function returns a set of localized MOs of a specific variant
        """
        
        # localized pm orbitals or ibos
        
        orbocc, mo_occ = mf_info(mf)
        
        # rhf reference
        if mo_occ[0].size == mo_occ[1].size:
            rhf = jnp.allclose(orbocc[0], orbocc[1]) and jnp.allclose(mo_occ[0], mo_occ[1])
        else:
            rhf = False

        # overlap matrix
        s = mol.intor_symmetric('int1e_ovlp')

        # molecular dimensions
        alpha, beta = dim(mo_occ)

        # init mo_coeff_out
        mo_coeff_out = (jnp.zeros_like(orbocc[0]), jnp.zeros_like(orbocc[1]))

        # loop over spins
        for i, spin_mo in enumerate((alpha, beta)):

            # localize orbitals
            if mo_basis == 'fb':
                # foster-boys MOs
                orbloc = lo.boys.boys(mol, orbocc[i], conv_tol=LOC_TOL)
                new_array = mo_coeff_out[i].at[..., spin_mo].set(orbloc)
                mo_coeff_out = mo_coeff_out[:i] + (new_array,) + mo_coeff_out[i+1:]
                
            elif mo_basis == 'can':
                return orbocc, mo_occ
                
            elif mo_basis == 'pm':
                print("pipek-mezey procedure with given pop_method")
                
                if sweep:
                    u0 = pm_jacobi_sweep(mol, orbocc[i], mf.get_ovlp(), pop_method, loc_exp)
                    orbloc = pipek.pm(mol, orbocc[i], conv_tol=LOC_TOL,
                                      pop_method=pop_method, exponent=loc_exp, init_guess=u0)
                else: 
                    orbloc = pipek.pm(mol, orbocc[i], conv_tol=LOC_TOL,
                                    pop_method=pop_method, exponent=loc_exp, init_guess="atomic")
                                
                new_array = mo_coeff_out[i].at[..., spin_mo].set(orbloc)
                mo_coeff_out = mo_coeff_out[:i] + (new_array,) + mo_coeff_out[i+1:]
            
            else:
                raise NotImplementedError("mo_basis {} not implemented".format(mo_basis))

            # closed-shell reference
            if rhf:
                new_array = mo_coeff_out[i+1].at[..., spin_mo].set(mo_coeff_out[i][:, spin_mo])
                mo_coeff_out = mo_coeff_out[:i+1] + (new_array,) + mo_coeff_out[i+2:]
                break

        return mo_coeff_out, mo_occ
    
    
def dim(mo_occ: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    determine molecular dimensions
    """
    if isinstance(mo_occ, jnp.ndarray) or isinstance(mo_occ, np.ndarray) and mo_occ.ndim == 1:
        return jnp.where(jnp.abs(mo_occ) > 0.)[0], jnp.where(jnp.abs(mo_occ) > 1.)[0]
    else:
        return jnp.where(jnp.abs(mo_occ[0]) > 0.)[0], jnp.where(jnp.abs(mo_occ[1]) > 0.)[0]
    
    
def mf_info(mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT]) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], \
                                                             Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    retrieve mf information (mo coefficients & occupations)
    """
    
    # dimensions
    alpha, beta = dim(mf.mo_occ)
    # mo occupations
    mo_occ = (jnp.ones_like(alpha), jnp.ones_like(beta))
    # mo coefficients
    if jnp.asarray(mf.mo_coeff).ndim == 2:
        mo_coeff = (mf.mo_coeff[:, alpha], mf.mo_coeff[:, beta])
    else:
        mo_coeff = (mf.mo_coeff[0][:, alpha], mf.mo_coeff[1][:, beta])
        
    return jnp.asarray(mo_coeff), mo_occ

def pm_jacobi_sweep(mol, orbocc, s1e, pop_method, exponent=2, conv_tol=LOC_TOL):
    orbocc = np.asarray(stop_grad(orbocc))
    mlo = pipek.PM(mol, orbocc)
    mlo.pop_method = pop_method
    mlo.exponent = exponent
    mlo.conv_tol = conv_tol
    _ = mlo.kernel()
    mlo = pipek.jacobi_sweep(mlo)
    orbloc = mlo.mo_coeff
    u0 = orbocc.T @ s1e @ orbloc
    return u0
