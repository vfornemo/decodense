from pyscfad import gto, scf, dft, lo
from pyscfad.lo import pipek, iao, orth, boys
from pyscfad.lib import numpy as jnp
from typing import Dict, Tuple, List, Union, Optional, Any
import numpy as np
from pyscfad.lib import stop_grad
from functools import reduce

LOC_TOL = 1.e-10


def loc_orbs(mol: gto.Mole, mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT], \
            #  mo_coeff_in: jnp.ndarray, mo_occ: jnp.ndarray, \
             mo_basis: str, pop_method: str, mo_init: str, loc_exp: int, \
             verbose: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        this function returns a set of localized MOs of a specific variant
        """
        
        mo_coeff_in, mo_occ = mf_info(mf)
        
        # rhf reference
        if mo_occ[0].size == mo_occ[1].size:
            rhf = jnp.allclose(mo_coeff_in[0], mo_coeff_in[1]) and jnp.allclose(mo_occ[0], mo_occ[1])
        else:
            rhf = False

        # overlap matrix
        s = mol.intor_symmetric('int1e_ovlp')

        # molecular dimensions
        alpha, beta = dim(mo_occ)

        # init mo_coeff_out
        mo_coeff_out = (jnp.zeros_like(mo_coeff_in[0]), jnp.zeros_like(mo_coeff_in[1]))

        # loop over spins
        for i, spin_mo in enumerate((alpha, beta)):

            # construct start guess
            if mo_init == 'can':
                # canonical MOs as start guess
                mo_coeff_init = mo_coeff_in[i][:, spin_mo]
            elif mo_init == 'cholesky':
                # start guess via non-iterative cholesky factorization
                mo_coeff_init = lo.cholesky.cholesky_mos(mo_coeff_in[i][:, spin_mo])
            else:
                # IBOs as start guess
                mo_coeff_init = lo.ibo.ibo(mol, mo_coeff_in[i][:, spin_mo], exponent=loc_exp, verbose=0)

            # localize orbitals
            if mo_basis == 'fb':
                # foster-boys MOs
                # loc = lo.Boys(mol)
                # loc.conv_tol = LOC_TOL
                # if 0 < verbose: loc.verbose = 4
                # mo_coeff_out[i][:, spin_mo] = loc.kernel(mo_coeff_init)
                new_array = mo_coeff_out[i].at[..., spin_mo].set(lo.boys.boys(mol, mo_coeff_init, conv_tol = LOC_TOL))
                mo_coeff_out = mo_coeff_out[:i] + (new_array,) + mo_coeff_out[i+1:]
                
            elif mo_basis == 'can':
                return mo_coeff_in, mo_occ
                
            elif mo_basis == 'pm':
                print("pipek-mezey procedure with given pop_method")
                                
                # loc = pipek.pm(mol, mo_coeff_init, init_guess = mo_coeff_init, \
                #                pop_method = pop_method, exponent = loc_exp, conv_tol = LOC_TOL)
                
                # orbocc = pm_jacobi_sweep(mol, mf.mo_coeff, mf.mo_occ, mf.get_ovlp(), pop_method)
                # loc = pipek.pm(mol, orbocc,
                #             pop_method = pop_method, conv_tol = LOC_TOL)
                
                loc = pipek.pm(mol, mo_coeff_init, pop_method = pop_method, exponent = loc_exp, conv_tol = LOC_TOL)
                
                new_array = mo_coeff_out[i].at[..., spin_mo].set(loc)
                mo_coeff_out = mo_coeff_out[:i] + (new_array,) + mo_coeff_out[i+1:]
            
            else:
                raise NotImplementedError("mo_basis {} not implemented".format(mo_basis))

            # closed-shell reference
            if rhf:
                # mo_coeff_out[i+1][:, spin_mo] = mo_coeff_out[i][:, spin_mo]
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
    # print("mf.mo_coeff", mf.mo_coeff)
    
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


def pm_jacobi_sweep(mol, mo_coeff, mo_occ, s1e, pop_method, conv_tol=LOC_TOL):
    orbocc = np.asarray(stop_grad(mo_coeff[:, mo_occ>0]))
    mlo = pipek.PM(stop_grad(mol), orbocc)
    mlo.pop_method = pop_method
    mlo.conv_tol = conv_tol
    _ = mlo.kernel()
    mlo = pipek.jacobi_sweep(mlo)
    orbloc = mlo.mo_coeff
    u0 = reduce(np.dot, (orbocc.T, stop_grad(s1e), orbloc))
    return jnp.dot(mo_coeff[:, mo_occ>0], u0)

