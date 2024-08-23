#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
orbitals module
"""

__author__ = 'Janus Juul Eriksen, Technical University of Denmark, DK'
__maintainer__ = 'Janus Juul Eriksen'
__email__ = 'janus@kemi.dtu.dk'
__status__ = 'Development'

import numpy as np
from jax import numpy as jnp
from pyscfad import gto, scf, dft, lo, lib
from pyscfad.lo import pipek, iao, orth, boys
from typing import List, Tuple, Dict, Union, Any

from .tools import dim, make_rdm1, contract, mf_info

LOC_TOL = 1.e-10


def loc_orbs(mol: gto.Mole, mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT], \
             mo_coeff_in: jnp.ndarray, mo_occ: jnp.ndarray, \
             mo_basis: str, pop_method: str, mo_init: str, loc_exp: int, \
             ndo: bool, verbose: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        this function returns a set of localized MOs of a specific variant
        """

        orbocc, mo_occ = mf_info(mf)

        # rhf reference
        if mo_occ[0].size == mo_occ[1].size:
            rhf = jnp.allclose(mo_coeff_in[0], mo_coeff_in[1]) and jnp.allclose(mo_occ[0], mo_occ[1])
        else:
            rhf = False

        # ndo assertion
        if ndo:
            raise NotImplementedError('localization of NDOs is not implemented')

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
                orbloc = lo.boys.boys(mol, orbocc[i], conv_tol=LOC_TOL)
                new_array = mo_coeff_out[i].at[..., spin_mo].set(orbloc)
                mo_coeff_out = mo_coeff_out[:i] + (new_array,) + mo_coeff_out[i+1:]
              
            elif mo_basis == 'can':
                return orbocc, mo_occ

            elif mo_basis == 'pm':
                # pipek-mezey procedure with given pop_method
                print("pipek-mezey procedure with given pop_method")

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


def assign_rdm1s(mol: gto.Mole, mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT], \
                 mo_coeff: jnp.ndarray, mo_occ: jnp.ndarray, pop_method: str, part: str, ndo: bool, \
                 verbose: int, **kwargs: Any) -> List[jnp.ndarray]:
        """
        this function returns a list of population weights of each spin-orbital on the individual atoms
        """

        # rhf reference
        if mo_occ[0].size == mo_occ[1].size:
            rhf = jnp.allclose(mo_coeff[0], mo_coeff[1]) and jnp.allclose(mo_occ[0], mo_occ[1])
        else:
            rhf = False

        # molecular dimensions
        alpha, beta = dim(mo_occ)

        # max number of occupied spin-orbs
        n_spin = max(alpha.size, beta.size)

        # mol object projected into minao basis
        if pop_method == 'iao':
            # ndo assertion
            if ndo:
                raise NotImplementedError('IAO-based populations for NDOs is not implemented')
            pmol = lo.iao.reference_mol(mol)
        else:
            pmol = mol

        # init population weights array
        weights = [jnp.zeros([n_spin, pmol.natm], dtype=jnp.float64), jnp.zeros([n_spin, pmol.natm], dtype=jnp.float64)]

        # loop over spin
        for i, spin_mo in enumerate((alpha, beta)):

            pops = pipek.atomic_pops(mol, mo_coeff[i], method = pop_method)
            weights[i] = pops.diagonal(axis1=1, axis2=2).transpose()
        
            # closed-shell reference
            if rhf:
                weights[i+1] = weights[i]
                break

        # verbose print
        if 0 < verbose:
            symbols = tuple(pmol.atom_pure_symbol(i) for i in range(pmol.natm))
            print('\n *** partial population weights: ***')
            print(' spin  ' + 'MO       ' + '      '.join(['{:}'.format(i) for i in symbols]))
            for i, spin_mo in enumerate((alpha, beta)):
                for j in spin_mo:
                    with jnp.printoptions(suppress=True, linewidth=200, formatter={'float': '{:6.3f}'.format}):
                        print('  {:s}    {:>2d}   {:}'.format('a' if i == 0 else 'b', j, weights[i][j]))
            with jnp.printoptions(suppress=True, linewidth=200, formatter={'float': '{:6.3f}'.format}):
                print('   total    {:}'.format(jnp.sum(weights[0], axis=0) + jnp.sum(weights[1], axis=0)))

        return weights


def _population_mul(natm: int, ao_labels: jnp.ndarray, ovlp: jnp.ndarray, rdm1: jnp.ndarray) -> jnp.ndarray:
        """
        this function returns the mulliken populations on the individual atoms
        """
        # mulliken population array
        pop = contract('ij,ji->i', rdm1, ovlp)
        # init populations
        populations = jnp.zeros(natm)

        # loop over AOs
        for i, k in enumerate(ao_labels):
            populations[k[0]] += pop[i]

        return populations


def _population_becke(natm: int, charge_matrix: jnp.ndarray, orb: jnp.ndarray) -> jnp.ndarray:
        """
        this function returns the becke populations on the individual atoms
        """
        # init populations
        populations = jnp.zeros(natm)

        # loop over atoms
        for i in range(natm):
            populations[i] = contract('ki,kl,lj->ij', orb, charge_matrix[i], orb)

        return populations

# orbital generation wrapper
def gen_orbs(mol, mf, decomp):
        """
        this function generates orbitals
        """
        # get orbitals and mo occupation

        mo_coeff, mo_occ = mf_info(mf)
        mo_coeff, mo_occ = loc_orbs(mol, mf, mo_coeff, mo_occ, \
                            decomp.mo_basis, decomp.pop_method, decomp.mo_init, decomp.loc_exp, \
                            decomp.ndo, decomp.verbose)
        return mo_coeff, mo_occ