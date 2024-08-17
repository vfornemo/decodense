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
                loc = lo.Boys(mol)
                loc.conv_tol = LOC_TOL
                if 0 < verbose: loc.verbose = 4
                mo_coeff_out[i][:, spin_mo] = loc.kernel(mo_coeff_init)

            elif mo_basis == 'can':
                return orbocc, mo_occ

            elif mo_basis == 'pm':
                # pipek-mezey procedure with given pop_method
                print("pipek-mezey procedure with given pop_method")

                orbloc = pipek.pm(mol, orbocc[i], conv_tol=LOC_TOL,
                                pop_method=pop_method, exponent=loc_exp, init_guess="atomic")
                   
                new_array = mo_coeff_out[i].at[..., spin_mo].set(orbloc)
                mo_coeff_out = mo_coeff_out[:i] + (new_array,) + mo_coeff_out[i+1:]
                
                # loc = lo.PM(mol, mf=mf)
                # loc.conv_tol = LOC_CONV
                # loc.pop_method = pop_method
                # loc.exponent = loc_exp
                # if 0 < verbose: loc.verbose = 4
                # mo_coeff_out[i][:, spin_mo] = loc.kernel(mo_coeff_init)
            
            else:
                raise NotImplementedError("mo_basis {} not implemented".format(mo_basis))

            # closed-shell reference
            if rhf:
                # mo_coeff_out[i+1][:, spin_mo] = mo_coeff_out[i][:, spin_mo]
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
        # # declare nested kernel function in global scope
        # global get_weights

        # # dft logical
        # dft_calc = isinstance(mf, dft.rks.KohnShamDFT)

        # rhf reference
        if mo_occ[0].size == mo_occ[1].size:
            rhf = jnp.allclose(mo_coeff[0], mo_coeff[1]) and jnp.allclose(mo_occ[0], mo_occ[1])
        else:
            rhf = False

        # # overlap matrix
        # s = mol.intor_symmetric('int1e_ovlp')

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

        # # number of atoms
        # natm = pmol.natm

        # # AO labels
        # ao_labels = pmol.ao_labels(fmt=None)

        # # overlap matrix
        # if pop_method == 'mulliken':
        #     ovlp = s
        # else:
        #     ovlp = jnp.eye(pmol.nao_nr())

        # def get_weights(orb_idx: int):
        #     """
        #     this function computes the full set of population weights
        #     """
        #     # get orbital
        #     orb = mo[:, orb_idx].reshape(mo.shape[0], 1)
        #     if pop_method == 'becke':
        #         # population weights of orb
        #         return _population_becke(natm, charge_matrix, orb)
        #     else:
        #         # orbital-specific rdm1
        #         rdm1_orb = make_rdm1(orb, mocc[orb_idx])
        #         # population weights of rdm1_orb
        #         return _population_mul(natm, ao_labels, ovlp, rdm1_orb)

        # init population weights array
        weights = [jnp.zeros([n_spin, pmol.natm], dtype=jnp.float64), jnp.zeros([n_spin, pmol.natm], dtype=jnp.float64)]

        # loop over spin
        for i, spin_mo in enumerate((alpha, beta)):

            # # get mo coefficients and occupation
            # if pop_method == 'mulliken':
            #     mo = mo_coeff[i][:, spin_mo]
            # elif pop_method == 'lowdin':
            #     mo = contract('ki,kl,lj->ij', lo.orth.orth_ao(pmol, method='lowdin', s=s), s, mo_coeff[i][:, spin_mo])
            # elif pop_method == 'meta_lowdin':
            #     mo = contract('ki,kl,lj->ij', lo.orth.orth_ao(pmol, method='meta_lowdin', s=s), s, mo_coeff[i][:, spin_mo])
            # elif pop_method == 'iao':
            #     iao = lo.iao.iao(mol, mo_coeff[i][:, spin_mo])
            #     iao = lo.orth.vec_lowdin(iao, s)
            #     # iao = lo.vec_lowdin(iao, s)
            #     mo = contract('ki,kl,lj->ij', iao, s, mo_coeff[i][:, spin_mo])
            # elif pop_method == 'becke':
            #     if getattr(pmol, 'pbc_intor', None):
            #         raise NotImplementedError('PM becke scheme for PBC systems')
            #     if dft_calc:
            #         grid_coords, grid_weights = mf.grids.get_partition(mol, concat=False)
            #         ni = mf._numint
            #     else:
            #         mf_becke = mol.RKS()
            #         grid_coords, grid_weights = mf_becke.grids.get_partition(mol, concat=False)
            #         ni = mf_becke._numint
            #     charge_matrix = jnp.zeros([natm, pmol.nao_nr(), pmol.nao_nr()], dtype=jnp.float64)
            #     for j in range(natm):
            #         ao = ni.eval_ao(mol, grid_coords[j], deriv=0)
            #         aow = jnp.einsum('pi,p->pi', ao, grid_weights[j])
            #         charge_matrix[j] = contract('ki,kj->ij', aow, ao)
            #     mo = mo_coeff[i][:, spin_mo]
            # mocc = mo_occ[i][spin_mo]

            # domain
            # domain = jnp.arange(spin_mo.size)
            # execute kernel
            # weights[i] = list(map(get_weights, domain)) # type: ignore
            
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
