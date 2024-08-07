#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
properties module
"""

__author__ = 'Janus Juul Eriksen, Technical University of Denmark, DK'
__maintainer__ = 'Janus Juul Eriksen'
__email__ = 'janus@kemi.dtu.dk'
__status__ = 'Development'

import numpy as np
from jax import numpy as jnp
from itertools import starmap
from pyscfad import gto, scf, dft, df, lo, lib
from pyscf import solvent
from pyscf.dft import numint
from pyscf import tools as pyscf_tools
from typing import List, Tuple, Dict, Union, Any

from .tools import dim, make_rdm1, orbsym, contract
from .decomp import CompKeys

# block size in _mm_pot()
BLKSIZE = 200


def prop_tot(mol: gto.Mole, mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT], \
             mo_coeff: Tuple[jnp.ndarray, jnp.ndarray], mo_occ: Tuple[jnp.ndarray, jnp.ndarray], \
             rdm1_eff: jnp.ndarray, pop_method: str, prop_type: str, part: str, ndo: bool, \
             gauge_origin: jnp.ndarray, weights: List[jnp.ndarray]) -> Dict[str, Union[jnp.ndarray, List[jnp.ndarray]]]:
        """
        this function returns atom-decomposed mean-field properties
        """
        # declare nested kernel functions in global scope
        global prop_atom
        global prop_eda
        global prop_orb

        # dft logical
        dft_calc = isinstance(mf, dft.rks.KohnShamDFT)

        # ao dipole integrals with specified gauge origin
        if prop_type == 'dipole':
            with mol.with_common_origin(gauge_origin):
                ao_dip = mol.intor_symmetric('int1e_r', comp=3)
        else:
            ao_dip = None

        # compute total 1-RDMs (AO basis)
        if rdm1_eff is None:
            rdm1_eff = jnp.array([make_rdm1(mo_coeff[0], mo_occ[0]), make_rdm1(mo_coeff[1], mo_occ[1])])
        if rdm1_eff.ndim == 2:
            rdm1_eff = jnp.array([rdm1_eff, rdm1_eff]) * .5
        rdm1_tot = jnp.array([make_rdm1(mo_coeff[0], mo_occ[0]), make_rdm1(mo_coeff[1], mo_occ[1])])

        # mol object projected into minao basis
        if pop_method == 'iao':
            pmol = lo.iao.reference_mol(mol)
        else:
            pmol = mol

        # effective atomic charges
        if part in ['atoms', 'eda']:
            charge_atom = -jnp.sum(weights[0] + weights[1], axis=0) + pmol.atom_charges()
        else:
            charge_atom = 0.

        # possible mm region
        mm_mol = getattr(mf, 'mm_mol', None)

        # possible cosmo/pcm solvent model
        if getattr(mf, 'with_solvent', None):
            e_solvent = _solvent(mol, jnp.sum(rdm1_eff, axis=0), mf.with_solvent)
        else:
            e_solvent = None

        # nuclear repulsion property
        if prop_type == 'energy':
            prop_nuc_rep = _e_nuc(pmol, mm_mol)
        elif prop_type == 'dipole':
            prop_nuc_rep = _dip_nuc(pmol, charge_atom, gauge_origin)

        # core hamiltonian
        kin, nuc, sub_nuc, mm_pot = _h_core(mol, mm_mol)
        # fock potential
        vj, vk = mf.get_jk(mol=mol, dm=rdm1_eff)

        # calculate xc energy density
        if dft_calc:
            # ndo assertion
            if ndo:
                raise NotImplementedError('NDOs for KS-DFT do not yield a lossless decomposition')
            # xc-type and ao_deriv
            xc_type, ao_deriv = _xc_ao_deriv(mf.xc)
            # update exchange operator wrt range-separated parameter and exact exchange components
            vk = _vk_dft(mol, mf, mf.xc, rdm1_eff, vk)
            # ao function values on given grid
            ao_value = _ao_val(mol, mf.grids.coords, ao_deriv)
            # grid weights
            grid_weights = mf.grids.weights
            # compute all intermediates
            c0_tot, c1_tot, rho_tot = _make_rho(ao_value, rdm1_eff, xc_type)
            # evaluate xc energy density
            eps_xc = dft.libxc.eval_xc(mf.xc, rho_tot, spin=0 if isinstance(rho_tot, jnp.ndarray) else -1)[0]
            # nlc (vv10)
            if mf.nlc.upper() == 'VV10':
                nlc_pars = dft.libxc.nlc_coeff(mf.xc)[0][0]
                ao_value_nlc = _ao_val(mol, mf.nlcgrids.coords, 1)
                grid_weights_nlc = mf.nlcgrids.weights
                c0_vv10, c1_vv10, rho_vv10 = _make_rho(ao_value_nlc, jnp.sum(rdm1_eff, axis=0), 'GGA')
                eps_xc_nlc = numint._vv10nlc(rho_vv10, mf.nlcgrids.coords, rho_vv10, \
                                             grid_weights_nlc, mf.nlcgrids.coords, nlc_pars)[0]
            else:
                eps_xc_nlc = None
        else:
            xc_type = ''
            grid_weights = grid_weights_nlc = None
            ao_value = ao_value_nlc = None
            eps_xc = eps_xc_nlc = None
            c0_tot = c1_tot = None
            c0_vv10 = c1_vv10 = None

        # molecular dimensions
        alpha, beta = dim(mo_occ)

        # atomic labels
        if part == 'eda':
            ao_labels = mol.ao_labels(fmt=None)

        def prop_atom(atom_idx: int) -> Dict[str, Any]:
                """
                this function returns atom-wise energy/dipole contributions
                """
                # init results
                res = {}
                # atom-specific rdm1
                rdm1_atom = jnp.zeros_like(rdm1_tot)
                # loop over spins
                if prop_type == 'energy':
                    res[CompKeys.coul] = 0.
                    res[CompKeys.exch] = 0.
                for i, spin_mo in enumerate((alpha, beta)):
                    # loop over spin-orbitals
                    for m, j in enumerate(spin_mo):
                        # get orbital(s)
                        orb = mo_coeff[i][:, j].reshape(mo_coeff[i].shape[0], -1)
                        # orbital-specific rdm1
                        rdm1_orb = make_rdm1(orb, mo_occ[i][j])
                        # weighted contribution to rdm1_atom
                        rdm1_atom[i] += rdm1_orb * weights[i][m][atom_idx] / jnp.sum(weights[i][m])
                    # coulumb & exchange energy associated with given atom
                    if prop_type == 'energy':
                        res[CompKeys.coul] += _trace(jnp.sum(vj, axis=0), rdm1_atom[i], scaling = .5)
                        res[CompKeys.exch] -= _trace(vk[i], rdm1_atom[i], scaling = .5)
                # common energy contributions associated with given atom
                if prop_type == 'energy':
                    res[CompKeys.kin] = _trace(kin, jnp.sum(rdm1_atom, axis=0))
                    res[CompKeys.nuc_att_glob] = _trace(sub_nuc[atom_idx], jnp.sum(rdm1_tot, axis=0), scaling = .5)
                    res[CompKeys.nuc_att_loc] = _trace(nuc, jnp.sum(rdm1_atom, axis=0), scaling = .5)
                    if mm_pot is not None:
                        res[CompKeys.solvent] = _trace(mm_pot, jnp.sum(rdm1_atom, axis=0))
                    if e_solvent is not None:
                        res[CompKeys.solvent] = e_solvent[atom_idx]
                    # additional xc energy contribution
                    if dft_calc:
                        # atom-specific rho
                        _, _, rho_atom = _make_rho(ao_value, jnp.sum(rdm1_atom, axis=0), xc_type)
                        # energy from individual atoms
                        res[CompKeys.xc] = _e_xc(eps_xc, grid_weights, rho_atom)
                        # nlc (vv10)
                        if eps_xc_nlc is not None:
                            _, _, rho_atom_vv10 = _make_rho(ao_value_nlc, jnp.sum(rdm1_atom, axis=0), 'GGA')
                            res[CompKeys.xc] += _e_xc(eps_xc_nlc, grid_weights_nlc, rho_atom_vv10)
                elif prop_type == 'dipole':
                    res[CompKeys.el] = -_trace(ao_dip, jnp.sum(rdm1_atom, axis=0))
                # sum up electronic contributions
                if prop_type == 'energy':
                    res[CompKeys.el] = sum(res.values())
                return res

        def prop_eda(atom_idx: int) -> Dict[str, Any]:
                """
                this function returns EDA energy/dipole contributions
                """
                # init results
                res = {}
                # get AOs on atom k
                select = jnp.where([atom[0] == atom_idx for atom in ao_labels])[0]
                # common energy contributions associated with given atom
                if prop_type == 'energy':
                    res[CompKeys.coul] = 0.
                    res[CompKeys.exch] = 0.
                    # loop over spins
                    for i, _ in enumerate((alpha, beta)):
                        res[CompKeys.coul] += _trace(jnp.sum(vj, axis=0)[select], rdm1_tot[i][select], scaling = .5)
                        res[CompKeys.exch] -= _trace(vk[i][select], rdm1_tot[i][select], scaling = .5)
                    res[CompKeys.kin] = _trace(kin[select], jnp.sum(rdm1_tot, axis=0)[select])
                    res[CompKeys.nuc_att_glob] = _trace(sub_nuc[atom_idx], jnp.sum(rdm1_tot, axis=0), scaling = .5)
                    res[CompKeys.nuc_att_loc] = _trace(nuc[select], jnp.sum(rdm1_tot, axis=0)[select], scaling = .5)
                    if mm_pot is not None:
                        res[CompKeys.solvent] = _trace(mm_pot[select], jnp.sum(rdm1_tot, axis=0)[select])
                    if e_solvent is not None:
                        res[CompKeys.solvent] = e_solvent[atom_idx]
                    # additional xc energy contribution
                    if dft_calc:
                        # atom-specific rho
                        rho_atom = _make_rho_interm2(c0_tot[:, select], \
                                                     c1_tot if c1_tot is None else c1_tot[:, :, select], \
                                                     ao_value[:, :, select], xc_type)
                        # energy from individual atoms
                        res[CompKeys.xc] = _e_xc(eps_xc, grid_weights, rho_atom)
                        # nlc (vv10)
                        if eps_xc_nlc is not None:
                            rho_atom_vv10 = _make_rho_interm2(c0_vv10[:, select], \
                                                              c1_vv10 if c1_vv10 is None else c1_vv10[:, :, select], \
                                                              ao_value_nlc[:, :, select], 'GGA')
                            res[CompKeys.xc] += _e_xc(eps_xc_nlc, grid_weights_nlc, rho_atom_vv10)
                elif prop_type == 'dipole':
                    res[CompKeys.el] = -_trace(ao_dip[:, select], jnp.sum(rdm1_tot, axis=0)[select])
                # sum up electronic contributions
                if prop_type == 'energy':
                    res[CompKeys.el] = sum(res.values())
                return res

        def prop_orb(spin_idx: int, orb_idx: int) -> Dict[str, Any]:
                """
                this function returns bond-wise energy/dipole contributions
                """
                # init res
                res = {}
                # get orbital(s)
                orb = mo_coeff[spin_idx][:, orb_idx].reshape(mo_coeff[spin_idx].shape[0], -1)
                # orbital-specific rdm1
                rdm1_orb = make_rdm1(orb, mo_occ[spin_idx][orb_idx])
                # total energy or dipole moment associated with given spin-orbital
                if prop_type == 'energy':
                    res[CompKeys.coul] = _trace(jnp.sum(vj, axis=0), rdm1_orb, scaling = .5)
                    res[CompKeys.exch] = -_trace(vk[spin_idx], rdm1_orb, scaling = .5)
                    res[CompKeys.kin] = _trace(kin, rdm1_orb)
                    res[CompKeys.nuc_att] = _trace(nuc, rdm1_orb)
                    if mm_pot is not None:
                        res[CompKeys.solvent] = _trace(mm_pot, rdm1_orb)
                    # additional xc energy contribution
                    if dft_calc:
                        # orbital-specific rho
                        _, _, rho_orb = _make_rho(ao_value, rdm1_orb, xc_type)
                        # xc energy from individual orbitals
                        res[CompKeys.xc] = _e_xc(eps_xc, grid_weights, rho_orb)
                        # nlc (vv10)
                        if eps_xc_nlc is not None:
                            _, _, rho_orb_vv10 = _make_rho(ao_value_nlc, rdm1_orb, 'GGA')
                            res[CompKeys.xc] += _e_xc(eps_xc_nlc, grid_weights_nlc, rho_orb_vv10)
                elif prop_type == 'dipole':
                    res[CompKeys.el] = -_trace(ao_dip, rdm1_orb)
                # sum up electronic contributions
                if prop_type == 'energy':
                    res[CompKeys.el] = sum(res.values())
                return res

        # perform decomposition
        if part in ['atoms', 'eda']:
            # domain
            domain = jnp.arange(pmol.natm)
            # execute kernel
            res = list(map(prop_atom if part == 'atoms' else prop_eda, domain)) # type: ignore
            # init atom-specific energy or dipole arrays
            if prop_type == 'energy':
                prop = {comp_key: jnp.zeros(pmol.natm, dtype=jnp.float64) for comp_key in res[0].keys()}
            elif prop_type == 'dipole':
                prop = {comp_key: jnp.zeros([pmol.natm, 3], dtype=jnp.float64) for comp_key in res[0].keys()}
            # collect results
            for k, r in enumerate(res):
                for key, val in r.items():
                    prop[key][k] = val
            if ndo:
                prop[CompKeys.struct] = jnp.zeros_like(prop_nuc_rep)
            else:
                prop[CompKeys.struct] = prop_nuc_rep
            return {**prop, CompKeys.charge_atom: charge_atom}
        else: # orbs
            # domain
            domain = jnp.array([(i, j) for i, orbs in enumerate((alpha, beta)) for j in orbs])
            # execute kernel
            res = list(starmap(prop_orb, domain)) # type: ignore
            # init orbital-specific energy or dipole array
            if prop_type == 'energy':
                prop = {comp_key: [jnp.zeros(alpha.size), jnp.zeros(beta.size)] for comp_key in res[0].keys()}
            elif prop_type == 'dipole':
                prop = {comp_key: [jnp.zeros([alpha.size, 3], dtype=jnp.float64), \
                                   jnp.zeros([beta.size, 3], dtype=jnp.float64)] for comp_key in res[0].keys()}
            # collect results
            for k, r in enumerate(res):
                for key, val in r.items():
                    prop[key][domain[k, 0]][domain[k, 1]] = val
            if ndo:
                prop[CompKeys.struct] = jnp.zeros_like(prop_nuc_rep)
            else:
                prop[CompKeys.struct] = prop_nuc_rep
            return {**prop, CompKeys.mo_occ: mo_occ, CompKeys.orbsym: orbsym(mol, mo_coeff)}


def _e_nuc(mol: gto.Mole, mm_mol: Union[None, gto.Mole]) -> jnp.ndarray:
        """
        this function returns the nuclear repulsion energy
        """
        # coordinates and charges of nuclei
        coords = mol.atom_coords()
        charges = mol.atom_charges()
        # internuclear distances (with self-repulsion removed)
        dist = gto.inter_distance(mol)
        dist[jnp.diag_indices_from(dist)] = 1e200
        e_nuc = contract('i,ij,j->i', charges, 1. / dist, charges) * .5
        # possible interaction with mm sites
        if mm_mol is not None:
            mm_coords = mm_mol.atom_coords()
            mm_charges = mm_mol.atom_charges()
            for j in range(mol.natm):
                q2, r2 = charges[j], coords[j]
                r = lib.norm(r2 - mm_coords, axis=1)
                e_nuc[j] += q2 * jnp.sum(mm_charges / r)
        return e_nuc


def _dip_nuc(mol: gto.Mole, atom_charges: jnp.ndarray, gauge_origin: jnp.ndarray) -> jnp.ndarray:
        """
        this function returns the nuclear contribution to the molecular dipole moment
        """
        # coordinates and formal/actual charges of nuclei
        coords = mol.atom_coords()
        form_charges = mol.atom_charges()
        act_charges = form_charges - atom_charges
        return contract('i,ix->ix', form_charges, coords) - contract('i,x->ix', act_charges, gauge_origin)


def _h_core(mol: gto.Mole, mm_mol: Union[None, gto.Mole]) -> Tuple[jnp.ndarray, jnp.ndarray, \
                                                                   jnp.ndarray, Union[None, jnp.ndarray]]:
        """
        this function returns the components of the core hamiltonian
        """
        # kinetic integrals
        kin = mol.intor_symmetric('int1e_kin')
        # coordinates and charges of nuclei
        coords = mol.atom_coords()
        charges = mol.atom_charges()
        # individual atomic potentials
        sub_nuc = jnp.zeros([mol.natm, mol.nao_nr(), mol.nao_nr()], dtype=jnp.float64)
        for k in range(mol.natm):
            with mol.with_rinv_origin(coords[k]):
                sub_nuc[k] = -mol.intor('int1e_rinv') * charges[k]
        # total nuclear potential
        nuc = jnp.sum(sub_nuc, axis=0)
        # possible mm potential
        if mm_mol is not None:
            mm_pot = _mm_pot(mol, mm_mol)
        else:
            mm_pot = None
        return kin, nuc, sub_nuc, mm_pot


def _mm_pot(mol: gto.Mole, mm_mol: gto.Mole) -> jnp.ndarray:
        """
        this function returns the full mm potential
        (adapted from: qmmm/itrf.py:get_hcore() in PySCF)
        """
        # settings
        coords = mm_mol.atom_coords()
        charges = mm_mol.atom_charges()
        blksize = BLKSIZE
        # integrals
        intor = 'int3c2e_cart' if mol.cart else 'int3c2e_sph'
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas,
                                             mol._env, intor)
        # compute interaction potential
        mm_pot = 0
        for i0, i1 in lib.prange(0, charges.size, blksize):
            fakemol = gto.fakemol_for_charges(coords[i0:i1])
            j3c = df.incore.aux_e2(mol, fakemol, intor=intor,
                                   aosym='s2ij', cintopt=cintopt)
            mm_pot += jnp.einsum('xk,k->x', j3c, -charges[i0:i1])
        mm_pot = lib.unpack_tril(mm_pot)
        return mm_pot


def _solvent(mol: gto.Mole, rdm1: jnp.ndarray, \
             solvent_model: solvent.ddcosmo.DDCOSMO) -> jnp.ndarray:
        """
        this function return atom-specific PCM/COSMO contributions
        (adapted from: solvent/ddcosmo.py:_get_vind() in PySCF)
        """
        # settings
        r_vdw      = solvent_model._intermediates['r_vdw'     ]
        ylm_1sph   = solvent_model._intermediates['ylm_1sph'  ]
        ui         = solvent_model._intermediates['ui'        ]
        Lmat       = solvent_model._intermediates['Lmat'      ]
        cached_pol = solvent_model._intermediates['cached_pol']
        dielectric = solvent_model.eps
        f_epsilon = (dielectric - 1.) / dielectric if dielectric > 0. else 1.
        # electrostatic potential
        phi = solvent.ddcosmo.make_phi(solvent_model, rdm1, r_vdw, ui, ylm_1sph)
        # X and psi (cf. https://github.com/filippolipparini/ddPCM/blob/master/reference.pdf)
        Xvec = jnp.linalg.solve(Lmat, phi.ravel()).reshape(mol.natm,-1)
        psi  = solvent.ddcosmo.make_psi_vmat(solvent_model, rdm1, r_vdw, \
                                             ui, ylm_1sph, cached_pol, Xvec, Lmat)[0]
        return .5 * f_epsilon * jnp.einsum('jx,jx->j', psi, Xvec)


def _xc_ao_deriv(xc_func: str) -> Tuple[str, int]:
        """
        this function returns the type of xc functional and the level of ao derivatives needed
        """
        xc_type = dft.libxc.xc_type(xc_func)
        if xc_type == 'LDA':
            ao_deriv = 0
        elif xc_type in ['GGA', 'NLC']:
            ao_deriv = 1
        elif xc_type == 'MGGA':
            ao_deriv = 2
        return xc_type, ao_deriv


def _make_rho_interm1(ao_value: jnp.ndarray, \
                      rdm1: jnp.ndarray, xc_type: str) -> Tuple[jnp.ndarray, Union[None, jnp.ndarray]]:
        """
        this function returns the rho intermediates (c0, c1) needed in _make_rho()
        (adpated from: dft/numint.py:eval_rho() in PySCF)
        """
        # determine dimensions based on xctype
        xctype = xc_type.upper()
        if xctype == 'LDA' or xctype == 'HF':
            ngrids, nao = ao_value.shape
        else:
            ngrids, nao = ao_value[0].shape
        # compute rho intermediate based on xctype
        if xctype == 'LDA' or xctype == 'HF':
            c0 = contract('ik,kj->ij', ao_value, rdm1)
            c1 = None
        elif xctype in ('GGA', 'NLC'):
            c0 = contract('ik,kj->ij', ao_value[0], rdm1)
            c1 = None
        else: # meta-GGA
            c0 = contract('ik,kj->ij', ao_value[0], rdm1)
            c1 = jnp.empty((3, ngrids, nao), dtype=jnp.float64)
            for i in range(1, 4):
                c1[i-1] = contract('ik,jk->ij', ao_value[i], rdm1)
        return c0, c1


def _make_rho_interm2(c0: jnp.ndarray, c1: jnp.ndarray, \
                      ao_value: jnp.ndarray, xc_type: str) -> jnp.ndarray:
        """
        this function returns rho from intermediates (c0, c1)
        (adpated from: dft/numint.py:eval_rho() in PySCF)
        """
        # determine dimensions based on xctype
        xctype = xc_type.upper()
        if xctype == 'LDA' or xctype == 'HF':
            ngrids = ao_value.shape[0]
        else:
            ngrids = ao_value[0].shape[0]
        # compute rho intermediate based on xctype
        if xctype == 'LDA' or xctype == 'HF':
            rho = contract('pi,pi->p', ao_value, c0)
        elif xctype in ('GGA', 'NLC'):
            rho = jnp.empty((4, ngrids), dtype=jnp.float64)
            rho[0] = contract('pi,pi->p', c0, ao_value[0])
            for i in range(1, 4):
                rho[i] = contract('pi,pi->p', c0, ao_value[i]) * 2.
        else: # meta-GGA
            rho = jnp.empty((6, ngrids), dtype=jnp.float64)
            rho[0] = contract('pi,pi->p', ao_value[0], c0)
            rho[5] = 0.
            for i in range(1, 4):
                rho[i] = contract('pi,pi->p', c0, ao_value[i]) * 2.
                rho[5] += contract('pi,pi->p', c1[i-1], ao_value[i])
            XX, YY, ZZ = 4, 7, 9
            ao_value_2 = ao_value[XX] + ao_value[YY] + ao_value[ZZ]
            rho[4] = contract('pi,pi->p', c0, ao_value_2)
            rho[4] += rho[5]
            rho[4] *= 2.
            rho[5] *= .5
        return rho


def _make_rho(ao_value: jnp.ndarray, rdm1: jnp.ndarray, \
              xc_type: str) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        this function returns important dft intermediates, e.g., energy density, grid weights, etc.
        """
        # rho corresponding to given 1-RDM
        if rdm1.ndim == 2:
            c0, c1 = _make_rho_interm1(ao_value, rdm1, xc_type)
            rho = _make_rho_interm2(c0, c1, ao_value, xc_type)
        else:
            if jnp.allclose(rdm1[0], rdm1[1]):
                c0, c1 = _make_rho_interm1(ao_value, rdm1[0] * 2., xc_type)
                rho = _make_rho_interm2(c0, c1, ao_value, xc_type)
            else:
                c0, c1 = zip(_make_rho_interm1(ao_value, rdm1[0], xc_type), \
                             _make_rho_interm1(ao_value, rdm1[1], xc_type))
                rho = (_make_rho_interm2(c0[0], c1[0], ao_value, xc_type), \
                       _make_rho_interm2(c0[1], c1[1], ao_value, xc_type))
                c0 = jnp.sum(c0, axis=0)
                if c1[0] is not None:
                    c1 = jnp.sum(c1, axis=0)
                else:
                    c1 = None
        return c0, c1, rho


def _vk_dft(mol: gto.Mole, mf: dft.rks.KohnShamDFT, \
            xc_func: str, rdm1: jnp.ndarray, vk: jnp.ndarray) -> jnp.ndarray:
        """
        this function returns the appropriate dft exchange operator
        """
        # range-separated and exact exchange parameters
        ks_omega, ks_alpha, ks_hyb = mf._numint.rsh_and_hybrid_coeff(xc_func)
        # scale amount of exact exchange
        vk *= ks_hyb
        # range separated coulomb operator
        if abs(ks_omega) > 1e-10:
            vk_lr = mf.get_k(mol, rdm1, omega=ks_omega)
            vk_lr *= (ks_alpha - ks_hyb)
            vk += vk_lr
        return vk


def _ao_val(mol: gto.Mole, grids_coords: jnp.ndarray, ao_deriv: int) -> jnp.ndarray:
        """
        this function returns ao function values on the given grid
        """
        return numint.eval_ao(mol, grids_coords, deriv=ao_deriv)


def _trace(op: jnp.ndarray, rdm1: jnp.ndarray, scaling: float = 1.) -> Union[float, jnp.ndarray]:
        """
        this function returns the trace between an operator and an rdm1
        """
        if op.ndim == 2:
            return contract('ij,ij', op, rdm1) * scaling
        else:
            return contract('xij,ij->x', op, rdm1) * scaling


def _e_xc(eps_xc: jnp.ndarray, grid_weights: jnp.ndarray, rho: jnp.ndarray) -> float:
        """
        this function returns a contribution to the exchange-correlation energy from given rmd1 (via rho)
        """
        return contract('i,i,i->', eps_xc, rho if rho.ndim == 1 else rho[0], grid_weights)


