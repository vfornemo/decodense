#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
results module containing all functions related to printing the results of an mf_decomp calculation
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import numpy as np
from pyscf import gto, lib


def results(system, mol, e_hf, e_hf_loc, e_dft, e_dft_loc, centres_hf, centres_dft, \
            e_nuc, e_xc, e_hf_ref, e_dft_ref):
    """
    this function prints the results of an mf_decomp calculation

    :param system: system info. dict
    :param mol: pyscf mol object
    :param e_hf: canonical hf decomposed results. numpy array of shape (nocc,)
    :param e_hf_loc: localized hf decomposed results. numpy array of shape (nocc,)
    :param e_hf: canonical dft decomposed results. numpy array of shape (nocc,)
    :param e_dft_loc: localized hf decomposed results. numpy array of shape (nocc,)
    :param centres_hf: centre assignments for localized hf results. numpy array of shape (nocc, 2) [*strings]
    :param centres_dft: centre assignments for localized dft results. numpy array of shape (nocc, 2) [*strings]
    :param e_nuc: nuclear repulsion energy. scalar
    :param e_xc: exchange-correlation energy. scalar
    :param e_hf_ref: reference hf energy. scalar
    :param e_dft_ref: reference dft energy. scalar
    """
    # inter-atomic distance array
    rr = gto.mole.inter_distance(mol) * lib.param.BOHR

    # print header
    print('\n\n results for: {:} with localization procedure: {:}'. \
            format(system['molecule'], system['loc_proc']))


    # print hf results
    print('\n\n hartree-fock\n')
    print('  MO  |   canonical   |   localized   |     atom(s)    |   bond length')
    print('------------------------------------------------------------------------')
    for i in range(mol.nocc):

        # core or valence orbital(s)
        core = centres_hf[i, 0] == centres_hf[i, 1]

        print('  {:>2d}  | {:>10.3f}    | {:>10.3f}    |{:^15s}| {:>10s}'. \
                format(i, e_hf[i], e_hf_loc[i], \
                       mol.atom_symbol(centres_hf[i, 0]) if core else '{:s} & {:s}'. \
                       format(mol.atom_symbol(centres_hf[i, 0]), mol.atom_symbol(centres_hf[i, 1])), \
                       '' if core else '{:>.3f}'. \
                        format(rr[centres_hf[i, 0], centres_hf[i, 1]])))

    print('------------------------------------------------------------------------')
    print('------------------------------------------------------------------------')
    print('  sum | {:>10.3f}    | {:>10.3f}    |'. \
            format(np.sum(e_hf), np.sum(e_hf_loc)))
    print('------------------------------------------------------------------------')
    print('  nuc | {:>+10.3f}    | {:>+10.3f}    |'. \
            format(e_nuc, e_nuc))
    print('------------------------------------------------------------------------')
    print('------------------------------------------------------------------------')
    print('  sum | {:>12.5f}  | {:>12.5f}  |'. \
            format(np.sum(e_hf) + e_nuc, np.sum(e_hf_loc) + e_nuc))
    print('\n *** HF reference energy  = {:.5f}'. \
            format(e_hf_ref))


    # print dft results
    print('\n\n dft ({:s})\n'.format(system['xc_func']))
    print('  MO  |   canonical   |   localized   |     atom(s)    |   bond length')
    print('------------------------------------------------------------------------')
    for i in range(mol.nocc):

        # core or valence orbital(s)
        core = centres_dft[i, 0] == centres_dft[i, 1]

        print('  {:>2d}  | {:>10.3f}    | {:>10.3f}    |{:^15s}| {:>10s}'. \
                format(i, e_dft[i], e_dft_loc[i], \
                       mol.atom_symbol(centres_dft[i, 0]) if core else '{:s} & {:s}'. \
                       format(mol.atom_symbol(centres_dft[i, 0]), mol.atom_symbol(centres_dft[i, 1])), \
                       '' if core else '{:>.3f}'. \
                        format(rr[centres_dft[i, 0], centres_dft[i, 1]])))

    print('------------------------------------------------------------------------')
    print('------------------------------------------------------------------------')
    print('  sum | {:>10.3f}    | {:>10.3f}    |'. \
            format(np.sum(e_dft), np.sum(e_dft_loc)))
    print('------------------------------------------------------------------------')
    print('  nuc | {:>+10.3f}    | {:>+10.3f}    |'. \
            format(e_nuc, e_nuc))
    print('  xc  | {:>+10.3f}    | {:>+10.3f}    |'. \
            format(e_xc, e_xc))
    print('------------------------------------------------------------------------')
    print('------------------------------------------------------------------------')
    print('  sum | {:>12.5f}  | {:>12.5f}  |'. \
            format(np.sum(e_dft) + e_nuc + e_xc, np.sum(e_dft_loc) + e_nuc + e_xc))
    print('\n *** DFT reference energy = {:.5f}\n\n'. \
            format(e_dft_ref))

