#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
main mf_decomp program
"""

__author__ = 'Janus Juul Eriksen, Technical University of Denmark, DK'
__maintainer__ = 'Janus Juul Eriksen'
__email__ = 'janus@kemi.dtu.dk'
__status__ = 'Development'

import numpy as np
from jax import numpy as jnp
import pandas as pd
from pyscfad import gto, scf, dft
from typing import Dict, Tuple, List, Union, Optional, Any

from .decomp import DecompCls, sanity_check
from .orbitals import loc_orbs, assign_rdm1s
from .properties import prop_tot
from .tools import make_natorb, mf_info, write_rdm1
from .results import fmt
from .ad.properties_ad import prop_tot_ad


def main(mol: gto.Mole, decomp: DecompCls, \
         mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT], \
         mo_coeff: jnp.ndarray = None, \
         mo_occ: jnp.ndarray = None,
         rdm1_orb: jnp.ndarray = None, \
         rdm1_eff: jnp.ndarray = None, AD: bool = False) -> pd.DataFrame:
        """
        main decodense program
        """
        # sanity check
        sanity_check(mol, decomp)

        # compute population weights
        weights = assign_rdm1s(mol, mf, mo_coeff, mo_occ, decomp.pop_method, decomp.part, \
                               decomp.ndo, decomp.verbose)

        # compute decomposed results
        
        if AD == True:
            decomp.res = prop_tot_ad(mol, mf, mo_coeff, mo_occ, rdm1_eff, \
                              decomp.pop_method, decomp.prop, decomp.part, \
                              decomp.ndo, decomp.gauge_origin, weights)
        else:
            decomp.res = prop_tot(mol, mf, mo_coeff, mo_occ, rdm1_eff, \
                                  decomp.pop_method, decomp.prop, decomp.part, \
                                  decomp.ndo, decomp.gauge_origin, weights)
        

        # write rdm1s
        if decomp.write != '':
            write_rdm1(mol, decomp.part, mo_coeff, mo_occ, decomp.write, weights)

        if AD == True:
            return decomp.res
        else:
            return fmt(mol, decomp.res, decomp.unit, decomp.ndo)

