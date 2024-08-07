#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy

import jax
from jax import jacrev
from jax import numpy as jnp

from pyscfad.ops import stop_grad
from pyscfad import gto, scf
from pyscfad.lo import pipek
from pyscfad import config

config.update('pyscfad_scf_implicit_diff', True)


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

def assign_rdm1s(mol, mo_coeff, pop_method):
    """
    list of population weights of each spin-orbital on the individual atoms
    """
    pops = pipek.atomic_pops(mol, mo_coeff, method = pop_method)
    return pops.diagonal(axis1=1, axis2=2).T

def prop_tot(mol, mf, mo_coeff, mo_occ, weights, ext):
    """
    atom-decomposed energy
    """
    # total 1-rdm
    rdm1_tot = jnp.einsum('ip,jp->ij', mo_occ * mo_coeff, mo_coeff)

    # core hamiltonian
    kin, nuc, sub_nuc = _h_core(mf, mol)
    # fock potential
    vj, vk = mf.get_jk(mol=mol, dm=rdm1_tot)

    def prop_atom(weights):
        """
        this function returns atomic contributions
        """
        # atom-specific rdm1
        def _atom_rdm1(mo_coeff, mo_occ, weights):
            # orbital-specific rdm1
            rdm1_orb = jnp.outer(mo_occ * mo_coeff, mo_coeff)
            # weighted contribution to rdm1_atom
            return rdm1_orb * weights

        # atomic 1-rdms
        rdm1_atom = jnp.sum(jax.vmap(_atom_rdm1, (1,0,0))(mo_coeff, mo_occ, weights), axis=0)
        # energy contributions
        kin_el = jnp.einsum('ij,ji', kin, rdm1_atom)
        nuc_att_loc_el = jnp.einsum('ij,ji', nuc, rdm1_atom) * .5
        coul_el = jnp.einsum('ij,ji', vj, rdm1_atom) * .5
        exch_el = -jnp.einsum('ij,ji', vk, rdm1_atom) * .25
        ext_el = jnp.einsum('ij,ji', ext, rdm1_atom)
        return kin_el + nuc_att_loc_el + coul_el + exch_el + ext_el

    # perform decomposition
    res = jax.vmap(prop_atom, (1,))(weights)
    # add global nuc-el attraction
    return res + jnp.einsum('xij,ji->x', sub_nuc, rdm1_tot) * .5

def _h_core(mf, mol):
    """
    this function returns the components of the core hamiltonian
    """
    # kinetic integrals
    kin = mol.intor_symmetric('int1e_kin')
    # charges of nuclei
    charges = mol.atom_charges()
    # individual atomic potentials
    natm = mol.natm
    nao = mol.nao
    sub_nuc = jnp.zeros((natm, nao, nao), dtype=float)
    for k in range(natm):
        with mol.with_rinv_at_nucleus(k):
            sub_nuc = sub_nuc.at[k].set(-mol.intor('int1e_rinv') * charges[k])
    # total nuclear potential
    nuc = jnp.sum(sub_nuc, axis=0)
    return kin, nuc, sub_nuc

def main(mol, mf, mo_coeff, mo_occ, pop_method, ext):
    """
    main program
    """
    # compute population weights
    weights = assign_rdm1s(mol, mo_coeff, pop_method)
    # atomic energies
    return prop_tot(mol, mf, mo_coeff, mo_occ, weights, ext)

def pm_jacobi_sweep(mol, orbocc, s1e, pop_method, exponent=2, conv_tol=LOC_TOL):
    orbocc = numpy.asarray(stop_grad(orbocc))
    mlo = pipek.PM(mol, orbocc)
    mlo.pop_method = pop_method
    mlo.exponent = exponent
    mlo.conv_tol = conv_tol
    _ = mlo.kernel()
    mlo = pipek.jacobi_sweep(mlo)
    orbloc = mlo.mo_coeff
    u0 = orbocc.T @ s1e @ orbloc
    return u0

def energy(E, mol, mo, pop_method, exponent=2, sweep=False):
    """
    compute atomic contributions to molecular energy
    """
    mf = scf.RHF(mol)
    ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    h1 = mf.get_hcore()
    field = jnp.einsum('x,xij->ij', E, ao_dip)
    mf.get_hcore = lambda *args, **kwargs: h1 + field
    mf.kernel()
    assert mf.converged, 'mf not converged'
    # localized pm orbitals or ibos
    mo_occ = mf.mo_occ[mf.mo_occ>0]
    orbocc = mf.mo_coeff[:, mf.mo_occ>0]
    if mo == 'can':
        orbloc = orbocc
    else:
        if sweep:
            u0 = pm_jacobi_sweep(mol, orbocc, mf.get_ovlp(), pop_method, exponent)
            orbloc = pipek.pm(mol, orbocc, conv_tol=LOC_TOL,
                              pop_method=pop_method, exponent=exponent, init_guess=u0)
        else:
            orbloc = pipek.pm(mol, orbocc, conv_tol=LOC_TOL,
                              pop_method=pop_method, exponent=exponent, init_guess="atomic")
    return main(mol, mf, orbloc, mo_occ, pop_method, field)

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
            pol = jnp.sum(-jacrev(jacrev(energy))(E0, mol, mo, pop_method, sweep=False), axis=0)
            # assert differences
            print(f'{mol_name:} / {mo:} / {pop_method:}:')
            print('total polarizabilities:\n', pol - pol_ref)
            print('isotropic polarizabilities:\n', (jnp.trace(pol) - jnp.trace(pol_ref)) / 3.)
