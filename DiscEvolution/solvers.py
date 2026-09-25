import os
import json
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from cycler import cycler
import h5py

from DiscEvolution.constants import *
from DiscEvolution.grid import Grid
from DiscEvolution.star import SimpleStar
from DiscEvolution.eos import IrradiatedEOS, LocallyIsothermalEOS, SimpleDiscEOS, DeadZoneEOS
from DiscEvolution.disc import *
from DiscEvolution.viscous_evolution import ViscousEvolution, ViscousEvolutionFV, LBP_Solution, HybridWindModel, TaboneSolution
from DiscEvolution.disc import AccretionDisc
from DiscEvolution.dust import *
from DiscEvolution.dust import PlanetesimalFormation
from DiscEvolution.planet_formation import *
from DiscEvolution.diffusion import TracerDiffusion
from DiscEvolution.opacity import Tazzari2016, Zhu2012
from DiscEvolution.chemistry import *
from scipy.optimize import brentq
from copy import deepcopy


# ---------------- James, September 2, 2026 -----------------------
# The intent is to move over all the individual solvers from the run model script branches 
# i.e. 'Booth-alpha' 'Booth-Rd' "LBP" 'Booth-Mdot' 'winds-alpha'
# I moved the two solvers I made for DeadZoneEOS (used in winds-alpha) but left the rest for later
# For now, simply collapsing the unused if-else branches in the run_model file is good enough but 
# for clarity and ease of use, moving them here and importing them in the run_model script will be better. 


def alphaSS_from_Mdot_psi(eos, disc, wind_model, Mdot_target, guess = None, max_iterations=20, tol=1e-5):
    """
    Solve for alpha at the star surface given Mdot_target and fixed psi using iterative refinement.
    
    The iteration scheme is: alpha_new = alpha_old * (Mdot_target / Mdot_actual)
    
    Parameters
    ----------
    disc : Disc object
        The disc on which to compute Mdot
    wind_model : wind model object
        The wind/viscous model (must have viscous_velocity method)
    Mdot_target : float
        Target accretion rate (Msun/yr)
    max_iterations : int
        Maximum number of iterations
    tol : float
        tolerance for convergence, stops when iteration Mdot is within tol of target Mdot
        
    Returns
    -------
    alpha_converged : float
        The converged value of alpha
    """
    if guess != None:
        alpha = guess 
    else: 
        alpha = eos._alpha_t

    
    for iteration in range(max_iterations):
        # Update self with current alpha
        eos._alpha_t = alpha
        eos.update(0, disc.Sigma)  # Re-solves thermal balance equation, updates psi and other values automatically
        
        # Compute current Mdot
        vr = wind_model.viscous_velocity(disc)
        Mdot_actual = disc.Mdot(vr[0])
        
        # Check if within tolerance
        rel_error = np.abs(Mdot_actual - Mdot_target) / Mdot_target
        if rel_error < tol:
            print(f"Alpha solver converged to within {tol:%} of target accretion rate after iteration {iteration}: alpha={alpha:.6e}, Mdot={Mdot_actual:.6e}")
            return alpha
        
        # Update alpha
        alpha_new = alpha * (Mdot_target / Mdot_actual)
        alpha = 0.5 * (alpha_new + alpha)

    print(f"Warning: Alpha solver did not converge to within {tol:%} of target accretion rate after {max_iterations} iterations")
    print(f"  Final alpha: {alpha:.6e}")
    print(f"  Final Mdot: {disc.Mdot(wind_model.viscous_velocity(disc)[0]):.6e}")
    print(f"  Target Mdot: {Mdot_target:.6e}")
    
    return alpha


def psi_from_alphaSS_Mdot(eos, disc, wind_model, Mdot_target, guess = None, max_iterations=20, tol=1e-5):
    """
    Solve for psi at the star surface given Mdot_target and fixed alphaSS using iterative refinement.
    
    The iteration scheme is: psi_new = (1 + psi_old)*(Mdot_target / Mdot_actual) - 1 
    
    Parameters
    ----------
    disc : Disc object
        The disc on which to compute Mdot
    wind_model : wind model object
        The wind/viscous model (must have viscous_velocity method)
    Mdot_target : float
        Target accretion rate (Msun/yr)
    max_iterations : int
        Maximum number of iterations
    tol : float
        tolerance for convergence, stops when iteration Mdot is within tol of target Mdot
        
    Returns
    -------
    psi : float
        The converged value of psi
    """
    if guess != None: 
        psi = guess
    else: 
        psi = eos._psi

    for iteration in range(max_iterations):
        # Update eos and wind model with psi 
        eos._psi = psi
        eos.update(0, disc.Sigma) 
        wind_model._psi = psi

        # Compute current Mdot
        vr = wind_model.viscous_velocity(disc)
        Mdot_actual = disc.Mdot(vr[0])

        # Check if within tolerance
        rel_error = np.abs(Mdot_actual - Mdot_target) / Mdot_target
        if rel_error < tol:
            print(f"Psi solver converged to within {tol:%} of target accretion rate after iteration {iteration}: psi={psi:.6e}, Mdot={Mdot_actual:.6e}")
            return psi
        
        # Update psi 
        psi_new = ( (1 + psi) * (Mdot_target / Mdot_actual) ) - 1 
        psi = 0.5 * (psi_new + psi)

    print(f"Warning: Psi solver did not converge to within {tol:%} of target accretion rate after {max_iterations} iterations")
    print(f"  Final psi: {psi:.6e}")
    print(f"  Final Mdot: {disc.Mdot(wind_model.viscous_velocity(disc)[0]):.6e}")
    print(f"  Target Mdot: {Mdot_target:.6e}")

    return psi


def psi_from_alpha_SS_Mdot_RootFinder(eos, disc, wind_model, Mdot_target, psi_space = np.logspace(-4, 2.5, num=100), verbose=False):
    """
    Solve for psi at fixed alphaSS such that F(psi) = Mdot_target - Mdot(psi) = 0

    Parameters
    ----------
    eos, disc, wind_model : as in psi_from_alphaSS_Mdot
    Mdot_target : float
        Target accretion rate (Msun/yr)
    psi_space : ndarray
        Monotonically increasing grid of psi values used to bracket the root
    xtol : float
        Absolute tolerance on psi passed to brentq

    Returns
    -------
    psi_root : float
        The value of psi for which Mdot(psi) == Mdot_target
    """
    def F(psi):
        eos._psi = psi
        eos.update(0, disc.Sigma)
        wind_model._psi = psi

        vr = wind_model.viscous_velocity(disc)
        Mdot_actual = disc.Mdot(vr[0])
        return Mdot_target - Mdot_actual

    F_vals = np.array([F(psi) for psi in psi_space])

    # Locate the first bracket where F changes sign
    sign_changes = np.where(np.diff(np.sign(F_vals)) != 0)[0]           # Return indices where consecutive cells change sign, left side of the bracket
    if sign_changes.size == 0:
        if verbose:
            _plot_F_vs_psi(psi_space, F_vals)
        raise ValueError(
            "psi_from_alpha_SS_Mdot_RootFinder: no sign change in Mdot_target - Mdot(psi) found over "
            f"psi_space=[{psi_space[0]:.3e}, {psi_space[-1]:.3e}] "
        )
    idx = sign_changes[0]       # pick left most bracket with a sign change -> smallest psi
    psi_lo, psi_hi = psi_space[idx], psi_space[idx + 1]         # Set a bracket in psi space around the psi value where F changed sign

    # Find the Root (where F(psi) = 0)
    psi_root = brentq(F, psi_lo, psi_hi)
    print(f"Psi root finder converged: psi={psi_root:.6e}, Mdot={Mdot_target - F(psi_root):.6e}")

    if verbose:
        _plot_F_vs_psi(psi_space, F_vals, psi_lo, psi_hi, psi_root)

    return psi_root

def _plot_F_vs_psi(psi_space, F_vals, psi_lo=None, psi_hi=None, psi_root=None, mdot_val=None, alpha_val=None) -> None:
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    # ticks on all four sides
    ax.tick_params(axis="both", which="major", direction="in", length=6, width=1.1, top=True, right=True)
    ax.tick_params(axis="both", which="minor", direction="in", length=3, width=0.8, top=True, right=True)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, which="major", alpha=0.18, lw=0.8)

    plt.semilogx(psi_space, F_vals, label=r'F($\psi$)')
    ax.axhline(0, color='k', ls = '--', lw=0.8)
    if psi_lo is not None and psi_hi is not None:
        ax.axvspan(psi_lo, psi_hi, color='orange', alpha=0.3, label='bracket')
    if psi_root is not None:
        ax.axvline(psi_root, color='r', ls='--', lw = 0.8, label=f'root = {psi_root:.3f}')

    ax.set_xlabel(r'$\psi$')
    ax.set_ylabel(r'F($\psi$) = $\dot{M}_{\rm target}$ - $\dot{M}(\psi)$')
    # ax.set_title(r'$\dot{M}_{\rm target}$ ')
    ax.legend(frameon=False, fontsize=9, loc="best")
    plt.show()