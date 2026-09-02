import os
import json
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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