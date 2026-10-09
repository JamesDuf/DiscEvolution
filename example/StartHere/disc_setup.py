"""
Disc initial-condition solvers for DiscEvolution runs.

A DiscEvolution disc is set by four numbers that are all coupled through
the viscous alpha-disc relations: the viscosity parameter `alpha`, the
characteristic radius `Rd`, the disc mass `Mdisk`, and the initial
accretion rate `Mdot`. You normally know three of these from observations
or from the parameter grid you want to explore, and need to solve
(iteratively, because the equation of state depends on Sigma which depends
on the answer) for the fourth.

Each function below implements one such "fix three, solve for one" recipe.
Only the recipe actually used for the Balogh et al. HJ-formation runs is
implemented here:

    winds_alpha_disc()  -- fix Rd, Mdot, Mdisk (and the disc-wind mass-loss
                            parameter psi_DW) and solve for the alpha that
                            reproduces the target Mdot.

The original run_model_discchem_stream.py additionally implements five
more variants (fixed-Rd/solve-for-alpha without winds, solve-for-Rd,
solve-for-Mdot, the Lynden-Bell & Pringle self-similar solution, and the
wind equivalents of the Rd/Mdot solves). If you need one of those, look
there for a template and add it to `_SETUP_FUNCS` below -- the calling
script (run_model_student.py) does not need to change.
"""

import numpy as np

from DiscEvolution.constants import AU, Msun
from DiscEvolution.eos import IrradiatedEOS, LocallyIsothermalEOS, SimpleDiscEOS, DeadZoneEOS
from DiscEvolution.disc import AccretionDisc
from DiscEvolution.viscous_evolution import HybridWindModel
from DiscEvolution.solvers import *


def make_eos(eos_params, star, alpha_t, kappa=None, psi=None, e_rad=None,
    Mdot=None, 
    t_initial_yr=None,
):
    """
    Build the equation-of-state object named in eos_params['type'].

    alpha_t here must be the *viscous* alpha (what DiscEvolution calls
    alpha_SS): if a disc wind is present, this is smaller than the total
    alpha by a factor (1 + psi), since the wind carries away angular
    momentum without contributing to viscous heating.
    """
    eos_type = eos_params["type"]
    if eos_type == "SimpleDiscEOS":
        return SimpleDiscEOS(star, alpha_t=alpha_t)
    elif eos_type == "LocallyIsothermalEOS":
        return LocallyIsothermalEOS(star, eos_params["h0"], eos_params["q"], alpha_t)
    elif eos_type == "IrradiatedEOS":
        return IrradiatedEOS(star, alpha_t=alpha_t, kappa=kappa, psi=psi, e_rad=e_rad,
                              Tmax=eos_params["Tmax"])
    elif eos_type == "DeadzoneEOS":
        return DeadZoneEOS(
                star,
                psi=psi, 
                Mdot=Mdot,
                alpha_guess=alpha_t,          # use the alpha_SS dead given as free parameter not the old alpha_SS that was being passed in the JSON 
                evolution_model=eos_params["evolution_model"],
                ionization_model=eos_params["ionization_model"],       
                r0=eos_params["r0"], 
                r1=eos_params["r1"],
                r_floor=eos_params["r_floor"],
                t0=eos_params["t0"], 
                t1=eos_params["t1"],
                t_initial_yr=t_initial_yr,
                kappa=kappa, 
                e_rad=e_rad, 
                Tmax=eos_params["Tmax"],
                warm_start=eos_params["warm_start"],
            )
    else:
        raise ValueError(f"Unknown eos type {eos_type!r}")


def winds_alpha_disc(grid, star, disc_params, eos_params, wind_params, kappa):
    """
    Solve for alpha given a fixed Rd, Mdot, Mdisk and disc-wind strength.

    Units in / out (this is the part that trips people up):
        grid.Rc                        -- radii, in AU
        disc_params['Mdot']            -- target accretion rate, in Msun/yr
        disc_params['M']               -- disc mass, in Msun (converted to
                                           grams internally, since AU/Msun
                                           are cgs constants and G = 1 in
                                           these units)
        disc_params['Rd']              -- characteristic radius, in AU
        returned Sigma                 -- surface density, in g/cm^2

    Physics:
        Sigma(R) starts as the standard self-similar power-law profile
        Sigma ~ (R/Rd)^-gamma * exp[-(R/Rd)^(2-gamma)], renormalized to
        the target disc mass. We then iterate:
            1. build the EOS (temperature structure) for the current alpha
            2. measure the Mdot that alpha actually produces
            3. rescale alpha by (target Mdot / actual Mdot)
        Steps are damped by averaging old/new alpha 50/50 each iteration,
        since the direct update can oscillate. 100 iterations converges
        comfortably for the (psi, Mdot, Mdisk, Rd) ranges used in this
        project.

        With a disc wind (psi_DW > 0) not all of alpha drives viscous
        accretion: `lambda_DW` is the wind lever-arm parameter (Tabone
        et al. 2022 notation) and `alpha_SS = alpha / (1 + psi)` is the
        viscous-only alpha that actually goes into the EOS.

    Returns
    -------
    disc, eos, Sigma, alpha, alpha_SS, lambda_DW
    """
    Mdot_target = disc_params["Mdot"]        # Msun/yr
    Mdisk = disc_params["M"] * Msun          # g
    Rd = disc_params["Rd"]                   # AU
    gamma = disc_params["gamma"]              # Sigma power-law index
    alpha = disc_params["alpha"]              # initial guess, total alpha

    psi = wind_params["psi_DW"]               # wind mass-loss parameter
    e_rad = wind_params["e_rad"]               # wind lever-arm efficiency
    if psi > 0:
        lambda_DW = 1 / (2 * (1 - e_rad) * (3 / psi + 1)) + 1
    else:
        # Pure-viscous limit (psi -> 0): the wind velocity v_DW in
        # HybridWindModel is proportional to psi, so it's exactly zero here
        # regardless of lambda_DW -- there's no finite limit of the lambda_DW
        # formula itself (it diverges as psi -> 0), so just pick any finite
        # placeholder value; it multiplies a wind term that's already zero.
        lambda_DW = np.inf
    alpha_SS = alpha / (1 + psi)

    R = grid.Rc
    Sigma = (R / Rd) ** (-gamma) * np.exp(-(R / Rd) ** (2 - gamma))

    # Normalize to the requested disc mass (AccretionDisc.Mtot() integrates
    # 2*pi*R*Sigma over the grid in cgs, so this works even though `eos` is
    # not defined yet -- we only need Sigma for the mass integral here).
    disc = AccretionDisc(grid, star, eos=None, Sigma=Sigma)
    Sigma *= Mdisk / disc.Mtot()

    gas = HybridWindModel(psi, lambda_DW)

    for _ in range(100):
        eos = make_eos(eos_params, star, alpha_SS, kappa=kappa, psi=psi, e_rad=e_rad)
        eos.set_grid(grid)
        eos.update(0, Sigma)

        disc = AccretionDisc(grid, star, eos, Sigma)
        v_r = gas.viscous_velocity(disc, Sigma)
        Mdot_actual = disc.Mdot(v_r)[0]      # Msun/yr

        alpha = 0.5 * (alpha + alpha * Mdot_target / Mdot_actual)
        alpha_SS = alpha / (1 + psi)

    return disc, eos, Sigma, alpha, alpha_SS, lambda_DW

def winds_psi_disc(grid, star, disc_params, eos_params, wind_params, kappa):
    """ """
    # --------- Initial Setup ------------
    Mdot_target = disc_params["Mdot"]        # Msun/yr
    Mdisk = disc_params["M"] * Msun          # g
    Rd = disc_params["Rd"]                   # AU
    gamma = disc_params["gamma"]             # Sigma power-law index
    # alpha = disc_params["alpha"]             # total alpha

    psi_boot = 1.0 # wind_params["psi_DW"]              # bootstrap wind mass-loss parameter
    e_rad = wind_params["e_rad"]             # wind lever-arm efficiency
    if psi_boot > 0:
        lambda_DW_boot= 1 / (2 * (1 - e_rad) * (3 / psi_boot + 1)) + 1
    else:
        # Pure-viscous limit (psi -> 0): the wind velocity v_DW in
        # HybridWindModel is proportional to psi, so it's exactly zero here
        # regardless of lambda_DW -- there's no finite limit of the lambda_DW
        # formula itself (it diverges as psi -> 0), so just pick any finite
        # placeholder value; it multiplies a wind term that's already zero.
        lambda_DW_boot = np.inf

    R = grid.Rc
    Sigma = (R / Rd) ** (-gamma) * np.exp(-(R / Rd) ** (2 - gamma))

    # Normalize to the requested disc mass (AccretionDisc.Mtot() integrates
    # 2*pi*R*Sigma over the grid in cgs, so this works even though `eos` is
    # not defined yet -- we only need Sigma for the mass integral here).
    disc = AccretionDisc(grid, star, eos=None, Sigma=Sigma)
    Sigma *= Mdisk / disc.Mtot()

    # gas_temp = HybridWindModel(psi_boot, lambda_DW_boot)

    # scale Sigma by current Mtot just in case Sigma is not quite at the correct value to have the desired Mdisk (which often happens)
    Mtot = disc.Mtot()
    Sigma[:] *= Mdisk / Mtot


    # ---------- Psi boot ---------------
    alpha_SS_dead = eos_params["alpha_SS_dead"]
    alpha_SS_active = eos_params["alpha_SS_active"]
    alpha_boot = alpha_SS_dead                           # Bootstrap should start from the known dead-zone alpha  #TODO: make this the inner alpha in new deadzone cases? 

    eos_temp = make_eos(eos_params, star, alpha_boot, kappa=kappa, psi=psi_boot, e_rad=e_rad)
    eos_temp.set_grid(grid)
    eos_temp.update(0, Sigma)

    disc_temp = AccretionDisc(grid, star, eos, Sigma)
    gas_temp = HybridWindModel(psi_boot, lambda_DW_boot)

    # solve the interior (dead-zone) psi for target Mdot, holding
    # alpha_SS_dead fixed at its config value
    psi_dead = psi_from_alpha_SS_Mdot_RootFinder(
        eos, 
        disc_temp, 
        gas_temp, 
        Mdot,
    )

    # ----------- Build the actual eos ----------
    eos = make_eos(eos_params, star, alpha_SS_dead, kappa=kappa, psi=psi_dead, e_rad=e_rad)
    eos.set_grid(grid)
    eos.update(0, Sigma)

    if isinstance(eos, DeadZoneEOS):
        alpha_DW = psi_dead * alpha_SS_dead
        psi_active = alpha_DW / alpha_SS_active

        eos.build_alpha_psi_arrays(
            alpha_dead=alpha_SS_dead,
            alpha_active=alpha_SS_active,
            psi_dead=psi_dead,
            psi_active=psi_active,
            w=eos_params.get("w", 1.0),
        )
        eos.update(0, Sigma)

        psi_profile = np.asarray(eos._psi, dtype=float)
    else:
        psi_profile = np.full_like(R, psi_dead, dtype=float)            # This is constant array for non-deadzone eos's
 
    # Construct final disc
    disc = AccretionDisc(grid, star, eos, Sigma)

    # Recompute the wind lever arm consistently with the solved dead-zone psi. 
    psi_safe = np.clip(psi_profile, 1e-8, None)         # safe for division by zero
    lambda_DW = 1.0 / ( 2.0*(1.0 - e_rad)*(3.0/psi_safe + 1.0) ) + 1.0

    return disc, eos, Sigma, lambda_DW, psi_profile    #TODO: return the right thing

# --------- Dispatch table ---------
# add an entry here (and a matching function above, or
# ported over from run_model_discchem_stream.py) to support another
# disc-initialization recipe without touching run_model_student.py.

_SETUP_FUNCS = {
    "winds-alpha": winds_alpha_disc,
    "winds-psi": winds_psi_disc,
}


def setup_disc(grid, star, config, kappa):
    """
    Build the initial disc for `config['grid']['type']`.

    This is the single entry point run_model_student.py calls; it just
    looks up the right recipe above and forwards the relevant config
    sections to it.
    """
    grid_type = config["grid"]["type"]
    try:
        func = _SETUP_FUNCS[grid_type]
    except KeyError:
        raise ValueError(
            f"Unknown or unsupported grid type {grid_type!r}. "
            f"This simplified module implements: {list(_SETUP_FUNCS)}. "
            "See run_model_discchem_stream.py for the other five disc-"
            "initialization recipes (Booth-alpha/Rd/Mdot, LBP, winds-Rd/"
            "winds-Mdot) if you need one of those -- port the branch over "
            "into a function here and add it to _SETUP_FUNCS."
        )
    return func(grid, star, config["disc"], config["eos"], config["winds"], kappa)
