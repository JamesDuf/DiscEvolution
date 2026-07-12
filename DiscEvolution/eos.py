from __future__ import print_function
import numpy as np
from DiscEvolution.brent import brentq
from DiscEvolution import opacity
from DiscEvolution.constants import *

################################################################################
# Thermodynamics classes
################################################################################
class EOS_Table(object):
    """Base class for equation of state evaluated at certain locations.

    Stores pre-computed temperatures, viscosities etc. Derived classes need to
    provide the funcitons called by set_grid.
    """
    def __init__(self):
        self._gamma = 1.0
        self._mu    = 2.4
    
    def set_grid(self, grid):
        self._R      = grid.Rc
        self._set_arrays()

    def _set_arrays(self):
        R  = self._R
        self._cs     = self._f_cs(R)
        self._H      = self._f_H(R)
        self._nu     = self._f_nu(R)
        self._alpha  = self._f_alpha(R)

    @property
    def cs(self):
        return self._cs

    @property
    def H(self):
        return self._H

    @property
    def nu(self):
        return self._nu
    
    @property
    def visc_mol(self):
        return self._f_visc_mol()

    @property
    def alpha(self):
        return self._alpha

    @property
    def gamma(self):
        return self._gamma

    @property
    def mu(self):
        return self._mu

    def update(self, dt, Sigma, amax=None, star=None):
        """Update the eos"""
        pass

    def ASCII_header(self):
        """Print eos header"""
        head = '# {} gamma: {}, mu: {}'
        return head.format(self.__class__.__name__,
                           self.gamma, self.mu)

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        def fmt(x):  return "{}".format(x)
        return self.__class__.__name__, { "gamma" : fmt(self.gamma),
                                          "mu" : fmt(self.mu) }
    
class LocallyIsothermalEOS(EOS_Table):
    """Simple locally isothermal power law equation of state:

    args:
        h0      : aspect ratio at 1AU
        q       : power-law index of sound-speed
        alpha_t : turbulent alpha parameter
        star    : stellar properties
        mu      : mean molecular weight, default=2.4
    """
    def __init__(self, star, h0, q, alpha_t, mu=2.4):
        super(LocallyIsothermalEOS, self).__init__()
        
        self._h0 = h0
        self._cs0 = h0 * star.M**0.5
        self._q = q
        self._alpha_t = alpha_t
        self._H0 = h0
        self._T0 = (AU*Omega0)**2 * mu / GasConst
        self._mu = mu

    def _f_cs(self, R):
        return self._cs0 * R**self._q

    def _f_H(self, R):
        return self._H0 * R**(1.5+self._q)
    
    def _f_nu(self, R):
        return self._alpha_t * self._f_cs(R) * self._f_H(R)
    
    def _f_visc_mol(self):
        return 2/3 * np.sqrt(self.mu * m_H * GasConst * self.T/ np.pi ) / sig_H2

    def _f_visc_mol(self):
        return 2/3 * np.sqrt(self.mu * m_H * GasConst * self.T/ np.pi ) / sig_H2

    def _f_alpha(self, R):
        return self._alpha_t

    @property
    def T(self):
        return self._T0 * self.cs**2

    @property
    def Pr(self):
        return np.zeros_like(self._R)

    def ASCII_header(self):
        """LocallyIsothermalEOS header string"""
        head = super(LocallyIsothermalEOS, self).ASCII_header()
        head += ', h0: {}, q: {}, alpha: {}'
        return head.format(self._h0, self._q, self._alpha_t)

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        name, head = super(LocallyIsothermalEOS, self).HDF5_attributes()
        head["h0"]   = "{}".format(self._h0)
        head["q"]     = "{}".format(self._q)
        head["alpha"] = "{}".format(self._alpha_t)
        return name, head

    @staticmethod
    def from_file(filename):
        raise NotImplementedError('')

    @property
    def star(self):
        return self._star


class SimpleDiscEOS(EOS_Table):
    """Simple approximate irradiated/viscous equation of state from Liu et al.
    (2019).

    args:
        alpha_t : turbulent alpha parameter
        star    : stellar properties
        mu      : mean molecular weight, default=2.33
        K0      : Opacity constant (K = K0 T), default = 0.01
    """
    def __init__(self, star, alpha_t, mu=2.33, K0=0.01):
        super(SimpleDiscEOS, self).__init__()
        
        self._alpha_t = alpha_t
        self._mu = mu
        self._K0 = K0
        self._star = star
        
        self._Tnu = np.sqrt(27/64*alpha_t*Omega0*GasConst*K0/(mu*sig_SB))

        self._set_constants()

    def _set_constants(self):
        star = self._star

        Ls = star.Rs**2 * (star.T_eff / 5770)**4
        self._Tirr0 = 150 * Ls**(2/7.) * star.M**(-4/7)
        self._Tnu0 = self._Tnu * star.M**0.25

        self._cs0 = (Omega0**-1/AU) * (GasConst / self._mu)**0.5
        self._H0  = (Omega0**-1/AU) * (GasConst / (self._mu*self._star.M))**0.5
        self._nu0 = self._alpha_t * self._cs0**2 / Omega0

    def update(self, dt, Sigma, amax=1e-5, star=None):
        if star:
            self._star = star

        self._set_constants()

        Tirr = self._Tirr0 * self._R**(-3/7.)
        Tvis = self._Tnu * Sigma * self._R**-0.75

        self._T = (Tirr**4 + Tvis**4)**0.25
        self._Sigma = Sigma
        
        self._set_arrays()

    def set_grid(self, grid):
        self._R = grid.Rc
        self._T = None

    def _set_arrays(self):
        super(SimpleDiscEOS,self)._set_arrays()
        self._Pr = self._f_Pr()
    
    def __H(self, R, T):
        return self._H0 * np.sqrt(T * R*R*R)

    def _f_cs(self, R):
        return self._cs0 * self._T**0.5

    def _f_H(self, R):
        return self.__H(R, self._T)
    
    def _f_nu(self, R):
        return self._alpha_t * self._f_cs(R) * self._f_H(R)
    
    def _f_visc_mol(self):
        return 2/3 * np.sqrt(self.mu * m_H * GasConst * self.T/ np.pi ) / sig_H2

    def _f_alpha(self, R):
        return self._alpha_t

    def _f_Pr(self):
        kappa = self._K0 * self._T
        tau = 0.5 * self._Sigma * kappa
        f_esc = 1 + 2/(3*tau*tau)        
        # Paardekooper et al. (2011) are missing a factor 4 (Bitsch & Kley 2011).  Corrected here.
        Pr_1 =  4. * 2.25 * self._gamma * (self._gamma - 1) * f_esc
        return 1. / Pr_1

    @property
    def T(self):
        return self._T

    @property
    def Pr(self):
        return self._Pr
    
    @property
    def nu0(self):
        return self._nu0

    def ASCII_header(self):
        """LocallyIsothermalEOS header string"""
        head = super(SimpleDiscEOS, self).ASCII_header()
        head += ', alpha: {}, mu: {}, K0= {}'
        return head.format(self._alpha_t, self._mu, self._K0)

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        name, head = super(SimpleDiscEOS, self).HDF5_attributes()
        head["alpha"] = "{}".format(self._alpha_t)
        head['mu'] = "{}".format(self._mu)
        head['K0'] = "{}".format(self._K0)
        return name, head

    @staticmethod
    def from_file(filename):
        import star

        star = star.from_file(filename)
        alpha = None

        with open(filename) as f:
            for line in f:
                if not line.startswith('#'):
                    raise AttributeError("Error: EOS type not found in header")
                elif "SimpleDiscEOS" in line:
                    string = line 
                    break 
                else:
                    continue

        kwargs = {}
        for item in string.split(','):    
            key, val = [ x.strip() for x in item.split(':')]

            if key == 'mu' or key == 'K0':
                kwargs[key] = float(val.strip())
            elif key == 'alpha':
                alpha = float(val.strip())


        return SimpleDiscEOS(star, alpha, **kwargs)


    @property
    def star(self):
        return self._star


_sqrt2pi = np.sqrt(2*np.pi)
class IrradiatedEOS(EOS_Table):
    """Model for an active irradiated disc.

    From Nakamoto & Nakagawa (1994), Hueso & Guillot (2005).

    args:
        star    : Stellar properties
        alpha_t : Viscous alpha parameter
        Tc      : External irradiation temperature (nebular), default=10
        Tmax    : Maximum temperature allowed in the disc, default=1500
        mu      : Mean molecular weight, default = 2.4
        gamma   : Ratio of specific heats
        kappa   : Opacity, default=Zhu2012
        accrete : Whether to include heating due to accretion,
                  default=True
        psi : Ratio of disk winds to viscous turbulent alpha, default: psi = 0.
        e_rad : fraction of energy lost to radiation (Suzuki et. al 2016), default = 1

    Notes: 
        If disk winds are being used, different choices of e_rad provide different heating
        cases. See Suzuki et. al (2016). The special/edge cases are as follows.
        - If e_rad = 3/(3 + psi), all (and only) turbulent energy goes into heating.
        - If e_rad ~ 1, the weak winds case (from Suzuki et. al 2016) is applied.

        If the user wishes to be self-consistent, one must choose a magnetic lever 
        arm parameter (lambda) such that lambda = 1 + psi/(2(1 - e_rad)(3 + psi)). 
    """
    def __init__(self, star, alpha_t, Tc=10, Tmax=1500., mu=2.4, gamma=1.4,
                 kappa=None,
                 accrete=True, tol=None, psi=0, e_rad=1): # tol is no longer used
        super(IrradiatedEOS, self).__init__()

        self._star = star
        
        self._dlogHdlogRm1 = 2/7.

        self._alpha_t = alpha_t
        
        self._Tc = Tc
        self._Tmax = Tmax
        self._mu = mu

        self._accrete = accrete
        self._gamma = gamma

        if kappa is None:
            self._kappa = opacity.Zhu2012
        else:
            self._kappa = kappa
        
        self._T = None

        self._psi = psi

        self._e_rad = e_rad

        self._compute_constants()

    def _compute_constants(self):
        self._sigTc4 = sig_SB*self._Tc**4
        self._cs0 = (Omega0**-1/AU) * (GasConst / self._mu)**0.5
        self._H0  = (Omega0**-1/AU) * (GasConst / (self._mu*self._star.M))**0.5


    def update(self, dt, Sigma, amax=1e-5, star=None):
        if star:
            self._star = star
            self._compute_constants()
        star = self._star
            
        # Temperature/density independent quantities:
        R = self._R
        Om_k = Omega0 * star.Omega_k(R)

        X = star.Rau/R
        f_flat  = (2/(3*np.pi)) * X**3
        f_flare = 0.5 * self._dlogHdlogRm1 * X**2
        tauPovertauR = 2.4 # Ratio of Planck to Rosseland optical depths.  Nakamoto => 2.4.
        
        # Heat capacity
        mu = self._mu
        #C_V = (k_B / (self._gamma - 1)) * (1 / (mu * m_H))
        
        alpha = self._alpha_t
        if not self._accrete:
            alpha = 0.

        # Local references 
        max_heat = sig_SB * (self._Tmax*self._Tmax)*(self._Tmax*self._Tmax)
        star_heat = sig_SB * star.T_eff**4
        sqrt2pi = np.sqrt(2*np.pi)            
        def balance(Tm):
            """Thermal balance"""
            cs = np.sqrt(GasConst * Tm / mu)
            H = cs / Om_k

            kappa = self._kappa(Sigma / (sqrt2pi * H), Tm, amax)
            tau = 0.5 * Sigma * kappa
            tauR = tau
            tau_P = tauPovertauR * tauR
            H /= AU

            # External irradiation
            dEdt = self._sigTc4
            
            # Compute the heating from stellar irradiation
            dEdt += star_heat * (f_flat + f_flare * (H/R))

            # Viscous Heating
            # If psi > 0, includes heating from disk winds based off and 
            # derived from the model proposed by Suzuki et. al (2018, 
            #  doi:10.1051/0004-6361/201628955).
            visc_heat = self._e_rad*1.125*alpha*cs*cs * Om_k * (1 + self._psi/3)   
            #dEdt += visc_heat*(0.375*tau*Sigma + 1./(kappa))
            # Reformulation by MLB Jan 2026 for easier comparison with literature
            visc_heat = visc_heat * Sigma
            dEdt += visc_heat*(3./8.*tauR + 1./(1.*tau_P))
            
            # Prevent heating above the temperature cap:
            dEdt = np.minimum(dEdt, max_heat)

            # Cooling
            Tm2 = Tm*Tm
            dEdt -= sig_SB * Tm2*Tm2

            # Change in temperature
            return (dEdt/Omega0) # / (C_V*Sigma)

        # Solve the balance using brent's method (needs ~ 20 iterations)
        T0 = self._Tc
        T1 = self._Tmax
        if self._T is not None:
            dedt = balance(self._T)
            T0 = np.where(dedt > 0, self._T, T0)
            T1 = np.where(dedt < 0, self._T, T1)

        self._T =  brentq(balance, T0, T1)
        self._Sigma = Sigma

        # Save the opacity:
        cs = np.sqrt(GasConst * self._T / mu)
        H = cs / Om_k
        self._kappa_arr = self._kappa(Sigma / (sqrt2pi * H), self._T, amax)
        self._set_arrays()


    def set_grid(self, grid):
        self._R = grid.Rc
        self._T = None

    def _set_arrays(self):
        super(IrradiatedEOS,self)._set_arrays()
        self._Pr = self._f_Pr()
    
    def __H(self, R, T):
        return self._H0 * np.sqrt(T * R*R*R)

    def _f_cs(self, R):
        return self._cs0 * self._T**0.5

    def _f_H(self, R):
        return self.__H(R, self._T)
    
    def _f_nu(self, R):
        return self._alpha_t * self._f_cs(R) * self._f_H(R)
    
    def _f_visc_mol(self):
        return 2/3 * np.sqrt(self.mu * m_H * GasConst * self.T/ np.pi ) / sig_H2

    def _f_alpha(self, R):
        return self._alpha_t

    def _f_Pr(self):
        kappa = self._kappa_arr
        tau = 0.5 * self._Sigma * kappa
        # Added intermediate optical depth term.
        f_esc = 1 + 2.*np.sqrt(3.)/(3.*tau)+2/(3*tau*tau)
        # Paardekooper et al. (2011) are missing a factor 4 (Bitsch & Kley 2011).  Corrected here.
        Pr_1 =  4. * 2.25 * self._gamma * (self._gamma - 1) * f_esc
        return 1. / Pr_1

    @property
    def T(self):
        return self._T

    @property
    def Pr(self):
        return self._Pr

    @property
    def star(self):
        return self._star

    def ASCII_header(self):
        """IrradiatedEOS header"""
        head = super(IrradiatedEOS, self).ASCII_header()
        head += ', opacity: {}, T_extern: {}K, accrete: {}, alpha: {}'
        head += ', Tmax: {}K'
        return head.format(self._kappa.__class__.__name__,
                           self._Tc, self._accrete, self._alpha_t,
                           self._Tmax)

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        name, head = super(IrradiatedEOS, self).HDF5_attributes()

        head["opacity"]  = self._kappa.__class__.__name__
        head["T_extern"] = "{} K".format(self._Tc)
        head["accrete"]  = "{}".format(bool(self._accrete))
        head["alpha"]    = "{}".format(self._alpha_t)
        head["Tmax"]     = "{} K".format(self._Tmax)

        return name, head

    @staticmethod
    def from_file(filename):
        import star

        star = star.from_file(filename)
        alpha = None

        with open(filename) as f:
            for line in f:
                if not line.startswith('#'):
                    raise AttributeError("Error: EOS type not found in header")
                elif "IrradiatedEOS" in line:
                    string = line 
                    break 
                else:
                    continue

        kwargs = {}
        for item in string.split(','):    
            key, val = [ x.strip() for x in item.split(':')]

            if   key == 'gamma' or key == 'mu':
                kwargs[key] = float(val.strip())
            elif key == 'alpha':
                alpha = float(val.strip())
            elif key == 'accrete':
                kwargs[key] = bool(val.strip())
            elif key == 'T_extern':
                kwargs['Tc'] = float(val.replace('K','').strip())

        return IrradiatedEOS(star, alpha, **kwargs)

class DeadZoneEOS(IrradiatedEOS):
    """Irradiated EOS with a dead-zone transition"""
    
    def __init__(
        # general parameters
        self, 
        star, 

        # alpha solver
        psi,
        Mdot,
        alpha_guess = 1e-3,

        # Deadzone radius evolution
        evolution_model = 'linear',
        r0=None,      # Dead zone radius at t0 (AU)
        r1=None,      # Dead zone radius at t1 (AU)
        r_floor=None,  # Minimum dead zone radius (AU)
        t0=None,       # Reference time 0 (yr)
        t1=None,       # Reference time 1 (yr)
        t_initial_yr=0.0, # Simulation start time in yr

        # Standard IrradiatedEOS parameters
        Tc=10.0,
        Tmax=1500.0,
        mu=2.4,
        gamma=1.4,
        kappa=None,
        accrete=True,
        e_rad=1.0,

        # Thermal solver behaviour
        warm_start=True,  # If False, clear self._T each step for a full-bracket solve
    ):
        # Parent Initializer
        super(DeadZoneEOS, self).__init__(
            star,
            alpha_t=alpha_guess,
            Tc=Tc,
            Tmax=Tmax,
            mu=mu,
            gamma=gamma,
            kappa=kappa,
            accrete=accrete,
            psi=psi,
            e_rad=e_rad,
        )

        # Store DeadZone-specific parameters
        self._Mdot = Mdot
        self._alpha_guess = alpha_guess
        
        # Evolution model choice
        if evolution_model not in ('linear', 'exponential', 'static'):
            raise ValueError(f"Unknown evolution_model: {evolution_model}")
        self._evolution_model = evolution_model
        
        # Evolution parameters
        self._r0 = r0
        self._r1 = r1
        self._r_floor = r_floor
        self._t0 = t0
        self._t1 = t1
        
        # Will be computed correctly each time update() is called
        self._R_dz = None
        
        # Track absolute time in years
        self._t_current_yr = t_initial_yr
        self._t_initial_yr = t_initial_yr
        
        # Stored scalar profile values defining the tanh alpha/psi transition.
        # Populated by build_alpha_psi_arrays(); once set, update() rebuilds the
        # spatial arrays each step as the dead zone moves.
        self._profile_set = False
        self._alpha_dead = None
        self._alpha_active = None
        self._psi_dead = None
        self._psi_active = None
        self._w = None
        
        # Whether to warm-start the thermal solver from the previous temperature.
        # If False, self._T is cleared each step so brentq solves from the full
        # [Tc, Tmax] bracket (guaranteed valid, avoids bracket-inversion failures).
        self._warm_start = warm_start

    def update(self, dt, Sigma, amax=1e-5, star=None):
        """
        Update the EOS with dead zone radius evolution.
        
        Parameters
        ----------
        dt : float
            Time step in code units
        Sigma : array
            Surface density (g/cm^2)
        amax : float, optional
            Maximum grain size (cm). Default: 1e-5
        star : Star object, optional
            Updated stellar properties
            
        Returns
        -------
        None
            Updates internal state: temperature, alpha arrays, dead zone radius
        """
        
        # Increment absolute time 
        self._t_current_yr += dt / yr
        
        # Update dead zone radius at current time
        self._R_dz = self._compute_R_dz(self._t_current_yr)
        
        # Rebuild the spatial alpha/psi arrays so the viscosity structure
        # follows the moving dead zone (only once a profile has been set)
        if self._profile_set:
            self._rebuild_alpha_psi()
        
        # If warm-start is disabled, clear the cached temperature so the parent
        # solves from the full [Tc, Tmax] bracket (still guaranteed to bracket a root)
        if self._warm_start == False:
            self._T = None
        
        # Call parent's thermal balance update
        super(DeadZoneEOS, self).update(dt, Sigma, amax=amax, star=star)

    def alpha_from_Mdot_psi(self, disc, wind_model, Mdot_target, max_iterations=20, tol=1e-3):
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
        
        alpha = self._alpha_t # alpha_guess from initializer
        
        for iteration in range(max_iterations):
            # Update self with current alpha
            self._alpha_t = alpha
            self.update(0, disc.Sigma)  # Re-solves thermal balance equation, updates psi and other values automatically
            
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
            alpha = alpha_new

        # update alpha_t with most recent alpha value even if it didn't converge
        self._alpha_t = alpha

        print(f"Warning: Alpha solver did not converge to within {tol:%} of target accretion rate after {max_iterations} iterations")
        print(f"  Final alpha: {alpha:.6e}")
        print(f"  Final Mdot: {disc.Mdot(wind_model.viscous_velocity(disc)[0]):.6e}")
        print(f"  Target Mdot: {Mdot_target:.6e}")
        
        return alpha

    def _compute_R_dz_linear(self, t):
        """
        Linearly move dead zone radius inward from r0 to r_floor.
        
        Computes velocity from two calibration points (r0, t0) and (r1, t1),
        then linearly extrapolates, flooring at r_floor.

        Treats (r0, t0) as the intial point and (r1, t1) as the final point.  
        
        Parameters
        ----------
        t : float
            Time in yr
            
        Returns
        -------
        R_dz : float
            Dead zone radius in AU
        """
        
        # Compute velocity from calibration points
        if self._t1 <= self._t0:
            raise ValueError("Need t1 > t0")
        
        velocity = (self._r0 - self._r1) / (self._t1 - self._t0)  # AU/yr
        
        # Linear extrapolation from r0 at t0
        t_rel = max(0.0, t - self._t0)
        R_dz = self._r0 - velocity * t_rel
        
        # Floor at r_floor
        return max(R_dz, self._r_floor)

    def _compute_R_dz_exponential(self, t):
        """
        Exponentially move dead zone radius inward from r0 to r_floor.
        
        Uses two calibration points (r0, t0) and (r1, t1) to compute the
        decay time constant tau, then evolves exponentially.

        Treats (r0, t0) as the intial point and (r1, t1) as the final point.
        
        Parameters
        ----------
        t : float
            Time in yr
            
        Returns
        -------
        R_dz : float
            Dead zone radius in AU
        """
        
        # Validate parameters
        if not (self._r_floor < self._r1 < self._r0):
            raise ValueError("Need r_floor < r1 < r0 for inward exponential decay")
        
        if self._t1 <= self._t0:
            raise ValueError("Need t1 > t0")
        
        # Compute decay timescale from calibration points
        tau = -(self._t1 - self._t0) / np.log((self._r1 - self._r_floor) / (self._r0 - self._r_floor))
        
        # Prevent R_dz from exceeding r0 before t0
        if t <= self._t0:
            return self._r0
        
        # Exponential decay
        R_dz = self._r_floor + (self._r0 - self._r_floor) * np.exp(-(t - self._t0) / tau)
        
        # Floor at r_floor (for numerical safety)
        return max(R_dz, self._r_floor)

    def _compute_R_dz_static(self, t):
        """
        Return a static (constant) dead zone radius.
        
        Useful for testing or non-evolving dead zone scenarios.
        
        Parameters
        ----------
        t : float
            Time in yr (ignored)
            
        Returns
        -------
        R_dz : float
            Dead zone radius in AU (constant = self._r0)
        """
        return self._r0

    def _compute_R_dz(self, t):
        """
        Compute dead zone radius at time t using the selected evolution model.
        
        Parameters
        ----------
        t : float
            Time in yr
            
        Returns
        -------
        R_dz : float
            Dead zone radius in AU
        """
        if self._evolution_model == 'linear':
            return self._compute_R_dz_linear(t)
        elif self._evolution_model == 'exponential':
            return self._compute_R_dz_exponential(t)
        elif self._evolution_model == 'static':
            return self._compute_R_dz_static(t)
        else:
            raise ValueError(f"Unknown evolution_model: {self._evolution_model}")

    def build_alpha_psi_arrays(self, alpha_active, alpha_dead=None, psi_dead=None, psi_active=0.01, w=1.0):
        """
        Define the tanh alpha/psi profile and build the initial spatial arrays.
        
        Stores the scalar dead-zone and active-zone values that define the
        transition. After this is called once, update() automatically rebuilds
        the spatial arrays each step as the dead zone radius moves (so the dead
        zone values should be captured here while self._alpha_t / self._psi are
        still scalars, e.g. straight after alpha_from_Mdot_psi).
        
        The dead zone (r < R_dz) has alpha_dead and psi_dead.
        The active region (r > R_dz) has alpha_active and psi_active.
        
        Parameters
        ----------
        alpha_active : float
            Turbulent alpha in the active region (r > R_dz)
        alpha_dead : float, optional
            Alpha in dead zone. If None, captured from the current scalar
            self._alpha_t (from the solver), or the previously stored value.
        psi_dead : float, optional
            Wind parameter in dead zone. If None, captured from the current
            scalar self._psi, or the previously stored value.
        psi_active : float, optional
            Wind parameter in active region. Default: 0.01
        w : float, optional
            Transition width in # of scale heights. Default: 1.0
            
        Returns
        -------
        None
            Stores scalar profile values and assigns the spatial arrays to
            self._alpha_t and self._psi.
        """
        
        # Resolve the dead-zone scalars. Prefer an explicit argument; otherwise
        # use the stored scalar (if a profile was already set) or capture from
        # the current scalar attribute. Never re-read once these are arrays.
        if alpha_dead is None:
            alpha_dead = self._alpha_dead if self._profile_set else self._alpha_t
        if psi_dead is None:
            psi_dead = self._psi_dead if self._profile_set else self._psi
        
        # Validate scalar inputs before storing
        for name, val in (('alpha_dead', alpha_dead), ('alpha_active', alpha_active),
                          ('psi_dead', psi_dead), ('psi_active', psi_active)):
            if not np.isscalar(val) or not np.isfinite(val):
                raise ValueError(f"build_alpha_psi_arrays: {name} must be a finite scalar (got {val!r})")
        if alpha_dead <= 0 or alpha_active <= 0:
            raise ValueError("build_alpha_psi_arrays: alpha values must be positive")
        if psi_dead < 0 or psi_active < 0:
            raise ValueError("build_alpha_psi_arrays: psi values must be non-negative")
        if w <= 0:
            raise ValueError(f"build_alpha_psi_arrays: transition width w must be > 0 (got {w})")
        
        # Store the scalar profile (used by update() to rebuild every step)
        self._alpha_dead = float(alpha_dead)
        self._alpha_active = float(alpha_active)
        self._psi_dead = float(psi_dead)
        self._psi_active = float(psi_active)
        self._w = float(w)
        self._profile_set = True
        
        # Build the spatial arrays at the current dead zone radius
        self._rebuild_alpha_psi()
        
        print(f"Built alpha/psi arrays with tanh transition:")
        print(f"  alpha_dead={self._alpha_dead:.6e}, alpha_active={self._alpha_active:.6e}")
        print(f"  psi_dead={self._psi_dead:.6e}, psi_active={self._psi_active:.6e}")
        print(f"  R_dz={self._R_dz:.4f} AU, transition width w={self._w:.4f} scale heights")

    def _rebuild_alpha_psi(self):
        """
        Rebuild the spatial alpha and psi arrays from the stored scalar profile.
        
        Uses the stored dead/active scalars and the current self._R_dz to build
        a tanh transition, then assigns the arrays to self._alpha_t and self._psi
        (the attributes the parent's thermal balance reads). Called every step by
        update() so the structure follows the moving dead zone.
        """
        if not self._profile_set:
            raise RuntimeError("_rebuild_alpha_psi: profile not set; call build_alpha_psi_arrays first")
        if self._R_dz is None:
            raise RuntimeError("_rebuild_alpha_psi: R_dz is None; call update() first")
        
        alpha_dead = self._alpha_dead
        alpha_active = self._alpha_active
        psi_dead = self._psi_dead
        psi_active = self._psi_active
        w = self._w
        
        # One scale height at dead zone radius
        H = np.interp(self._R_dz, self._R, self._H)
        if not np.isfinite(H) or H <= 0:
            raise ValueError(f"_rebuild_alpha_psi: invalid scale height at R_dz={self._R_dz} (H={H})")
        
        # Build tanh transition: 0 inside dead zone, 1 in active region
        transition = 0.5 * ( 1 + np.tanh((self._R - self._R_dz) / (w * H)) )
        
        # Build spatially-varying arrays
        alpha_arr = alpha_dead + (alpha_active - alpha_dead) * transition
        psi_arr = psi_dead + (psi_active - psi_dead) * transition
        
        # Clamp to the physical range spanned by the endpoints (guards against
        # floating-point overshoot) and enforce positivity / non-negativity
        alpha_lo, alpha_hi = min(alpha_dead, alpha_active), max(alpha_dead, alpha_active)
        psi_lo, psi_hi = min(psi_dead, psi_active), max(psi_dead, psi_active)
        alpha_arr = np.clip(alpha_arr, alpha_lo, alpha_hi)
        psi_arr = np.clip(psi_arr, psi_lo, psi_hi)
        
        # Final safety check: no NaN/inf leaked through
        if not np.all(np.isfinite(alpha_arr)):
            raise ValueError("_rebuild_alpha_psi: non-finite values in alpha array")
        if not np.all(np.isfinite(psi_arr)):
            raise ValueError("_rebuild_alpha_psi: non-finite values in psi array")
        
        # Assign validated arrays to the attributes the parent's thermal
        # balance reads (self._alpha_t and self._psi)
        self._alpha_t = alpha_arr
        self._psi = psi_arr

    def ASCII_header(self):
        """DeadZoneEOS header.

        Built from the base table header plus IrradiatedEOS-style fields, but
        using the SCALAR dead-zone alpha (self._alpha_t may be an array once a
        profile has been built, which must not be written into the header).
        """
        head = EOS_Table.ASCII_header(self)
        head += ', opacity: {}, T_extern: {}K, accrete: {}, Tmax: {}K'.format(
            self._kappa.__class__.__name__, self._Tc, self._accrete, self._Tmax)
        head += ', Mdot: {}'.format(self._Mdot)
        head += ', evolution_model: {}, r0: {}, r1: {}, r_floor: {}'.format(
            self._evolution_model, self._r0, self._r1, self._r_floor)
        head += ', t0: {}, t1: {}, t_initial_yr: {}, warm_start: {}'.format(
            self._t0, self._t1, self._t_initial_yr, self._warm_start)
        head += ', alpha_dead: {}, alpha_active: {}, psi_dead: {}, psi_active: {}, w: {}'.format(
            self._alpha_dead, self._alpha_active, self._psi_dead, self._psi_active, self._w)
        return head

    def HDF5_attributes(self):
        """Class information for HDF5 headers."""
        name, head = EOS_Table.HDF5_attributes(self)

        head["opacity"]         = self._kappa.__class__.__name__
        head["T_extern"]        = "{} K".format(self._Tc)
        head["accrete"]         = "{}".format(bool(self._accrete))
        head["Tmax"]            = "{} K".format(self._Tmax)
        head["Mdot"]            = "{}".format(self._Mdot)
        head["evolution_model"] = "{}".format(self._evolution_model)
        head["r0"]              = "{}".format(self._r0)
        head["r1"]              = "{}".format(self._r1)
        head["r_floor"]         = "{}".format(self._r_floor)
        head["t0"]              = "{}".format(self._t0)
        head["t1"]              = "{}".format(self._t1)
        head["t_initial_yr"]    = "{}".format(self._t_initial_yr)
        head["warm_start"]      = "{}".format(bool(self._warm_start))
        head["alpha_dead"]      = "{}".format(self._alpha_dead)
        head["alpha_active"]    = "{}".format(self._alpha_active)
        head["psi_dead"]        = "{}".format(self._psi_dead)
        head["psi_active"]      = "{}".format(self._psi_active)
        head["w"]               = "{}".format(self._w)

        return name, head

    @staticmethod
    def from_file(filename):
        import star

        star = star.from_file(filename)

        # Locate the DeadZoneEOS header line
        string = None
        with open(filename) as f:
            for line in f:
                if not line.startswith('#'):
                    raise AttributeError("Error: EOS type not found in header")
                elif "DeadZoneEOS" in line:
                    string = line
                    break
        if string is None:
            raise AttributeError("Error: DeadZoneEOS header not found")

        # Parse 'key: value' pairs. Keys may carry the class-name prefix on the
        # first field (e.g. 'DeadZoneEOS gamma'); normalise to the last token.
        raw = {}
        for item in string.lstrip('#').split(','):
            if ':' not in item:
                continue
            key, val = item.split(':', 1)
            raw[key.strip().split()[-1]] = val.strip()

        def num(key, default=None):
            """Parse a numeric field, tolerating a 'K' suffix and 'None'."""
            if key not in raw or raw[key] == 'None':
                return default
            return float(raw[key].replace('K', '').strip())

        kwargs = dict(
            evolution_model = raw.get('evolution_model', 'linear'),
            r0           = num('r0'),
            r1           = num('r1'),
            r_floor      = num('r_floor'),
            t0           = num('t0'),
            t1           = num('t1'),
            t_initial_yr = num('t_initial_yr', 0.0),
            Tc           = num('T_extern', 10.0),
            Tmax         = num('Tmax', 1500.0),
            mu           = num('mu', 2.4),
            gamma        = num('gamma', 1.4),
            accrete      = (raw.get('accrete', 'True') == 'True'),
            warm_start   = (raw.get('warm_start', 'True') == 'True'),
        )

        psi         = num('psi_dead', 0.0)
        Mdot        = num('Mdot')
        alpha_guess = num('alpha_dead', 1e-3)

        eos = DeadZoneEOS(star, psi, Mdot, alpha_guess=alpha_guess, **kwargs)

        # Restore the stored tanh profile so the spatial arrays can be rebuilt
        alpha_active = num('alpha_active')
        psi_active   = num('psi_active')
        w            = num('w', 1.0)
        if alpha_active is not None and psi_active is not None:
            eos._alpha_dead   = alpha_guess
            eos._alpha_active = alpha_active
            eos._psi_dead     = psi
            eos._psi_active   = psi_active
            eos._w            = w
            eos._profile_set  = True

        return eos

    

def from_file(filename):
    with open(filename) as f:
        for line in f:
            if not line.startswith('#'):
                raise AttributeError("Error: EOS type not found in header")
            elif "DeadZoneEOS" in line:
                return DeadZoneEOS.from_file(filename)
            elif "IrradiatedEOS" in line:
                return IrradiatedEOS.from_file(filename)      
            elif "SimpleDiscEOS" in line:
                return SimpleDiscEOS.from_file(filename)
            elif "LocallyIsothermalEOS" in line:
                return LocallyIsothermalEOS.from_file(filename)
            else:
                continue


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from .star import SimpleStar
    from .grid import Grid

    alpha = 1e-3
    star = SimpleStar(M=1.0, R=3.0, T_eff=4280.)

    active  = IrradiatedEOS(star, alpha)
    passive = IrradiatedEOS(star, alpha, accrete=False)
    marco   = IrradiatedEOS(star, alpha, kappa=opacity.Tazzari2016())

    powerlaw = LocallyIsothermalEOS(star, 1/30., -0.25, alpha)

    grid = Grid(0.1, 500, 1000, spacing='log')
    
    Sigma = 2.2e3 / grid.Rc**1.5

    amax = 10 / grid.Rc**1.5
    
    c  = { 'active' : 'r', 'passive' : 'b', 'marco' : 'm',
           'isothermal' : 'g' }
    ls = { 0 : '-', 1 : '--' }
    for i in range(2):
        for eos, name in [[active, 'active'],
                          [marco, 'marco'],
                          [passive, 'passive'],
                          [powerlaw, 'isothermal']]:
            eos.set_grid(grid)
            eos.update(0, Sigma, amax=amax)

            label = None
            if ls[i] == '-':
                label = name
                
            plt.loglog(grid.Rc, eos.T, c[name] + ls[i], label=label)
        Sigma /= 10
    plt.legend()
    plt.show()
    
                    