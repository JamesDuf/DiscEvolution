from __future__ import print_function
import numpy as np
from DiscEvolution.brent import brentq
from DiscEvolution import opacity
from DiscEvolution.constants import *
import time 
import warnings

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
        alpha_guess,

        # Deadzone radius evolution
        evolution_model = 'linear',
        ionization_model = 'CR+XR',
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
        if evolution_model not in ('linear', 'exponential', 'static', 'ionization'):
            raise ValueError(f"Unknown evolution_model: {evolution_model}")
        self._evolution_model = evolution_model

        if ionization_model not in ('CR', 'XR', 'CR+XR'):
            raise ValueError(f"Unknown ionization_model: {ionization_model}")
        self._ionization_model = ionization_model
        
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

        # xe solver diagnostics 
        self._xe_prev = None                # will be used to pass previous xe to next iteration (warm_start-like but doesn't have a toggle)
        self._B_xe_prev = None
        self._C_xe_prev = None
        self._D_xe_prev = None
        self._zeta_prev = None               # total ionization rate [1/s], stored for output
        self._zeta_CR_prev = None            # cosmic-ray ionization rate [1/s]
        self._zeta_XR_prev = None            # X-ray ionization rate [1/s]
        self._eta_prev = None                # magnetic diffusivity [cm^2/s]
        self._Lambda_prev = None             # Elsasser number
        self._n_prev = None                  # midplane number density [1/cm^3]
        self._NH_prev = None                 # X-ray column density [1/cm^2]
        self._tau_prev = None                # X-ray optical depth
        self._JXR_prev = None                # X-ray attenuation factor
        self._sigmaXR_prev = None            # X-ray absorption cross-section [cm^2]

        # Timing steps to find bottleneck
        # profiling switches/counters
        self._timer = False                  # set to False to remove timing
        self._calls = 0
        self._t_total = 0.0
        self._t_xe = 0.0
        self._t_rest = 0.0

    # Cosmic Ray Ionization Rate (Alessi & Pudritz, 2017)
    def _zeta_CR(self, Sigma):
        zeta_0 = 1e-17        # cosmic-ray ionization rate [1/s] (Spitzer & Tomasko, 1968)
        Sigma_0 = 96.0        # CR attenuation length [g/cm^2] (Spitzer & Tomasko, 1968)

        zeta = (zeta_0/2.0) * np.exp(-(Sigma)/(Sigma_0)) # Sigma passed is already in cgs so ok
        
        return zeta #cgs

    # X Ray Ionization Rate (Alessi & Pudritz, 2017)
    def _zeta_XR(self, Sigma, source_R_Rsun = 12.0 , source_z_Rsun = 12.0, L_x_erg_s = 1e30, kTx_keV=4, E_keV=4, deltaE_eV=13.6): 
        # Lx = X ray luminostity of the star
        # kTx =  average xray energy taken to be 12 Rsol above midplane and at r = 12 Rsol to represent magnetospheric accretion onto the star 
        # E = primary electron energy
        # delta E = energy to make an ion pair
        # d = some point on the disc surface (compute from dr = |R_Source - R| and dz = |Zsource - Zsurface|) where Zsurface is taken to be 4 times the scale height (A & M, 2003)
        
        keV_to_erg = 1.602176634e-9

        source_R = source_R_Rsun * Rsun     # cm
        source_z = source_z_Rsun * Rsun     # cm
        R_cm = self._R * AU                 # cm
        
        # Approximate Chiang-style disk surface height
        # z_surface = 4.0 * self._H * AU      # cm 

        # Vector from X-ray source to disk surface
        dr = R_cm - source_R                # cm    
        # dz = z_surface - source_z           # cm
        dz = source_z 

        # Straight-line distance from source to disk surface
        d = np.sqrt(dr**2 + dz**2)          # cm

        sigma_XR_val = self._sigma_XR(E_keV)                                                                                # cm^2

        primary_ionizations_term = ( L_x_erg_s / (kTx_keV * keV_to_erg * 4 * np.pi * d**2) ) * sigma_XR_val                 # s^-1

        secondary_electrons_production_term = kTx_keV * 1000 / deltaE_eV                                                  # unitless

        NH  = self._NH_XR(Sigma, source_R_Rsun=source_R_Rsun, source_z_Rsun=source_z_Rsun)                                 # cm^-2
        tau = NH * sigma_XR_val                                                                                             # unitless
        JXR = self._J_XR(tau)                                                                                               # unitless
        attenuation_term = JXR

        # Store intermediates so they don't need to be recomputed for output
        self._sigmaXR_prev = sigma_XR_val
        self._NH_prev = NH
        self._tau_prev = tau
        self._JXR_prev = JXR

        zeta =  primary_ionizations_term * secondary_electrons_production_term * attenuation_term                         # s^-1
        return zeta 

    def _J_XR(self, tau, npts=300, x0 = 1, x1 = 100):      
        # x = E / E_keV                                                 # dimensionless energy parameter, E is primary electron energy 
        
        n = 2.81                                                        # Glassgold et al. 1997
        x = np.logspace(np.log10(x0), np.log10(x1), num=npts)           # Integral bounds from x0 = 1 to x1 = +inf -> use 100   from Matsumura & Pudritz 2003
        
        tau = np.atleast_1d(tau)                                        # Convert tau to an array  
        J = np.empty_like(tau, dtype=float)

        for i, tau_i in enumerate(tau):
            # clip to prevent overflow errors, force always negative to combat sign errors (which shouldn't happen anyways)
            exponent = np.clip(-x - tau_i * x**(-n), -700, 0)

            # integrand 
            integrand = x**(-n) * np.exp(exponent)

            #integrate 
            J[i] = np.trapezoid(integrand, x=x)

        return J                                                        # unitless

    def _sigma_XR(self, E_keV=4):
        sigma0 = 8.5E-23                        # cm^2
        n = 2.81                                # Glassgold et al. 1997

        sigma = sigma0 * (E_keV)**(-n)          # cm^2
        return sigma                            

    def _tau_XR(self, Sigma, source_R_Rsun, source_z_Rsun, E_keV=4):

        tau = self._NH_XR(Sigma, source_R_Rsun, source_z_Rsun) * self._sigma_XR(E_keV)                # unitless
        return tau                             

    def _NH_XR(self, Sigma, source_R_Rsun, source_z_Rsun, grazing_angle='trig'):

        Nperp = Sigma / (2.0 * self._mu * m_H)    # Integral reduces to this when using the midplane as lower bound (z = 0)

        if grazing_angle == '2003':
            alpha_prime = self._alpha_prime_XR_2003(source_R_Rsun, source_z_Rsun)

        if grazing_angle == 'trig':
            alpha_prime = self._alpha_prime_XR_trig(source_R_Rsun, source_z_Rsun)

        if grazing_angle not in ('trig', '2003'):
            raise ValueError("Invalid grazing-angle model choice")

        sin_alpha_prime = np.sin(alpha_prime)
        
        return Nperp / sin_alpha_prime


    def _alpha_prime_XR_2003(self, source_R_Rsun, source_z_Rsun):

        source_R = source_R_Rsun * Rsun                    # cm
        source_z = source_z_Rsun * Rsun                    # cm
        Rstar = self._star.Rs * Rsun                       # cm

        # disk surface def as 4 times P scale height
        H = 4.0 * self._H * AU                             # cm

        # disk radius
        a = self._R * AU                                   # cm

        dlnH_dlna = np.gradient(np.log(H), np.log(a))      # unitless

        # Chiang 2001
        alpha = np.arctan( dlnH_dlna * (H / a) ) - np.arctan( H / a ) + np.arcsin( (4 * Rstar)/( 3*np.pi*a) ) 

        # Matsumura & Pudritz 2003
        beta = np.arctan( dlnH_dlna * (H / a) )

        # Matsumura & Pudritz 2003
        gamma = np.arctan( (H - 0.5*Rstar)/(a - 0.5*np.sqrt(3)*Rstar) ) - np.arctan( (H - source_z)/(a - source_R) )

        alpha_prime = alpha - beta + gamma
        return alpha_prime

    def _alpha_prime_XR_trig(self, source_R_Rsun, source_z_Rsun):

        source_R = source_R_Rsun * Rsun     # cm
        source_z = source_z_Rsun * Rsun     # cm
        R_cm = self._R * AU                 # cm

        # Approximate Chiang-style disk surface height
        # z_surface = 4.0 * self._H * AU      # cm 

        # Vector from X-ray source to disk surface
        dr = R_cm - source_R                # cm    
        # dz = z_surface - source_z           # cm
        dz = source_z                       # cm

        # Straight-line distance from source to disk surface
        # d = np.sqrt(dr**2 + dz**2)          # cm

        # alpha_prime is the angle between ray and midplane
        # sin_alpha_prime = dz / d
        # alpha_prime = np.arcsin(sin_alpha_prime)

        alpha_prime = np.arctan2(dz, dr)

        return alpha_prime

    def _rho_mid(self, Sigma):
        rho_mid = Sigma / (np.sqrt(2*np.pi) * self._H * AU)   # g/cm^3
        return rho_mid
    
    def _n_density(self, Sigma):
        rho_mid = self._rho_mid(Sigma)
        n = rho_mid / (self._mu * m_H) 
        return n   

    # xe polynomial stuff (Alessi & Pudritz, 2017)
    def _xe(self,n, zeta, xe_previous=None):
        """ 
        Compute the ionization fraction at a each radius. Return xe array. 
         """

        T = self._T 
        xe = np.zeros_like(T, dtype=float)

        beta_d = 2.0 * 1e-6 * (T**-0.5)      # cm^3/s
        beta_r = 3.0 * 1e-11 * (T**-0.5)     # cm^3/s
        beta_t = 3.0 * 1e-9                 # cm^3/s
        xm = 0.0011                         # Metal fraction 

        # --------- Newton in log-space --------
        # Coefficients + constant  ->  A(xe**3) + B(xe**2) + C(xe) + D = 0
        B = (beta_t / beta_d) * xm
        C = (-zeta) / (beta_d * n)
        D = (-zeta * beta_t * xm) / (beta_d * beta_r * n)
        self._B_xe_prev = B
        self._C_xe_prev = C
        self._D_xe_prev = D

        # Fast solution if not ionized anywhere, zeta = 0 -> zero ionization rate -> zero ionization fraction 
        # We do this since the polynomial would still return a non-zero value despite zeta == 0
        # This removes unnecessary computations 
        # TODO: check if this is correct approach 
        active = zeta > 0.0                  
        if not np.any(active):              
            return xe                       
        Ba = B[active]
        Ca = C[active]
        Da = D[active]

        # Only allow values 10^-40 < xe < 1.0
        y_lower = np.full_like(Ba, -40.0)
        y_upper = np.zeros_like(Ba)

        # Initial guess
        if xe_previous is not None:
            previous = np.asarray(xe_previous)[active]
            y = np.log10(np.clip(previous, 1.0e-40, 1.0))
        else:
            # Use the two limiting ionization estimates
            xe_molecular = np.sqrt(zeta[active] / (beta_d[active] * n[active]))
            xe_metal = np.sqrt(zeta[active] / (beta_r[active] * n[active]))

            # Geometric mean of the limiting estimates
            y = 0.5 * (np.log10(xe_molecular)+ np.log10(xe_metal))
            y = np.clip(y, y_lower, y_upper)

        ln10 = np.log(10.0)

        for _ in range(20):
            x = 10.0**y
            f = x**3 + Ba*x**2 + Ca*x + Da      
            fp = 3.0*x**2 + 2.0*Ba*x + Ca  

            # Update the bracket using the current point
            y_upper = np.where(f > 0.0, y, y_upper)         # f > 0: current pt is above the root, so use it as upper bound
            y_lower = np.where(f <= 0.0, y, y_lower)        # f <= 0: current pt is below the root, so use it as lower bound

            # Derivative after change of variable
            derivative_log = ln10 * x * fp 

            # Newton Step
            with np.errstate(divide="ignore", invalid="ignore"):
                y_newton = y - f / derivative_log

            # Reject unsafe Newton steps, replace those points with bisection
            unsafe = (
                ~np.isfinite(y_newton)                      # reject NaNs 
                | (np.abs(derivative_log) < 1.0e-300)       # reject derivative too close to zero
                | (y_newton <= y_lower)                     # reject below lower bound 
                | (y_newton >= y_upper)                     # reject above upper bound
            )

            y_bisection = 0.5 * (y_lower + y_upper)

            # Safeguard 
            y_new = np.where(
                unsafe,
                y_bisection,
                y_newton
            )

            # Check if EVERY cell has converged: diff between previous step and this one is smaller than 1e-12
            if np.max(np.abs(y_new - y)) < 1.0e-12:
                y = y_new
                break

            # If not all converged yet, repeat 
            y = y_new

        # Convert back from log space (revert change of variable)
        xe_active = 10.0**y

        # Verify the solutions
        # Residuals 
        residual = np.abs(
            xe_active**3
            + Ba*xe_active**2
            + Ca*xe_active
            + Da
        )

        # Scale
        scale = (
            np.abs(xe_active**3)
            + np.abs(Ba*xe_active**2)
            + np.abs(Ca*xe_active)
            + np.abs(Da)
        )

        # Check that all solutions is posdef, and close enough to true solution
        valid = (
            np.isfinite(xe_active)
            & (xe_active > 0.0)
            & (
                residual
                <= 1.0e-6 * np.maximum(scale, 1.0e-300)
            )
        )

        # if not np.all(valid):
        #     bad = np.where(~valid)[0]
        #     print(f"xe residual check failed at {len(bad)} cell(s); "
        #         f"R={self._R[active][bad]}, xe={xe_active[bad]}, "
        #         f"residual={residual[bad]}, scale={scale[bad]}")
        #     raise ValueError("Electron-fraction solver failed to converge")

        # if not np.all(valid):
        #     n_bad = np.sum(~valid)
        #     import warnings
        #     warnings.warn(f"_xe: {n_bad} cell(s) failed residual check "
        #                    f"(likely low-density outer-disc cells)")
        
        # Place our solutions in the radial array, but leave the spots where zeta <= 0 zero 
        xe[active] = xe_active

        return xe



        # # -------- Faster solver with Vectorized Cardano Formula -------- -> gave negative roots 
        # # Coefficients + constant  ->  A(xe**3) + B(xe**2) + C(xe) + D = 0
        # # A = 1.0                           # unused with Cardano
        # B = (beta_t / beta_d) * xm
        # C = (-zeta) / (beta_d * n)
        # D = (-zeta * beta_t * xm) / (beta_d * beta_r * n)

        # active = zeta > 0.0                 # zeta = 0 gives xe = 0
        # if not np.any(active):              
        #     return xe                       # Fast solution if not ionized anywhere

        # Bb = B[active]
        # Cc = C[active]
        # Dd = D[active]

        # # Convert to depressed cubic
        # # y^3 + p*y + q = 0  with  xe = y - B/3
        # p = Cc - (Bb**2 / 3.0)
        # q = 2.0 * (Bb**3 / 27.0) - (Bb * Cc / 3.0) + Dd

        # discriminant = (q / 2.0)**2 + (p / 3.0)**3
        # roots_positive = np.empty_like(Bb)

        # # Account for small round-off errors near discriminant = 0
        # scale = np.maximum(
        #     (q / 2.0)**2 + np.abs((p / 3.0)**3),
        #     np.finfo(float).tiny
        # )

        # one_real = discriminant >= (
        #     -100.0 * np.finfo(float).eps * scale
        # )

        # # One real root
        # sqrt_disc = np.sqrt(np.maximum(discriminant[one_real], 0.0))

        # y = (
        #     np.cbrt(-q[one_real] / 2.0 + sqrt_disc)
        #     + np.cbrt(-q[one_real] / 2.0 - sqrt_disc)
        # )

        # roots_positive[one_real] = y - Bb[one_real] / 3.0

        # # Three real roots: calculate all three and select the positive one
        # three_real = ~one_real

        # if np.any(three_real):
        #     p3 = p[three_real]
        #     q3 = q[three_real]
        #     B3 = Bb[three_real]

        #     amplitude = 2.0 * np.sqrt(-p3 / 3.0)

        #     argument = -q3 / (
        #         2.0 * np.sqrt(-(p3 / 3.0)**3)
        #     )
        #     argument = np.clip(argument, -1.0, 1.0)

        #     theta = np.arccos(argument)

        # candidate_roots = np.stack([
        #     amplitude * np.cos(theta / 3.0) - B3 / 3.0,
        #     amplitude * np.cos((theta + 2.0*np.pi) / 3.0)
        #     - B3 / 3.0,
        #     amplitude * np.cos((theta + 4.0*np.pi) / 3.0)
        #     - B3 / 3.0
        # ])

        # positive_candidates = np.where(
        #     candidate_roots > 0.0,
        #     candidate_roots,
        #     np.nan
        # )

        # selected = np.nanmax(positive_candidates, axis=0)

        # if np.any(~np.isfinite(selected)):
        #     raise ValueError("Failed to find a positive electron-fraction root using Vectorized Cardano Solver.")

        # roots_positive[three_real] = selected

        # if np.any(roots_positive <= 0.0):
        #     raise ValueError("Vectorized Cardano Solver returned a non-positive electron fraction.")

        # xe[active] = roots_positive

        # return xe




        # --------- Slower original roots solver --------
        #TODO:used as backup option
        # for i in range(len(T)):
        #     # Coefficients + constant  ->  A(xe**3) + B(xe**2) + C(xe) + D = 0
        #     A = 1.0
        #     B = (beta_t / beta_d[i]) * xm
        #     C = (-zeta[i]) / (beta_d[i] * n[i])
        #     D = (-zeta[i] * beta_t * xm) / (beta_d[i] * beta_r[i] * n[i])

        #     # solve
        #     roots = np.roots([A,B,C,D])

        #     # Keep roots whose imaginary part is essentially zero
        #     real_roots = roots.real[np.isclose(roots.imag, 0.0, atol=1e-12)]

        #     # Keep only positive roots (cannot have ion fraction < 0)
        #     positive_roots = real_roots[real_roots > 0.0]

        #     if len(positive_roots) != 1: 
        #         raise ValueError(f'xe solver error at radius {self._R[i]} AU: Exected to find only one positive root but found {roots}')

        #     xe[i] = positive_roots[0]

        # return xe

    # Magnetic Diffusivity 
    def _eta(self, T, xe): 
        eta = (234.0 * (T**0.5))/(xe)
        return eta  #cm^2/s
    
    def _Elsasser(self, eta, Omega, cs): 
        Lambda = (self._alpha_t * (cs**2))/(eta * Omega)
        return Lambda 

    def update(self, dt, Sigma, amax=1e-5, star=None, ionization_model=None):
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

        if ionization_model is None:
            ionization_model = self._ionization_model

        # In ionization mode, first iteration doesn't have a Temperature Profile,
        # so call super with dt = 0.0 (instead of t0 + dt like in next step) to get a Temperature profile
        # TODO: check this makes sense?  -> it doesnt evolve the disk, only solves the thermal balance with brent and update thermodynamics values like H, nu, cs, kappa, Pr 
        if self._evolution_model == "ionization" and self._T is None:
            super(DeadZoneEOS, self).update(0.0, Sigma, amax=amax, star=star)

        # Increment absolute time 
        self._t_current_yr += dt / yr
        
        # Update dead zone radius at current time
        self._R_dz = self._compute_R_dz(self._t_current_yr, Sigma, ionization_model)
        
        # Rebuild the spatial alpha/psi arrays so the viscosity structure
        # follows the moving dead zone (only once a profile has been set)
        if self._profile_set:
            self._rebuild_alpha_psi()

        #TODO: In ionization mode, if disable warm start it won't have a Temperature profile and I'm not sure how to handle that case. Prevent it for now 
        if self._warm_start == False and self._evolution_model == "ionization":
            raise ValueError('Cannot disable warm_start while using ionization deadzone model.')

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
        Linearly move dead zone radius inward from r0 to r1.
        
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
        if not (self._r_floor <= self._r1 < self._r0):
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

    def _compute_R_dz_ionization(self, Sigma, ionization_model=None):
        """
        Return the dead zone radius computed from ionization fraction.
        """

        if ionization_model is None:
            ionization_model = self._ionization_model

        if self._timer:
            t0 = time.perf_counter()                      # toggle event timer

        R     = self._R                                   # AU
        T     = self._T                                   # K 
        Omega = Omega0 * self._star.Omega_k(R)            # 1/s
        cs    = np.sqrt(GasConst * T / self._mu)          # cm/s

        n = self._n_density(Sigma)  
        self._n_prev = n

        # Always compute both components separately so CR and XR can be inspected independently
        zeta_CR = self._zeta_CR(Sigma)
        zeta_XR = self._zeta_XR(Sigma)
        self._zeta_CR_prev = zeta_CR
        self._zeta_XR_prev = zeta_XR

        # Ionization rate calculation
        if ionization_model == 'CR':             
            zeta = zeta_CR
        if ionization_model == 'XR':             
            zeta = zeta_XR
        if ionization_model == 'CR+XR':             
            zeta = zeta_CR + zeta_XR
        if ionization_model not in ('CR', 'XR', 'CR+XR'):
            raise ValueError("Invalid ionization model choice")

        self._zeta_prev = zeta                                # save zeta to pass to output

        if self._timer:
            tx0 = time.perf_counter()                         # timer toggle start of roots solver

        xe  = self._xe(n, zeta, xe_previous=self._xe_prev)    # solves polynomial
        self._xe_prev = xe                                    # save xe to pass to next iteration

        if self._timer:
            tx1 = time.perf_counter()                         # timer toggle end of roots solver

        eta  = self._eta(T, xe)                          
        self._eta_prev = eta
        
        Lambda = self._Elsasser(eta, Omega, cs)           # Lambda(R) -> find where it crosses unity
        self._Lambda_prev = Lambda
        # Deadzone mask
        # deadzone = Lambda <= 1.0

        # if deadzone.any() == False:
        #     return R[0]                                    # no dead zone at so return Rdz = R_in

        # if deadzone.all() == True:
        #     return R[-1]                                     # entire disc is dead so return Rdz = R_out

        # Rdz = R[deadzone].max()                            # outer edge of dead zone marks Rdz -> this used to cause issues because no interpolation
        
        # As long as Lambda increases monotonically, interpolate directly for Lambda = 1
        # Rdz = np.interp(1.0, Lambda, R)

        # TODO: add monoticity check perhaps
        Rdz = self._find_deadzone_radius(R, Lambda)         # Check for monotonicity and find the appropriate crossing


        # Timer diagnostics
        if self._timer:
            t1 = time.perf_counter()
            self._calls += 1
            self._t_total += (t1 - t0)
            self._t_xe += (tx1 - tx0)
            self._t_rest += (t1 - t0) - (tx1 - tx0)
            if self._calls % 200 == 0:
                avg_ms = 1e3 * self._t_total / self._calls
                xe_pct = 100.0 * self._t_xe / self._t_total
                print(f"[ionization step timer] calls={self._calls} avg={avg_ms:.3f} ms  xe={xe_pct:.1f}%")

        return Rdz

    def _find_deadzone_radius(self, R, Lambda):
        """
        Identify the dead zone radius by finding all regions where Lambda < 1
        and selecting the right edge of the deepest deadzone.
        
        For a disc with potentially multiple dead zones (due to non-monotonic Lambda),
        this method identifies all contiguous regions where Lambda < 1, measures
        the depth of each (min(Lambda)), and returns the outer radius of
        the deepest region (interpolates to find the exact unity crossing).
        
        Parameters
        ----------
        R : array
            Radii (AU)
        Lambda : array
            Elsasser number at each radius
            
        Returns
        -------
        Rdz : float
            Dead-zone radius (AU) - the right edge of the deepest deadzone
        """
        
        R = np.asarray(R, dtype=float)
        Lambda = np.asarray(Lambda, dtype=float)
        
        # Sort by radius (should already be sorted, but be safe)
        order = np.argsort(R)
        R = R[order]
        Lambda = Lambda[order]
        
        # Create a mask for dead zones (Lambda < 1)
        dead_mask = Lambda < 1.0
        
        # Handle edge cases
        if not np.any(dead_mask):
            return R[0]  # No dead zone, return inner radius
        
        if dead_mask.all():
            return R[-1]  # Entire disc is dead, return outer radius

        # Find all contiguous deadzone regions by detecting transitions
        dead_int = dead_mask.astype(int)        # Converts True / False to 1 / 0
        transitions = np.diff(dead_int)         # Diff between cells: +1 = active -> dead , -1 = dead -> active, 0 no change
        
        # Find start and end indices of dead regions
        dead_starts = np.where(transitions == 1)[0] + 1  # Start (active -> dead)           # +1 because diff gives the trans betw two pts st trans index = 0 means change from idx 0 to 1 -> dz starts at idx 1
        dead_ends = np.where(transitions == -1)[0]  # End (dead -> active)
        
        # Handle boundary cases where disc starts or ends in dead zone
        if dead_mask[0]:
            dead_starts = np.insert(dead_starts, 0, 0)
        if dead_mask[-1]:
            dead_ends = np.append(dead_ends, len(R) - 1)
        
        # Count number of deadzones 
        n_deadzones = len(dead_starts) 
        if n_deadzones > 1:
            ranges = [
                f"{R[start_idx]:.4f}-{R[end_idx]:.4f} AU"
                for start_idx, end_idx in zip(dead_starts, dead_ends)
            ]

            warnings.warn(
                "Multiple dead zones detected. "
                f"t = {self._t_current_yr:.4e} yr. "
                ,
                RuntimeWarning
            )                 

        # Find the deepest deadzone (lowest min Lambda)
        deepest_min_lambda = np.inf
        deepest_right_idx = None
        
        for start_idx, end_idx in zip(dead_starts, dead_ends):          # Loops over each DZ, each has a first dead idx and last dead idx
            min_lambda = np.min(Lambda[start_idx:end_idx + 1])          # gets all the Lambda values in the DZ and finds the smallest one
            
            if min_lambda < deepest_min_lambda:                         # Check if this DZ min Lambda is smaller than the previous one, if so replace it 
                deepest_min_lambda = min_lambda
                deepest_right_idx = end_idx
        
        # Safety fallback
        if deepest_right_idx is None:
            raise RuntimeError("Failed to identify a dead-zone region despite dead_mask containing True values.")

        i = deepest_right_idx

        # If the deepest dead zone reaches the outer boundary,
        # there is no active point to interpolate to.
        if i == len(R) - 1:
            return R[-1]                                            # Prevents out of bounds error

        # Interpolate the crossing Lambda = 1
        # Right edge should be dead -> active:
        # Lambda[i] < 1 and Lambda[i+1] >= 1
        if Lambda[i] < 1.0 and Lambda[i + 1] >= 1.0:                # Check if right edge is a crossing from dead to active (check if dead at i and active at i+1)   

            r_cross = np.interp(                                    
                1.0,    
                [Lambda[i], Lambda[i + 1]],
                [R[i], R[i + 1]]                                    # linearly interpolates to find where Lambda = 1 between the last dead point and the first active point
            )

            return r_cross                                          # Return interpolated radius 

        raise RuntimeError("Deepest dead-zone right edge was not a valid dead-to-active transition.")

    def _compute_R_dz(self, t, Sigma=None, ionization_model=None):
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

        if ionization_model is None:
            ionization_model = self._ionization_model

        if self._evolution_model == 'linear':
            return self._compute_R_dz_linear(t)
        elif self._evolution_model == 'exponential':
            return self._compute_R_dz_exponential(t)
        elif self._evolution_model == 'static':
            return self._compute_R_dz_static(t)
        elif self._evolution_model == 'ionization':
            return self._compute_R_dz_ionization(Sigma, ionization_model) # Sigma available in update() so ok
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

    def build_alpha_psi_arrays2(self, alpha_SS_DZ, alpha_SS_AZ, psi_DZ, w=1.0):
        
        # Resolve the dead-zone scalars. Prefer an explicit argument; otherwise
        # use the stored scalar (if a profile was already set) or capture from
        # the current scalar attribute. Never re-read once these are arrays.
        if alpha_SS_DZ is None:
            alpha_dead = self._alpha_dead if self._profile_set else self._alpha_t
        if psi_DZ is None:
            psi_dead = self._psi_dead if self._profile_set else self._psi
        
        # Validate scalar inputs before storing
        for name, val in (('alpha_dead', alpha_dead), ('alpha_active', alpha_active),
                          ('psi_dead', psi_dead)):
            if not np.isscalar(val) or not np.isfinite(val):
                raise ValueError(f"build_alpha_psi_arrays: {name} must be a finite scalar (got {val!r})")
        if alpha_dead <= 0 or alpha_active <= 0:
            raise ValueError("build_alpha_psi_arrays: alpha values must be positive")
        if psi_dead < 0:
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
    
                    