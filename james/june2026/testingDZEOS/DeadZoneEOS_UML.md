# DeadZoneEOS UML

If your viewer supports Mermaid, the class diagram is below.

```mermaid
classDiagram
direction TB

class EOS_Table {
  +gamma
  +mu
  +alpha
  +h
  +nu
  +set_grid(grid)
  +update(dt, Sigma, amax, star)
  +ASCII_header()
  +HDF5_attributes()
}

class IrradiatedEOS {
  +_Tc
  +_Tmax
  +_accrete
  +_psi
  +_e_rad
  +_kappa
  +_sigTc4
  +_T
  +_H
  +__init__(star, alpha_t, Tc, Tmax, mu, gamma, kappa, accrete, psi, e_rad)
  +update(dt, Sigma, amax, star)
  +ASCII_header()
  +HDF5_attributes()
  +from_file(filename)$
}

class DeadZoneEOS {
  +_Mdot
  +_alpha_guess
  +_evolution_model
  +_r0
  +_r1
  +_r_floor
  +_t0
  +_t1
  +_R_dz
  +_t_current_yr
  +_t_initial_yr
  +_profile_set
  +_alpha_dead
  +_alpha_active
  +_psi_dead
  +_psi_active
  +_w
  +_warm_start
  --
  +__init__(star, psi, Mdot, alpha_guess, evolution_model, r0, r1, r_floor, t0, t1, t_initial_yr, Tc, Tmax, mu, gamma, kappa, accrete, e_rad, warm_start)
  +update(dt, Sigma, amax, star)
  +alpha_from_Mdot_psi(disc, wind_model, Mdot_target, max_iterations, tol)
  +build_alpha_psi_arrays(alpha_active, alpha_dead, psi_dead, psi_active, w)
  +ASCII_header()
  +HDF5_attributes()
  +from_file(filename)$
  --
  -_compute_R_dz_linear(t)
  -_compute_R_dz_exponential(t)
  -_compute_R_dz_static(t)
  -_compute_R_dz(t)
  -_rebuild_alpha_psi()
}

class Star
class Disc
class WindModel {
  <<interface-like>>
  +viscous_velocity(disc)
}

EOS_Table <|-- IrradiatedEOS
IrradiatedEOS <|-- DeadZoneEOS

DeadZoneEOS ..> Star : initialized with
DeadZoneEOS ..> Disc : uses in alpha solver
DeadZoneEOS ..> WindModel : needs viscous_velocity()
```

## Plain-text fallback

DeadZoneEOS extends IrradiatedEOS, which extends EOS_Table.

- EOS_Table
  - Core EOS API: set_grid, update, ASCII_header, HDF5_attributes
- IrradiatedEOS
  - Adds irradiation/accretion thermal-balance fields and solver update
- DeadZoneEOS
  - Adds dead-zone state (R_dz evolution, profile scalars, warm_start)
  - Adds methods:
    - alpha_from_Mdot_psi
    - build_alpha_psi_arrays
    - _rebuild_alpha_psi
    - _compute_R_dz_{linear, exponential, static}
    - _compute_R_dz (dispatcher)
  - Overrides update, ASCII_header, HDF5_attributes
  - Provides from_file
