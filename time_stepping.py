import os
os.environ["KERAS_BACKEND"] = "jax"

import jax
import jax.numpy as jnp

import jax_cfd.base as cfd
import jax_cfd.spectral as spectral

from typing import Callable

def kolmogorov_ck13_step(
      Re: float, 
      grid: cfd.grids.Grid, 
      smooth: bool=True,
      damping: float=0.1
      ):
  wave_number = 4
  offsets = ((0, 0), (0, 0))
  # pylint: disable=g-long-lambda
  forcing_fn = lambda grid: cfd.forcings.kolmogorov_forcing(
      grid, k=wave_number, offsets=offsets)
  return spectral.equations.NavierStokes2D(
      viscosity = 1. / Re,
      grid = grid,
      drag = damping, 
      smooth = smooth,
      forcing_fn = forcing_fn)

def generate_time_forward_map(
    dt: float,
    Nt: int,
    grid: cfd.grids.Grid,
    Re: float,
    damping: float=0.1
) -> Callable[[jnp.ndarray], jnp.ndarray]:
  
  step_fn = spectral.time_stepping.crank_nicolson_rk4(
    kolmogorov_ck13_step(Re, grid, smooth=True, damping=damping), dt)

  time_forward_map = cfd.funcutils.repeated(jax.remat(step_fn), steps=Nt)
  return jax.jit(time_forward_map)

def generate_trajectory_fn(
    Re: float,
    T: float,
    dt_stable: float,
    grid: cfd.grids.Grid,
    t_substep: float=1.,
    damping: float=0.1
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    N_steps_total = jnp.floor(T / dt_stable)
    dt_exact = T / N_steps_total

    N_steps_per_substep = jnp.floor(t_substep / dt_exact)
    N_substeps = N_steps_total // N_steps_per_substep

    sub_step_fn = generate_time_forward_map(dt_exact, N_steps_per_substep, grid, Re, damping=damping)

    trajectory_fn = jax.jit(cfd.funcutils.trajectory(jax.remat(sub_step_fn), N_substeps))
    return trajectory_fn