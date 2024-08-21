import os
os.environ["KERAS_BACKEND"] = "jax"

DATA_SCALE = 4 # quirk of loading bad data -- TODO FIX DATASET

import jax
import jax.numpy as jnp
import numpy as np
import optax

import jax_cfd.base as cfd
from functools import partial

import time_stepping as ts
import interact_model as im
import da_optimisation as dao

from scipy.ndimage import zoom

# setup problem and create grid
Lx = 2 * jnp.pi
Ly = 2 * jnp.pi
Nx = 128
Ny = 128
Re = 100.

file_number = 500 # a trajectory from which IC is extracted
snap_number = 0 # within trajectory

velocity_assim = True # set False to assimilate vorticity

# assimilation parameters
T_unroll = 1.5
M_substep = 16 # how many stable timesteps in one assimilation timestep
filter_size = 16

# hyper parameters + optimizer 
lr = 0.2
n_opt_step = 100
opt_class = optax.adam

# (0) build grid, stable timestep etc
grid = cfd.grids.Grid((Nx, Ny), domain=((0, Lx), (0, Ly)))
max_vel_est = 5.
dt_stable = cfd.equations.stable_time_step(max_vel_est, 0.5, 1. / Re, grid) / 2.

# (1) load in high-res vorticity field
vort_init = jnp.load('/Users/jpage2/code/jax-cfd-data-gen/Re100test/vort_traj.' 
                     + str(file_number).zfill(4) 
                     + '.npy')[snap_number] / DATA_SCALE

# (2) create forward trajectory and downsample
dt_stable = np.round(dt_stable, 3)
t_substep = M_substep * dt_stable
trajectory_fn = ts.generate_trajectory_fn(Re, T_unroll + 1e-2, dt_stable, grid, t_substep=t_substep)

# wrapper to deal with the fact that traj fn in "interact_model" is over batches
def vort_traj_fn_wrapper(vort_phys, trajectory_fn):
  """ Add batch and channel dimensions, call routine. Note underlying
      trajectory function also gets batched via vmap """
  traj_phys = im.real_to_real_traj_fn(
    vort_phys[jnp.newaxis, ..., jnp.newaxis],
    jax.vmap(trajectory_fn)
    )[0,...,0]
  return traj_phys

real_traj_fn = jax.jit(
  partial(vort_traj_fn_wrapper, 
          trajectory_fn=trajectory_fn)
)
pooling_fn = jax.jit(partial(im.coarse_pool_trajectory, 
                             pool_width=filter_size, 
                             pool_height=filter_size))

if velocity_assim == True:
  vel_to_vort_fn = partial(im.compute_vort_traj, dx = Lx / Nx, dy = Ly / Ny)
  vort_to_vel_fn = partial(im.compute_vel_traj, dx = Lx / Nx, dy = Ly / Ny)
  vel_init = vort_to_vel_fn(vort_init[jnp.newaxis, ..., jnp.newaxis])[0, ...]
  vort_true_traj = real_traj_fn(vort_init)
  vel_true_traj = vort_to_vel_fn(vort_true_traj[..., jnp.newaxis])
  vel_true_coarse_traj = pooling_fn(vel_true_traj)

  # construct loss
  loss_fn = partial(dao.vel_loss,
                    vel_traj_coarse_true=vel_true_coarse_traj,
                    pooling_fn=pooling_fn,
                    trajectory_rollout_fn=real_traj_fn,
                    vel_to_vort_fn=vel_to_vort_fn,
                    vort_to_vel_fn=vort_to_vel_fn
                    )
  loss_fn_jitted = jax.jit(loss_fn)
  val_and_grad_fn = jax.value_and_grad(loss_fn_jitted)

  # (3) setup initial condition (downsampling)
  vel_init_coarse = pooling_fn(vel_init[jnp.newaxis, ...])[0, ...]
  vel_pred_init = zoom(vel_init_coarse, (filter_size, filter_size, 1), order=3)

  # (4) setup optimiser and iterate 
  optimizer = opt_class(lr)
  state_current = vel_pred_init
  opt_state = optimizer.init(state_current)
  for n in range(n_opt_step):
    state_current, opt_state, loss = dao.update_guess_vel(state_current, 
                                                          opt_state, 
                                                          optimizer,
                                                          val_and_grad_fn)
    print("Step: ", n+1, "Loss: ", loss)

  jnp.save(
    'assim_ex.npy', 
    jnp.concatenate([vel_init[..., jnp.newaxis],
                     vel_pred_init[..., jnp.newaxis], 
                     state_current[..., jnp.newaxis]], 
                     axis=-1)
                     )
else: 
  vort_true_traj = real_traj_fn(vort_init)
  # note add axis ("channel") then remove (coarse pooling designed for image convnet problems)
  vort_true_coarse_traj = pooling_fn(vort_true_traj[..., jnp.newaxis])[..., 0]

  # construct loss 
  loss_fn = partial(dao.vort_loss, 
                    vort_traj_coarse_true=vort_true_coarse_traj,
                    trajectory_rollout_fn=real_traj_fn,
                    pooling_fn=pooling_fn)
  loss_fn_jitted = jax.jit(loss_fn)
  val_and_grad_fn = jax.value_and_grad(loss_fn_jitted)

  # (3) setup initial condition
  vort_init_coarse = pooling_fn(vort_init[jnp.newaxis, ..., jnp.newaxis])[0, ..., 0]
  vort_pred_init = zoom(vort_init_coarse, filter_size, order=3)

  # (4) setup optimiser and iterate
  optimizer = opt_class(lr)
  state_current = vort_pred_init
  opt_state = optimizer.init(state_current)
  for n in range(n_opt_step):
    state_current, opt_state, loss = dao.update_guess_vort(state_current, 
                                                           opt_state, 
                                                           optimizer,
                                                           val_and_grad_fn)
    print("Step: ", n+1, "Loss: ", loss)


  jnp.save(
    'assim_ex.npy', 
    jnp.concatenate([vort_init[..., jnp.newaxis],
                     vort_pred_init[..., jnp.newaxis], 
                     state_current[..., jnp.newaxis]], 
                     axis=-1)
                     )