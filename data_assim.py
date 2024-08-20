import os
os.environ["KERAS_BACKEND"] = "jax"

import jax
import jax.numpy as jnp
import numpy as np
import optax

import jax_cfd.base as cfd
from functools import partial

import time_stepping as ts
import loss as lf
import interact_model as im

from scipy.ndimage import zoom

# setup problem and create grid
Lx = 2 * jnp.pi
Ly = 2 * jnp.pi
Nx = 128
Ny = 128
Re = 100.

# assimilation parameters
T_unroll = 1.5
M_substep = 8 # how many stable timesteps in one assimilation timestep
filter_size = 8

# hyper parameters + optimizer 
lr = 0.2
n_opt_step = 100
opt_class = optax.adam
file_number = 100 # a trajectory from which IC is extracted
snap_number = 0 # within trajectory

# (0) build grid, stable timestep etc
grid = cfd.grids.Grid((Nx, Ny), domain=((0, Lx), (0, Ly)))
max_vel_est = 5.
dt_stable = cfd.equations.stable_time_step(max_vel_est, 0.5, 1. / Re, grid) / 2.

# (1) load in high-res vorticity field
vort_init = jnp.load('/Users/jpage2/code/jax-cfd-data-gen/Re100test/vort_traj.' 
                     + str(file_number).zfill(4) 
                     + '.npy')[snap_number]

# (2) create forward trajectory and downsample
dt_stable = np.round(dt_stable, 3)
t_substep = M_substep * dt_stable
trajectory_fn = ts.generate_trajectory_fn(Re, T_unroll + 1e-2, dt_stable, grid, t_substep=t_substep)

def real_to_real_traj_fn(vort_phys, trajectory_fn):
  vort_rft = jnp.fft.rfftn(vort_phys, axes=(0,1))
  _, vort_traj_rft = trajectory_fn(vort_rft)
  return jnp.fft.irfftn(vort_traj_rft, axes=(1,2))

real_traj_fn = partial(real_to_real_traj_fn, trajectory_fn=trajectory_fn)
pooling_fn = jax.jit(partial(im.coarse_pool_trajectory, 
                             pool_width=filter_size, 
                             pool_height=filter_size))

vort_true_traj = real_traj_fn(vort_init)
# note add axis ("channel") then remove (coarse pooling designed for image convnet problems)
vort_true_coarse_traj = pooling_fn(vort_true_traj[..., jnp.newaxis])[..., 0]

# construct loss 
loss_fn = partial(lf.data_assim_vort, 
                  vort_traj_coarse_true=vort_true_coarse_traj,
                  trajectory_rollout_fn=real_traj_fn,
                  pooling_fn=pooling_fn)
loss_fn_jitted = jax.jit(loss_fn)
val_and_grad_fn = jax.value_and_grad(loss_fn_jitted)

# (3) setup initial condition
vort_init_coarse = pooling_fn(vort_init[jnp.newaxis, ..., jnp.newaxis])[0, ..., 0]
vort_pred_init = zoom(vort_init_coarse, filter_size, order=3)

# (4) setup optimiser
def update_guess(state_current, opt_state, optimizer):
  loss, grads = val_and_grad_fn(state_current)
  update_vec, opt_state = optimizer.update(grads, opt_state, state_current)
  state_current = optax.apply_updates(state_current, update_vec)
  return state_current, opt_state, loss
  
optimizer = opt_class(lr)
state_current = vort_pred_init

opt_state = optimizer.init(state_current)
for n in range(n_opt_step):
  state_current, opt_state, loss = update_guess(state_current, opt_state, optimizer)
  print("Step: ", n+1, "Loss: ", loss)

jnp.save(
  'assim_ex.npy', 
  jnp.concatenate([vort_init[..., jnp.newaxis],
                   vort_pred_init[..., jnp.newaxis], 
                   state_current[..., jnp.newaxis]], 
                   axis=-1)
                   )