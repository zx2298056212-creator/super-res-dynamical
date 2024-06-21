import os
os.environ["KERAS_BACKEND"] = "jax"

import jax
import jax.numpy as jnp
import numpy as np

import keras
from keras.callbacks import ModelCheckpoint
import jax_cfd.base as cfd

from functools import partial

import models
import time_stepping as ts
import loss as lf

# setup problem and create grid
Lx = 2 * jnp.pi
Ly = 2 * jnp.pi
Nx = 128
Ny = 128
Re = 40.
filter_size = 32 # how large are we average pooling?

data_loc = '/home/jacob/code/jax-cfd-data-gen/Re40test/'
file_front = 'vort_traj.'

grid = cfd.grids.Grid((Nx, Ny), domain=((0, Lx), (0, Ly)))

# estimate stable time step based on a "max velocity" using CFL condition
max_vel_est = 5.
dt_stable = cfd.equations.stable_time_step(max_vel_est, 0.5, 1./Re, grid) / 2.

# how we downsample -- not going to call in the functions atm so no 
# need to batch
def average_pool_trajectory(omega_traj, pool_width, pool_height):
  trajectory_length, Nx, Ny, Nchannels = omega_traj.shape
  assert Nx % pool_width == 0
  assert Ny % pool_height == 0

  omega_reshaped = omega_traj.reshape(
    (trajectory_length, Nx // pool_width, pool_width, Ny // pool_height, pool_height, Nchannels)
  )
  omega_pooled_traj = omega_reshaped.mean(axis=(2, 4))
  return omega_pooled_traj

pooling_fn = jax.jit(average_pool_trajectory, static_argnums=(1, 2))

# create downsampled data -- needs cleaning
files = [data_loc + file_front + str(n).zfill(4) + '.npy' for n in range(1000)]
vort_snapshots = [np.load(file_name)[::4] for file_name in files]
# add axis for channels (=1)
vort_snapshots = np.concatenate(vort_snapshots, axis=0)[..., np.newaxis]
np.random.shuffle(vort_snapshots)

vort_snapshots_coarse = average_pool_trajectory(vort_snapshots, filter_size, filter_size)
print(vort_snapshots.shape, vort_snapshots_coarse.shape)

# TODO compute N_grow above given filter size and Nx
super_model = models.super_res_v0(Nx // filter_size, Ny // filter_size, 32, N_grow=5)

checkpoint_callback = ModelCheckpoint(
    filepath='weights/sr_test_epoch_{epoch:02d}.weights.h5',
    save_weights_only=True,
    save_best_only=True,
    verbose=1
)

# train a few epochs on standard MSE prior to unrolling
super_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-4),
    loss='mse', 
    metrics=[keras.losses.MeanSquaredError()]
)
super_model.fit(vort_snapshots_coarse, vort_snapshots, 
                callbacks=[checkpoint_callback],
                batch_size=16, validation_split=0.1, epochs=50)


# now let's try and use our new loss
T_unroll = 2.5

# generate a trajectory function
dt_stable = np.round(dt_stable, 3)
trajectory_fn = ts.generate_trajectory_fn(Re, T_unroll + 1e-2, dt_stable, grid, t_substep=0.5)

# wrap trajectory function with FFTs to enable physical space -> physical space map
def real_to_real_traj_fn(vort_phys, traj_fn):
  vort_rft = jnp.fft.rfftn(vort_phys, axes=(1,2))[...,0]
  _, traj_rft = traj_fn(vort_rft)
  traj_phys = jnp.fft.irfftn(traj_rft, axes=(1,2))[...,jnp.newaxis]
  return traj_phys

real_traj_fn = partial(real_to_real_traj_fn, traj_fn=jax.vmap(trajectory_fn))
loss_fn = jax.jit(partial(lf.mse_and_traj, trajectory_rollout_fn=real_traj_fn, vort_max=1.))

super_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-4),
    loss=loss_fn, 
    metrics=[keras.losses.MeanSquaredError()]
)
super_model.fit(vort_snapshots_coarse, vort_snapshots, batch_size=16, epochs=5)