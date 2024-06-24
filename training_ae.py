import os
os.environ["KERAS_BACKEND"] = "jax"

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial 

import keras
from keras.callbacks import ModelCheckpoint
import jax_cfd.base as cfd

import models
import time_stepping as ts
import loss as lf


# setup problem and create grid
Lx = 2 * jnp.pi
Ly = 2 * jnp.pi
Nx = 128
Ny = 128
Re = 40.
vort_max = 25. # used to normalize fields in training 
T_unroll = 2.5

# data location 
data_loc = '/home/jacob/code/jax-cfd-data-gen/Re40test/'
file_front = 'vort_traj.'
files = [data_loc + file_front + str(n).zfill(4) + '.npy' for n in range(100)]

# define uniform grid 
grid = cfd.grids.Grid((Nx, Ny), domain=((0, Lx), (0, Ly)))

# estimate stable time step based on a "max velocity" using CFL condition
max_vel_est = 5.
dt_stable = cfd.equations.stable_time_step(max_vel_est, 0.5, 1./Re, grid) / 2.

# generate a trajectory function
dt_stable = np.round(dt_stable, 3)
trajectory_fn = ts.generate_trajectory_fn(Re, T_unroll + 1e-2, dt_stable, grid, t_substep=0.5)

# wrap trajectory function with FFTs to enable physical space -> physical space map
def real_to_real_traj_fn(vort_phys, traj_fn):
  # FT of space and select first (only) channel
  vort_rft = jnp.fft.rfftn(vort_phys, axes=(1,2))[...,0]
  _, traj_rft = traj_fn(vort_rft)
  # axes for FT move back because we now also have time dimension; add channel
  traj_phys = jnp.fft.irfftn(traj_rft, axes=(2,3))[...,jnp.newaxis]
  return traj_phys

real_traj_fn = partial(real_to_real_traj_fn, traj_fn=jax.vmap(trajectory_fn))
loss_fn = jax.jit(partial(lf.mse_and_traj, trajectory_rollout_fn=real_traj_fn))

# load training data
# TODO restore functionality with keras dataset
training_data = [np.load(file_name) for file_name in files]
training_data_ar = np.concatenate(training_data, axis=0)[..., np.newaxis] / vort_max

# build model
ae_model = models.ae_densenet_v7(Nx, Ny, 128)

checkpoint_callback = ModelCheckpoint(
    filepath='weights/sr_test_epoch_{epoch:02d}.weights.h5',
    save_weights_only=True,
    save_best_only=True,
    verbose=1
)

# train a few epochs on standard MSE prior to unrolling
ae_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-4),
    loss='mse', 
    metrics=[keras.losses.MeanSquaredError()]
)
ae_model.fit(training_data_ar, 
             training_data_ar, 
             batch_size=16, 
             validation_split=0.1,
             epochs=25)

# now use full loss 
ae_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-4),
    loss=loss_fn, 
    metrics=[keras.losses.MeanSquaredError()]
)
ae_model.fit(training_data_ar, 
             training_data_ar, 
             validation_split=0.1,
             callbacks=[checkpoint_callback], 
             batch_size=16, 
             epochs=100)
