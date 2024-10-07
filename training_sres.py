# TODO fix to work with either custom training step or tf.data.Dataset for augmentation
# TODO dynamic training increasing T_unroll <<< then release 
import os
os.environ["KERAS_BACKEND"] = "jax"

DATA_SCALE = 4 # quirk of loading bad data -- TODO FIX DATASET

import jax
import jax.numpy as jnp
import numpy as np

import yaml

import keras
import jax_cfd.base as cfd

from functools import partial

import models
import time_stepping as ts
import loss as lf
import interact_model as im
import sym_augment as sa

def load_config(path):
  with open(path, 'r') as file:
    return yaml.safe_load(file)

# Load configuration
config = load_config('config.yaml')
general = config['general']
grid_params = config['grid']
train_params = config['training']

# Use parameters from config file
data_loc = general['data_location']
weight_loc = general['weight_location']
file_front = general['file_prefix']
file_end = general['file_end']
n_files = general['n_files']
n_fields = general['n_fields']
loss_name = general['loss_fn']

Nx = grid_params['Nx']
Ny = grid_params['Ny']
Re = grid_params['Re']

filter_size = train_params['filter_size']
n_grow = train_params['n_grow']
T_unroll = train_params['T_unroll']
M_substep = train_params['M_substep']
batch_size = train_params['batch_size']
lr_mse = train_params['lr_mse']
lr_traj = train_params['lr_traj']
nval = train_params['nval']
n_mse_steps = train_params['n_mse_steps']
n_traj_steps = train_params['n_trajectory_steps']
alpha = train_params['alpha']

# setup problem and create grid
Lx = 2 * jnp.pi
Ly = 2 * jnp.pi

grid = cfd.grids.Grid((Nx, Ny), domain=((0, Lx), (0, Ly)))

# estimate stable time step based on a "max velocity" using CFL condition
max_vel_est = 5.
dt_stable = cfd.equations.stable_time_step(max_vel_est, 0.5, 1. / Re, grid) / 2.

# create downsampled data -- needs cleaning
files = [data_loc + file_front + str(n).zfill(4) + file_end for n in range(n_files)]
vort_snapshots = [np.load(file_name)[::2] for file_name in files]
print("Note halving the number of raw snapshots in training. ")
# add axis for channels (=1)
snapshots = np.concatenate(vort_snapshots, axis=0)[..., np.newaxis] / DATA_SCALE
np.random.shuffle(snapshots)

if n_fields > 1:
  snapshots = im.compute_vel_traj(snapshots, Lx / Nx, Ly / Ny)

snapshots_train = snapshots[:-nval]
snapshots_val = snapshots[-nval:]
snapshots_val_coarse = im.coarse_pool_trajectory(snapshots_val, filter_size, filter_size)

# batch the velocity/vorticity functions
vel_to_vort_fn = jax.jit(
  partial(im.compute_vort_traj, dx=Lx / Nx, dy=Ly / Ny)
  )
vort_to_vel_fn = jax.jit(
  partial(im.compute_vel_traj, dx=Lx / Nx, dy=Ly / Ny)
  )
vel_to_vort_fn_batched = jax.vmap(vel_to_vort_fn)
vort_to_vel_fn_batched = jax.vmap(vort_to_vel_fn)

# train a few epochs on standard MSE prior to unrolling
if n_fields == 1:
  super_model = models.super_res_vel_v3(Nx // filter_size,
                                        Ny // filter_size, 
                                        32, 
                                        N_grow=n_grow, 
                                        input_channels=n_fields)
  super_model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=lr_mse),
      loss='mse', 
      metrics=[keras.losses.MeanSquaredError()]
  )
else:
  super_model = models.super_res_vel_v3(Nx // filter_size,
                                        Ny // filter_size, 
                                        32, 
                                        N_grow=n_grow, 
                                        input_channels=n_fields)
  #loss_mse = jax.jit(partial(lf.mse_vel_and_vort, 
  #                           vel_to_vort_fn=vel_to_vort_fn_batched))
  super_model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=lr_mse),
      loss='mse', #loss_mse, 
      metrics=[keras.losses.MeanSquaredError()]
  )

# min val loss init
min_val_loss = np.inf

for n in range(n_mse_steps):
  print("MSE step: ", n)
  # data augmentation at each iteration through the data (a fudge due to issues with jax/tf map compat)
  snapshots_train = np.array([sa.translate_x(sa.shift_reflect_y(field)) for field in snapshots_train])
  snapshots_train_coarse = im.coarse_pool_trajectory(snapshots_train, filter_size, filter_size)

  history = super_model.fit(snapshots_train_coarse, 
                            snapshots_train, 
                            batch_size=batch_size,
                            validation_data=(snapshots_val_coarse, snapshots_val), epochs=1)

  current_val_loss = history.history['val_loss'][-1]

  # save model weights if val loss improved
  if current_val_loss < min_val_loss:
    min_val_loss = current_val_loss
    super_model.save_weights(weight_loc + 'sr_best_MSE.weights.h5')

# generate a trajectory function (for vorticity)
dt_stable = np.round(dt_stable, 3)
t_substep = dt_stable * M_substep
trajectory_fn = ts.generate_trajectory_fn(Re, T_unroll + 1e-2, dt_stable, grid, t_substep=t_substep)
real_traj_fn = partial(im.real_to_real_traj_fn, traj_fn=jax.vmap(trajectory_fn))

# now batch the pooling fn
pooling_fn = jax.jit(im.coarse_pool_trajectory, static_argnums=(1, 2))
pooling_fn_batched = jax.vmap(partial(pooling_fn, 
                                      pool_width=filter_size, 
                                      pool_height=filter_size))


# loss function selection
if loss_name == 'FINE':
  if n_fields == 1:
    loss_fn = jax.jit(partial(lf.mse_and_traj, 
                              trajectory_rollout_fn=real_traj_fn,
                              alpha=alpha))
  else:
    loss_fn = jax.jit(partial(lf.mse_and_traj_vel, 
                              trajectory_rollout_fn=real_traj_fn, 
                              vel_to_vort_fn=vel_to_vort_fn_batched,
                              vort_to_vel_fn=vort_to_vel_fn_batched,
                              alpha=alpha))
elif loss_name == 'COARSE':
  if n_fields == 1:
    loss_fn = jax.jit(partial(lf.mse_and_traj_coarse, 
                              trajectory_rollout_fn=real_traj_fn,
                              pooling_fn=pooling_fn_batched,
                              alpha=alpha))
  else:
    loss_fn = jax.jit(partial(lf.mse_and_traj_vel_coarse, 
                              trajectory_rollout_fn=real_traj_fn, 
                              vel_to_vort_fn=vel_to_vort_fn_batched,
                              vort_to_vel_fn=vort_to_vel_fn_batched,
                              pooling_fn=pooling_fn_batched,
                              alpha=alpha))

super_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_traj),
    loss=loss_fn, 
    metrics=[keras.losses.MeanSquaredError()]
)

# load weights if they exist
if min_val_loss < np.inf:
  super_model.load_weights(weight_loc + 'sr_best_MSE.weights.h5')
  print("Loaded pre-trained weights from an MSE fit.")
  min_val_loss = np.inf

for n in range(n_traj_steps):
  print("Traj step: ", n)
  # data augmentation at each iteration through the data (a fudge due to issues with jax/tf map compat)
  snapshots_train = np.array([sa.translate_x(sa.shift_reflect_y(field)) for field in snapshots_train])
  snapshots_train_coarse = im.coarse_pool_trajectory(snapshots_train, filter_size, filter_size)

  history = super_model.fit(snapshots_train_coarse, 
                            snapshots_train, 
                            batch_size=batch_size,
                            validation_data=(snapshots_val_coarse, snapshots_val), 
                            epochs=1)

  current_val_loss = history.history['val_loss'][-1]

  # save model weights if val loss improved
  if current_val_loss < min_val_loss:
    min_val_loss = current_val_loss
    super_model.save_weights(weight_loc + 'sr_best_traj.weights.h5')
