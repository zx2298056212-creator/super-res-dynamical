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
import interact_model as im
import sym_augment as sa

# setup problem and create grid
Lx = 2 * jnp.pi
Ly = 2 * jnp.pi
Nx = 128
Ny = 128
Re = 40.

ae_dimension = 32 * 1
vort_max = 25. # used to normalize fields in training 
T_unroll = 2.5
t_substep = 0.25

# hyper parameters
batch_size = 32
lr = 5e-4
nval = 5000

# training configuration
n_mse_steps = 25
n_dyn_steps = 100

# data location 
data_loc = '/mnt/ceph_rbd/ae-dynamical/Re40test/'
file_front = 'vort_traj.'
files = [data_loc + file_front + str(n).zfill(4) + '.npy' for n in range(1000)]

# define uniform grid 
grid = cfd.grids.Grid((Nx, Ny), domain=((0, Lx), (0, Ly)))

# estimate stable time step based on a "max velocity" using CFL condition
max_vel_est = 5.
dt_stable = cfd.equations.stable_time_step(max_vel_est, 0.5, 1./Re, grid) / 2.

# load training data
# TODO restore functionality with keras dataset
vort_snapshots = [np.load(file_name) for file_name in files]
# add axis for channels (=1)
vort_snapshots = np.concatenate(vort_snapshots, axis=0)[..., np.newaxis] / vort_max
np.random.shuffle(vort_snapshots)
vort_train = vort_snapshots[:-nval]
vort_val = vort_snapshots[-nval:]

# build model
ae_model = models.ae_densenet_v7(Nx, Ny, ae_dimension)

# train a few epochs on standard MSE prior to unrolling
ae_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss='mse', 
    metrics=[keras.losses.MeanSquaredError()]
)

# min val loss init
min_val_loss = np.inf

for n in range(n_mse_steps):
  print("MSE step: ", n)
  vort_train = np.array([sa.translate_x(sa.shift_reflect_y(omega)) for omega in vort_train])

  history = ae_model.fit(vort_train, 
                         vort_train, 
                         validation_data=(vort_val, vort_val), 
                         batch_size=batch_size, 
                         epochs=1)
  

  current_val_loss = history.history['val_loss'][-1]

  # save model weights if val loss improved
  if current_val_loss < min_val_loss:
    min_val_loss = current_val_loss
    ae_model.save_weights('/mnt/ceph_rbd/ae-dynamical/weights/ae_best_MSE.weights.h5')

# generate a trajectory function
dt_stable = np.round(dt_stable, 3)
trajectory_fn = ts.generate_trajectory_fn(Re, T_unroll + 1e-2, 
                                          dt_stable, 
                                          grid, 
                                          t_substep=t_substep)

real_traj_fn = partial(im.real_to_real_traj_fn, traj_fn=jax.vmap(trajectory_fn))
loss_fn = jax.jit(partial(lf.mse_and_traj, trajectory_rollout_fn=real_traj_fn, vort_max=vort_max))

# now use full loss 
ae_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-4),
    loss=loss_fn, 
    metrics=[keras.losses.MeanSquaredError()]
)

# load weights if they exist
if min_val_loss < np.inf:
  ae_model.load_weights('/mnt/ceph_rbd/ae-dynamical/weights/ae_best_MSE.weights.h5')
  print("Loaded pre-trained weights from an MSE fit.")
  min_val_loss = np.inf

for n in range(n_dyn_steps):
  print("Traj step: ", n)
  # data augmentation at each iteration through the data (a fudge due to issues with jax/tf map compat)
  vort_train = np.array([sa.translate_x(sa.shift_reflect_y(omega)) for omega in vort_train])

  history = ae_model.fit(vort_train, 
                         vort_train, 
                         validation_data=(vort_val, vort_val), 
                         epochs=1)

  current_val_loss = history.history['val_loss'][-1]

  # save model weights if val loss improved
  if current_val_loss < min_val_loss:
    min_val_loss = current_val_loss
    ae_model.save_weights('/mnt/ceph_rbd/ae-dynamical/weights/sr_best_traj.weights.h5')

