""" Train models on noisy trajectories (coarse approach and velocity ONLY)
    different dataset required since training loads full trajectories
    rather than snapshots. Single run with fixed dt etc. """
import os
os.environ["KERAS_BACKEND"] = "jax"

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

data_loc = '/mnt/ceph_rbd/jax-cfd-data-gen/Re1000/'
weight_loc = '/mnt/ceph_rbd/ae-dynamical/weights/'
file_front = 'vort_revi_traj_Re1000L2pi_'
file_end = '_0.npy'
n_files = 65
n_fields = 2 # 1 for vorticity, 2 for velocity

Nx = 512
Ny = 512
Re = 1000.0

# network hyp
filter_size = 32
n_grow = 5
T_unroll = 1.
M_substep = 32

# training hyp
batch_size = 16
lr_traj = 1e-4
nval = 10
n_traj_steps = 50

# setup problem and create grid
Lx = 2 * jnp.pi
Ly = 2 * jnp.pi

grid = cfd.grids.Grid((Nx, Ny), domain=((0, Lx), (0, Ly)))

# estimate stable time step based on a "max velocity" using CFL condition
max_vel_est = 5.
dt_stable = cfd.equations.stable_time_step(max_vel_est, 0.5, 1. / Re, grid) / 2.

# create downsampled data -- needs cleaning
files = [data_loc + file_front + str(n).zfill(2) + file_end for n in range(n_files)]

# verify shape
vort_trajs = np.array([np.load(file_name) for file_name in files])[..., np.newaxis] # (traj, Nt, Nx, Ny, 1)
vel_trajs = np.array([
  im.compute_vel_traj(
    np.load(file_name)[..., np.newaxis],
    Lx / Nx,
    Ly / Ny)  
    for file_name in files
    ])

traj_train = vel_trajs[:-nval]
traj_val = vel_trajs[-nval:]

# now batch the pooling fn
pooling_fn = jax.jit(im.coarse_pool_trajectory, static_argnums=(1, 2))
pooling_fn_batched = jax.vmap(partial(pooling_fn, 
                                      pool_width=filter_size, 
                                      pool_height=filter_size))
traj_train_coarse = pooling_fn_batched(traj_train)
traj_val_coarse = pooling_fn_batched(traj_val)

# batch the velocity/vorticity functions
vel_to_vort_fn = jax.jit(
  partial(im.compute_vort_traj, dx=Lx / Nx, dy=Ly / Ny)
  )
vort_to_vel_fn = jax.jit(
  partial(im.compute_vel_traj, dx=Lx / Nx, dy=Ly / Ny)
  )
vel_to_vort_fn_batched = jax.vmap(vel_to_vort_fn)
vort_to_vel_fn_batched = jax.vmap(vort_to_vel_fn)

# TODO JP batch the symmetry operations 


# min val loss init
min_val_loss = np.inf

# generate a trajectory function (for vorticity)
# verify that this matches the training data!
dt_stable = np.round(dt_stable, 3)
t_substep = dt_stable * M_substep
N_steps = int(T_unroll / t_substep)
print("dt_stable:", dt_stable, "T substep:", t_substep)
trajectory_fn = ts.generate_trajectory_fn(Re, T_unroll + 1e-2, dt_stable, grid, t_substep=t_substep)
real_traj_fn = partial(im.real_to_real_traj_fn, traj_fn=jax.vmap(trajectory_fn))

# build model which takes trajectory inputs
super_model = models.super_res_vel_v3_traj(Nx // filter_size,
                                           Ny // filter_size, 
                                           N_steps,
                                           32, 
                                           N_grow=n_grow, 
                                           input_channels=n_fields)



loss_fn = jax.jit(partial(lf.traj_vel_coarse_noise, 
                          trajectory_rollout_fn=real_traj_fn, 
                          vel_to_vort_fn=vel_to_vort_fn_batched,
                          vort_to_vel_fn=vort_to_vel_fn_batched,
                          pooling_fn=pooling_fn_batched))

super_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_traj),
    loss=loss_fn, 
    metrics=[keras.losses.MeanSquaredError()]
)

for n in range(n_traj_steps):
  print("Traj step: ", n)
  # TODO -- batch data augmentation of trajectories
  # # data augmentation at each iteration through the data (a fudge due to issues with jax/tf map compat)
  # snapshots_train = np.array([sa.translate_x(sa.shift_reflect_y(field)) for field in snapshots_train])
  # snapshots_train_coarse = im.coarse_pool_trajectory(snapshots_train, filter_size, filter_size)

  history = super_model.fit(traj_train_coarse, 
                            traj_train, 
                            batch_size=batch_size,
                            validation_data=(traj_val_coarse, traj_val), 
                            epochs=1)

  current_val_loss = history.history['val_loss'][-1]

  # save model weights if val loss improved
  if current_val_loss < min_val_loss:
    min_val_loss = current_val_loss
    super_model.save_weights(weight_loc + 'sr_best_traj_noise.weights.h5')
