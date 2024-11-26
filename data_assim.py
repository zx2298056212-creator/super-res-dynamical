import os
os.environ["KERAS_BACKEND"] = "jax"

import jax
import jax.numpy as jnp
import numpy as np

import jax_cfd.base as cfd
from functools import partial

import time_stepping as ts
import interact_model as im
import da_optimisation as dao

from scipy.ndimage import zoom

# wrapper to deal with the fact that traj fn in "interact_model" is over batches
def vort_traj_fn_wrapper(vort_phys, trajectory_fn):
  """ Add batch and channel dimensions, call routine. Note underlying
      trajectory function also gets batched via vmap """
  traj_phys = im.real_to_real_traj_fn(
    vort_phys[jnp.newaxis, ..., jnp.newaxis],
    jax.vmap(trajectory_fn)
    )[0,...,0]
  return traj_phys

class Assimilator:
  def __init__(
      self,
      Re: float,
      Nx: int,
      Ny: int,
      Lx: float,
      Ly: float,
      T_unroll: float,
      filter_size: int,
      opt_class, #: optax.Optimizer,
      learning_rate: float,
      vel_assim: bool=True,
      damping: float=0.1 # linear damping
  ):
    # problem and grid
    self.Re = Re
    self.Nx = Nx
    self.Ny = Ny
    self.Lx = Lx
    self.Ly = Ly
    self.grid = cfd.grids.Grid((self.Nx, self.Ny), 
                               domain=((0, self.Lx), (0, self.Ly)))
    
    # assim 
    self.filter_size = filter_size
    self.M_substep = filter_size
    self.T_unroll = T_unroll

    max_vel_est = 5.
    dt_stable = cfd.equations.stable_time_step(max_vel_est, 
                                               0.5, 
                                               1. / self.Re, 
                                               self.grid) / 2.
    
    self.dt_stable = np.round(dt_stable, 3)
    self.t_substep = self.M_substep * self.dt_stable
    trajectory_fn = ts.generate_trajectory_fn(self.Re, 
                                              self.T_unroll + 1e-2, 
                                              self.dt_stable, 
                                              self.grid, 
                                              t_substep=self.t_substep,
                                              damping=damping)
    
    self.real_traj_fn = jax.jit(
      partial(vort_traj_fn_wrapper, 
              trajectory_fn=trajectory_fn)
    )
    self.pooling_fn = jax.jit(partial(im.coarse_pool_trajectory, 
                                      pool_width=filter_size, 
                                      pool_height=filter_size))
    
    # optimizer
    self.lr = learning_rate
    self.optimizer = opt_class(self.lr)

    self.vel_assim = vel_assim
    if self.vel_assim == True:
      self.vel_to_vort_fn = partial(im.compute_vort_traj, 
                                    dx = self.Lx / self.Nx, 
                                    dy = self.Ly / self.Ny)
      self.vort_to_vel_fn = partial(im.compute_vel_traj, 
                                    dx = self.Lx / self.Nx, 
                                    dy = self.Ly / self.Ny)

  def _setup_vel_assim(self, vort_init: jnp.ndarray):
    self.vel_init = self.vort_to_vel_fn(vort_init[jnp.newaxis, ..., jnp.newaxis])[0, ...]
    vort_true_traj = self.real_traj_fn(vort_init)
    vel_true_traj = self.vort_to_vel_fn(vort_true_traj[..., jnp.newaxis])
    vel_true_coarse_traj = self.pooling_fn(vel_true_traj)

    # construct loss
    loss_fn = partial(dao.vel_loss,
                      vel_traj_coarse_true=vel_true_coarse_traj,
                      pooling_fn=self.pooling_fn,
                      trajectory_rollout_fn=self.real_traj_fn,
                      vel_to_vort_fn=self.vel_to_vort_fn,
                      vort_to_vel_fn=self.vort_to_vel_fn
                      )
    loss_fn_jitted = jax.jit(loss_fn)
    self.val_and_grad_fn = jax.value_and_grad(loss_fn_jitted)

  def _setup_vort_assim(self, vort_init: jnp.ndarray):
    vort_true_traj = self.real_traj_fn(vort_init)
    # note add axis ("channel") then remove (coarse pooling designed for image convnet problems)
    vort_true_coarse_traj = self.pooling_fn(vort_true_traj[..., jnp.newaxis])[..., 0]

    # construct loss 
    loss_fn = partial(dao.vort_loss, 
                      vort_traj_coarse_true=vort_true_coarse_traj,
                      trajectory_rollout_fn=self.real_traj_fn,
                      pooling_fn=self.pooling_fn)
    loss_fn_jitted = jax.jit(loss_fn)
    self.val_and_grad_fn = jax.value_and_grad(loss_fn_jitted)

  def assimilate(
      self,
      vort_snapshot: jnp.ndarray,
      n_opt_step: int
  ):
    if self.vel_assim == True:
      self._setup_vel_assim(vort_snapshot)
      vel_init_coarse = self.pooling_fn(self.vel_init[jnp.newaxis, ...])[0, ...]
      vel_pred_init = zoom(vel_init_coarse, 
                           (self.filter_size, self.filter_size, 1), 
                           order=3, 
                           mode='grid-wrap')
      # run optimiser
      state_current = vel_pred_init
      opt_state = self.optimizer.init(state_current)
      for n in range(n_opt_step):
        state_current, opt_state, loss = dao.update_guess_vel(state_current, 
                                                              opt_state, 
                                                              self.optimizer,
                                                              self.val_and_grad_fn)
        print("Step: ", n+1, "Loss: ", loss)

    else:
      self._setup_vort_assim(vort_snapshot)
      vort_init_coarse = self.pooling_fn(vort_snapshot[jnp.newaxis, ..., jnp.newaxis])[0, ..., 0]
      vort_pred_init = zoom(vort_init_coarse, filter_size, order=3)
      # run optimiser
      state_current = vort_pred_init
      opt_state = self.optimizer.init(state_current)
      for n in range(n_opt_step):
        state_current, opt_state, loss = dao.update_guess_vort(state_current, 
                                                               opt_state, 
                                                               self.optimizer,
                                                               self.val_and_grad_fn)
        print("Step: ", n+1, "Loss: ", loss)
    return state_current