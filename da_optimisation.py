# TODO fix pooling / vel-to-vort etc to take consistent shaped arrays
# i.e. make all batches OR not. At the moment it's mix and match
import optax 
import jax.numpy as jnp 
from models import leray_projection

from typing import Callable

def vort_loss(
    vort_pred: jnp.ndarray,
    vort_traj_coarse_true: jnp.ndarray,
    trajectory_rollout_fn: Callable[[jnp.ndarray], jnp.ndarray],
    pooling_fn: Callable[[jnp.ndarray], jnp.ndarray]
) -> float:
  # unroll prediction and coarse
  vort_traj_pred = trajectory_rollout_fn(vort_pred)
  vort_traj_coarse_pred = pooling_fn(vort_traj_pred[...,jnp.newaxis])[...,0]
  error_square = (vort_traj_coarse_pred - vort_traj_coarse_true) ** 2
  return jnp.mean(error_square)

def vel_loss(
    vel_pred: jnp.ndarray,
    vel_traj_coarse_true: jnp.ndarray,
    trajectory_rollout_fn: Callable[[jnp.ndarray], jnp.ndarray],
    pooling_fn: Callable[[jnp.ndarray], jnp.ndarray],
    vel_to_vort_fn: Callable[[jnp.ndarray], jnp.ndarray],
    vort_to_vel_fn: Callable[[jnp.ndarray], jnp.ndarray]
) -> float:
  """ Note vel_to_vort_fn and vort_to_vel_fn are assumed to take batches of trajectories (
      typically used in ML pipeline) hence we add dimesions to arrays and remove accordingly """
  # (1) convert velocity prediction to vorticity 
  # add dimensions for batch size and time 
  vort_pred = vel_to_vort_fn(vel_pred[jnp.newaxis, ...])[0, ..., 0]

  # (2) Unroll
  vort_traj_pred = trajectory_rollout_fn(vort_pred)

  # (3) back to vel -- note dimensionality mismatch, now for trajs
  vel_traj_pred = vort_to_vel_fn(vort_traj_pred[..., jnp.newaxis])

  # (3) find velocity for trajectory and coarse-grain
  vel_traj_coarse_pred = pooling_fn(vel_traj_pred)

  error_square = (vel_traj_coarse_pred - vel_traj_coarse_true) ** 2
  return jnp.mean(error_square)

def update_guess_vort(state_current,
                      opt_state, 
                      optimizer, 
                      val_and_grad_fn):
  loss, grads = val_and_grad_fn(state_current)
  update_vec, opt_state = optimizer.update(grads, opt_state, state_current)
  state_current = optax.apply_updates(state_current, update_vec)
  return state_current, opt_state, loss

def update_guess_vel(state_current,
                     opt_state,
                     optimizer,
                     val_and_grad_fn):
  """ Project updates onto div-free space before applying """
  loss, grads = val_and_grad_fn(state_current)
  # note leray is over batches, so add and then remove a dimension
  grads_div_free = leray_projection(grads[jnp.newaxis, ...])[0, ...]
  update_vec, opt_state = optimizer.update(grads_div_free, opt_state, state_current)
  state_current = optax.apply_updates(state_current, update_vec)
  return state_current, opt_state, loss