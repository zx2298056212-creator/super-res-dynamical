import jax.numpy as jnp
from typing import Callable

def mse_and_traj(
    vort_pred: jnp.ndarray, 
    vort_true: jnp.ndarray, 
    trajectory_rollout_fn: Callable[[jnp.ndarray], jnp.ndarray],
    vort_max: float=25.
    ):
  """ Trajectory fn expects real input and gives real output (i.e. fields in physical space)"""
  squared_errors_recon = (vort_true - vort_pred) ** 2

  # NB vort has been normalized by vort_max for prediction
  true_traj = trajectory_rollout_fn(vort_true * vort_max)
  pred_traj = trajectory_rollout_fn(vort_pred * vort_max)
  squared_errors_traj = (true_traj / vort_max - pred_traj / vort_max) ** 2
  return 0.5 * jnp.mean(squared_errors_recon) + 0.5 * jnp.mean(squared_errors_traj)

def mse_and_traj_coarse(
    vort_pred: jnp.ndarray, 
    vort_true: jnp.ndarray, 
    trajectory_rollout_fn: Callable[[jnp.ndarray], jnp.ndarray],
    pooling_fn: Callable[[jnp.ndarray], jnp.ndarray],
    alpha: float = 0.5
    ):
  """ Trajectory fn expects real input and gives real output (i.e. fields in physical space)"""
  # squared_errors_recon = (vort_true - vort_pred) ** 2
  squared_errors_recon = (pooling_fn(vort_true[:, jnp.newaxis, ...]) - pooling_fn(vort_pred[:, jnp.newaxis, ...])) ** 2

  # NB vort has been normalized by vort_max for prediction
  true_traj = trajectory_rollout_fn(vort_true)
  pred_traj = trajectory_rollout_fn(vort_pred)
  squared_errors_traj = ((true_traj) - 
                         (pred_traj)) ** 2
  return alpha * jnp.mean(squared_errors_recon) + (1. - alpha) * jnp.mean(squared_errors_traj)

def mse_and_traj_vel(
    vel_pred: jnp.ndarray, 
    vel_true: jnp.ndarray, 
    trajectory_rollout_fn: Callable[[jnp.ndarray], jnp.ndarray],
    vel_to_vort_fn: Callable[[jnp.ndarray], jnp.ndarray],
    vort_to_vel_fn: Callable[[jnp.ndarray], jnp.ndarray], # these need batching
    alpha: float = 0.5
    ):
  """ Trajectory fn expects real input and gives real output (i.e. fields in physical space)"""
  squared_errors_recon = (vel_true - vel_pred) ** 2

  # (1) need to first convert vel to vort
  vort_pred = vel_to_vort_fn(vel_pred[:, jnp.newaxis, ...])
  vort_true = vel_to_vort_fn(vort_pred[:, jnp.newaxis, ...])

  # (2) Unroll
  true_traj = trajectory_rollout_fn(vort_true)
  pred_traj = trajectory_rollout_fn(vort_pred)

  # (3) back to vel -- note dimensionality mismatch, now for trajs
  vel_true_traj = vort_to_vel_fn(true_traj)
  vel_pred_traj = vort_to_vel_fn(pred_traj)

  squared_errors_traj = (vel_true_traj - vel_pred_traj) ** 2
  return alpha * jnp.mean(squared_errors_recon) + (1. - alpha) * jnp.mean(squared_errors_traj)
