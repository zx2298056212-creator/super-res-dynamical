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
    ):
  """ Trajectory fn expects real input and gives real output (i.e. fields in physical space)"""
  squared_errors_recon = (vort_true - vort_pred) ** 2
  #squared_errors_recon = (pooling_fn(vort_true[:, jnp.newaxis, ...]) - pooling_fn(vort_pred[:, jnp.newaxis, ...])) ** 2

  # NB vort has been normalized by vort_max for prediction
  true_traj = trajectory_rollout_fn(vort_true)
  pred_traj = trajectory_rollout_fn(vort_pred)
  squared_errors_traj = ((true_traj) - 
                         (pred_traj)) ** 2
  return 0.5 * jnp.mean(squared_errors_recon) + 0.5 * jnp.mean(squared_errors_traj)
