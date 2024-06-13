import jax.numpy as jnp
from typing import Callable

def mse_and_traj(
    vort_true: jnp.ndarray, 
    vort_pred: jnp.ndarray, 
    trajectory_rollout_fn: Callable[[jnp.ndarray], jnp.ndarray],
    vort_max: float=25.
    ):
  """ Trajectory fn expects real input and gives real output (i.e. fields in physical space)"""
  squared_errors_recon = (vort_true - vort_pred) ** 2

  # NB vort has been normalized by vort_max for prediction
  true_traj = trajectory_rollout_fn(vort_true * vort_max)
  pred_traj = trajectory_rollout_fn(vort_pred * vort_max)
  squared_errors_traj = (true_traj / vort_max - pred_traj / vort_max) ** 2
  return jnp.mean(squared_errors_recon) + jnp.mean(squared_errors_traj)