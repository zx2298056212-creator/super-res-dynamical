import jax.numpy as jnp

# wrap trajectory function with FFTs to enable physical space -> physical space map
def real_to_real_traj_fn(vort_phys, traj_fn):
  # FT of space and select first (only) channel
  vort_rft = jnp.fft.rfftn(vort_phys, axes=(1,2))[...,0]
  _, traj_rft = traj_fn(vort_rft)
  # axes for FT move back because we now also have time dimension; add channel
  traj_phys = jnp.fft.irfftn(traj_rft, axes=(2,3))[...,jnp.newaxis]
  return traj_phys

def average_pool_trajectory(omega_traj, pool_width, pool_height):
  trajectory_length, Nx, Ny, Nchannels = omega_traj.shape
  assert Nx % pool_width == 0
  assert Ny % pool_height == 0

  omega_reshaped = omega_traj.reshape(
    (trajectory_length, Nx // pool_width, pool_width, Ny // pool_height, pool_height, Nchannels)
  )
  omega_pooled_traj = omega_reshaped.mean(axis=(2, 4))
  return omega_pooled_traj

def coarse_pool_trajectory(omega_traj, pool_width, pool_height):
  _, Nx, Ny, _ = omega_traj.shape
  assert Nx % pool_width == 0
  assert Ny % pool_height == 0
  coarse_x = pool_width
  coarse_y = pool_height

  omega_pooled_traj = omega_traj[:, ::coarse_x, ::coarse_y, :]
  return omega_pooled_traj