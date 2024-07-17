""" To do: convert to jax-based data augmentation """
import tensorflow as tf
import numpy as np
import jax.numpy as jnp

from typing import Union
Array = Union[np.ndarray, jnp.ndarray]

# def translate_x(x: Array) -> Array:
#   """
#   Translation (x) augmentation.

#   Args:
#     x: greyscale vorticity image

#   Returns:
#     An augmented image.
#   """
#   x_tf = tf.convert_to_tensor(x)
#   nx, _ = x_tf.shape
#   shift = tf.random.uniform(shape=[], minval=0, maxval=nx-1, dtype=tf.int32)

#   x_tf = tf.roll(x_tf, shift, 1)
#   return x_tf.numpy()

def translate_x(omega: Array) -> Array:
  Nx, _ = omega.shape[:2]
  shift = np.random.randint(0, Nx)
  omega_translated = np.roll(omega, shift, axis=0)
  return omega_translated

def shift_reflect_y(omega: Array) -> Array:
  _, Ny = omega.shape[:2]
  shift = np.random.randint(0, 8)

  for j in range(shift):
    omega = -np.flip(omega, axis=1)
  
  shift *= Ny // 8
  omega_shift_ref = np.roll(omega, shift, axis=1)
  return omega_shift_ref

# def shift_reflect_y(x: Array) -> Array:
#   """ 
#   Shift-reflect (y) augmentation.

#   Args:
#     x: greyscale vorticity image

#   Returns:
#     augmented image
#   """
#   x_tf = tf.convert_to_tensor(x)
#   _, Ny = x_tf.shape
#   shift = tf.random.uniform(shape=[], minval=0, maxval=7, dtype=tf.int32)
  
#   for j in range(shift):
#     x_tf = -tf.reverse(x_tf, tf.convert_to_tensor([1, 0], dtype=tf.int32)) 

#   shift *= Ny // 8
#   x_tf = tf.roll(x_tf, shift, 0)
#   return x_tf.numpy()
