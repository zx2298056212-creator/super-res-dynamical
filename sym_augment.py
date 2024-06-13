""" To do: convert to jax-based data augmentation """
import tensorflow as tf

def translate_x(x: tf.Tensor) -> tf.Tensor:
  """
  Translation (x) augmentation.

  Args:
    x: greyscale vorticity image

  Returns:
    An augmented image.
  """
  nx, _ = x.shape
  shift = tf.random.uniform(shape=[], minval=0, maxval=nx-1, dtype=tf.int32)

  x = tf.roll(x, shift, 1)
  return x

def shift_reflect_y(x): #: tf.Tensor) -> tf.Tensor:
  """ 
  Shift-reflect (y) augmentation.

  Args:
    x: greyscale vorticity image

  Returns:
    augmented image
  """
  Nx, Ny = x.shape
  shift = tf.random.uniform(shape=[], minval=0, maxval=7, dtype=tf.int32)
  
  for j in range(shift):
    x = -tf.reverse(x, tf.convert_to_tensor([1, 0], dtype=tf.int32)) 

  shift *= Ny // 8
  x = tf.roll(x, shift, 0)
  return x

def augment(image_input):
  image = tf.reshape(image_input, (128, 128))
  image = shift_reflect_y(image)
  image = translate_x(image)
  #image = rotate_pi(image)
  image = tf.reshape(image, [128, 128, 1])
  image_input = image
  return image_input
