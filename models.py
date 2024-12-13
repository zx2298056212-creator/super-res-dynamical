""" Attempt to convert old tf keras model to jax backend """
import keras
import keras.backend as K
import keras.ops as kops
from keras.layers import Conv2D, Lambda
from keras.layers import Input, MaxPooling2D, UpSampling2D
from keras.layers import BatchNormalization, Concatenate, Activation

from keras.models import Model
import jax.numpy as jnp

def pad_periodic(x, n_pad_rows=0, n_pad_cols=0):
  """
  Pads the rows and columns of a 4D tensor in a periodic manner.

    Args:
      x (np.ndarray): Input tensor with shape (batch_size, height, width, channels).
      n_pad_x (int): Number of rows to pad.
      n_pad_y (int): Number of columns to pad.

    Returns:
      np.ndarray: Padded tensor with shape (batch_size, height + n_pad_x, width + n_pad_y, channels).
  """
  top_rows = x[:, -n_pad_rows // 2:, :, :]
  bottom_rows = x[:, :n_pad_rows // 2, :, :]
  padded_rows_x = kops.concatenate([top_rows, x, bottom_rows], axis=1)

  left_cols = padded_rows_x[:, :, -n_pad_cols // 2:, :]
  right_cols = padded_rows_x[:, :, :n_pad_cols // 2, :]
  padded_rows_cols_x = kops.concatenate([left_cols, padded_rows_x, right_cols], axis=2)
  return padded_rows_cols_x


def periodic_convolution(x, n_filters, kernel, activation='relu', 
                         strides=(1,1), n_pad_rows=0, n_pad_cols=0):
  """ 
  Applies periodic boundary conditions before convolving an input tensor with a set of filters.

    Args:
        x (np.ndarray): Input tensor with shape (batch_size, height, width, channels).
        n_filters (int): Number of filters to use in the convolution.
        kernel (tuple(int)): Tuple specifying the dimensions of the convolution kernel. Should have length 2.
        activation (str or callable): Activation function to use after the convolution. Defaults to 'relu'.
        strides (tuple(int)): Tuple specifying the strides to use in the convolution. Should have length 2.
        n_pad_rows/cols (int): Number of rows/cols to pad before convolving the tensor.

    Returns:
        np.ndarray: Tensor resulting from the convolution, with shape (batch_size, new_height, new_width, n_filters).
  """
  # shape computation with padding; does not include batch dimension 
  padded_shape = (x.shape[1] + n_pad_rows, x.shape[2] + n_pad_cols, x.shape[-1])

  x_padded = Lambda(pad_periodic, arguments={'n_pad_rows': n_pad_rows, 'n_pad_cols': n_pad_cols},
                    output_shape=padded_shape)(x)
  return Conv2D(n_filters, kernel, activation=activation, padding='valid', strides=strides)(x_padded)


def residual_block_periodic_conv(x, n_filters,
                                 kernel=(1,1), strides=(1,1),
                                 n_pad_rows=1, n_pad_cols=1,
                                 activation='gelu'):
  layer_input = x 

  # number of filters must match input = x.shape[-1]
  n_filters = x.shape[-1]

  x = BatchNormalization()(x)
  x = Activation(activation)(x)

  x = periodic_convolution(x, n_filters, kernel, strides=strides,
                           n_pad_rows=n_pad_rows, n_pad_cols=n_pad_cols, activation=activation)
  
  x = BatchNormalization()(x)

  x = periodic_convolution(x, n_filters, kernel, strides=strides,
                           n_pad_rows=n_pad_rows, n_pad_cols=n_pad_cols, activation='linear')

  x = keras.layers.add([x, layer_input])
  return x

def super_res_v0(Nx_coarse, Ny_coarse, N_filters, N_grow=4, input_channels=1):
  """ Build a model to perform super-resolution, scaling up N_grow times.  """
  input_vort = Input(shape=(Nx_coarse, Ny_coarse, input_channels))
   
  # an initial linear layer prior to Residual blocks 
  x = periodic_convolution(input_vort, N_filters, kernel=(4, 4),
                           n_pad_rows=3, n_pad_cols=3, activation='linear')
  
  # upsample and apply residual block however many times we need to rescale 
  # note we are keeping our kernel constant size -- perhaps not ideal
  # might want to scale this too 
  for _ in range(N_grow):
    x = UpSampling2D((2,2))(x)
    x = residual_block_periodic_conv(x, N_filters, kernel=(4,4),
                                     n_pad_rows=3, n_pad_cols=3)
  
  x = periodic_convolution(x, input_channels, kernel=(4, 4),
                           n_pad_rows=3, n_pad_cols=3, activation='linear')
  return Model(input_vort, x)

def leray_projection(fields, eps=1e-8):
  batch_size, Nx, Ny, _ = fields.shape
  dx = 2 * jnp.pi / Nx 
  dy = 2 * jnp.pi / Ny

  fields_rft = jnp.fft.rfftn(fields, axes=(1,2))
  u_rft = fields_rft[..., 0]
  v_rft = fields_rft[..., 1]

  all_kx = 2 * jnp.pi * jnp.fft.fftfreq(Nx, dx)
  all_ky = 2 * jnp.pi * jnp.fft.rfftfreq(Ny, dy)
  
  kx_mesh, ky_mesh = jnp.meshgrid(all_kx, all_ky)
  kx_mesh = jnp.repeat(
    (kx_mesh.T)[jnp.newaxis, ..., jnp.newaxis],
    repeats=batch_size, 
    axis=0)
  ky_mesh = jnp.repeat(
    (ky_mesh.T)[jnp.newaxis, ..., jnp.newaxis],
    repeats=batch_size,
    axis=0)

  # (1) compute divergence 
  ikxu = 1j * kx_mesh * u_rft[..., jnp.newaxis]
  ikyv = 1j * ky_mesh * v_rft[..., jnp.newaxis]
  div_u_rft = ikxu + ikyv

  # (2) solve Poisson problem 
  phi_rft = - div_u_rft / (eps + kx_mesh ** 2 + ky_mesh ** 2)

  # (3) take grad into channels
  u_correct_ft = -jnp.concatenate([1j * kx_mesh * phi_rft,
                                   1j * ky_mesh * phi_rft], axis=-1)
  u_correction = jnp.fft.irfftn(u_correct_ft, axes=(1,2))
  return fields + u_correction

# objective is make div-func pre-compiled to avoid rebuild every batch
def div_free_2D_layer(u_in):
  """" Custom layer to project onto divergence-free solution via Leray:
          u_out = u_in - grad( nab^{-1} div(u_in) )
       Expected shape is (None, Nx, Ny, 2) """
  if len(u_in.shape) != 4:
    raise ValueError("Expected 4D input, input has shape", u_in.shape)
  if u_in.shape[-1] != 2:
    raise ValueError("Expected 2 channels (2D vel field) but input has ",
                     u_in.shape[-1],
                     "channels.")
  # output shape does not include batch dim -- check
  u_projected = Lambda(leray_projection, 
                       output_shape=u_in.shape[1:])(u_in)
  return u_projected

def exponential_filter(fields):
  batch_size, Nx, Ny, _ = fields.shape
  dx = 2 * jnp.pi / Nx 
  dy = 2 * jnp.pi / Ny

  fields_rft = jnp.fft.rfftn(fields, axes=(1,2))

  all_kx = 2 * jnp.pi * jnp.fft.fftfreq(Nx, dx)
  all_ky = 2 * jnp.pi * jnp.fft.rfftfreq(Ny, dy)
  
  kx_mesh, ky_mesh = jnp.meshgrid(all_kx, all_ky)
  kx_mesh = jnp.repeat(
    (kx_mesh.T)[jnp.newaxis, ..., jnp.newaxis],
    repeats=batch_size, 
    axis=0)
  ky_mesh = jnp.repeat(
    (ky_mesh.T)[jnp.newaxis, ..., jnp.newaxis],
    repeats=batch_size,
    axis=0)
  
  # following Dresdner et al filter exp(- alpha | k / k_max | ^ 2p | ); p = 32, alpha = 6
  k_all = jnp.sqrt(kx_mesh ** 2 + ky_mesh ** 2)
  k_max = jnp.max(k_all)
  filter_exp = jnp.exp( -6 * (k_all / k_max) ** 64)

  # filter field and invert
  filtered_field = fields_rft * filter_exp
  
  return jnp.fft.irfftn(filtered_field, axes=(1,2))

def circular_filter(fields):
  """ Based on JAX-CFD spectral code base; apply 2/3 de-aliasing to output field
      [smooth version; TODO read refs] """
  batch_size, Nx, Ny, _ = fields.shape
  dx = 2 * jnp.pi / Nx 
  dy = 2 * jnp.pi / Ny

  fields_rft = jnp.fft.rfftn(fields, axes=(1,2))

  all_kx = 2 * jnp.pi * jnp.fft.fftfreq(Nx, dx)
  all_ky = 2 * jnp.pi * jnp.fft.rfftfreq(Ny, dy)
  
  kx_mesh, ky_mesh = jnp.meshgrid(all_kx, all_ky)
  kx_mesh = jnp.repeat(
    (kx_mesh.T)[jnp.newaxis, ..., jnp.newaxis],
    repeats=batch_size, 
    axis=0)
  ky_mesh = jnp.repeat(
    (ky_mesh.T)[jnp.newaxis, ..., jnp.newaxis],
    repeats=batch_size,
    axis=0)
  
  k_all = jnp.sqrt(kx_mesh ** 2 + ky_mesh ** 2)
  k_max = jnp.max(k_all)

  # following based on JAX-CFD
  cphi = 0.65 * k_max
  filterfac = 23.6
  filter_ = jnp.exp(-filterfac * (k_all - cphi) ** 4.)
  filter_ = jnp.where(k_all <= cphi, jnp.ones_like(filter_), filter_)
  
  filtered_field = fields_rft * filter_
  return jnp.fft.irfftn(filtered_field, axes=(1,2))

# exp filter layer
def exp_filter_layer(u_in):
  """" Custom layer to apply exponential filter """
  u_projected = Lambda(exponential_filter, 
                       output_shape=u_in.shape[1:])(u_in)
  return u_projected

# circ filter layer
def circ_filter_layer(u_in):
  """ Custom layer for de-aliasing filter """
  u_projected = Lambda(circular_filter,
                       output_shape=u_in.shape[1:])(u_in)
  return u_projected

def super_res_vel_v1(Nx_coarse, Ny_coarse, N_filters, N_grow=4, input_channels=2):
  """ Build a model to perform super-resolution on VELOCITY DATA, 
      scaling up N_grow times.  """
  input_vort = Input(shape=(Nx_coarse, Ny_coarse, input_channels))
   
  # an initial linear layer prior to Residual blocks 
  x = periodic_convolution(input_vort, N_filters, kernel=(4, 4),
                           n_pad_rows=3, n_pad_cols=3, activation='linear')
  
  # upsample and apply residual block however many times we need to rescale 
  # note we are keeping our kernel constant size -- perhaps not ideal
  # might want to scale this too 
  for _ in range(N_grow):
    x = UpSampling2D((2,2))(x)
    x = residual_block_periodic_conv(x, N_filters, kernel=(4,4),
                                     n_pad_rows=3, n_pad_cols=3)
  
  x = periodic_convolution(x, input_channels, kernel=(4, 4),
                           n_pad_rows=3, n_pad_cols=3, activation='linear')
  # project out non-solenoidal component
  x = div_free_2D_layer(x)
  x = exp_filter_layer(x)
  return Model(input_vort, x)

def super_res_vel_v2(Nx_coarse, Ny_coarse, N_filters, N_grow=4, input_channels=2):
  """ Build a model to perform super-resolution on VELOCITY DATA, 
      scaling up N_grow times.  
      Change vs v1: increasing kernel size as we go up
  """
  input_vort = Input(shape=(Nx_coarse, Ny_coarse, input_channels))
   
  # an initial linear layer prior to Residual blocks 
  x = periodic_convolution(input_vort, N_filters, kernel=(4, 4),
                           n_pad_rows=3, n_pad_cols=3, activation='linear')
  
  # upsample and apply residual block however many times we need to rescale 
  # note we are keeping our kernel constant size -- perhaps not ideal
  # might want to scale this too 
  for n in range(N_grow):
    kern_width = 2 ** (2 + n)
    x = UpSampling2D((2,2))(x)
    x = residual_block_periodic_conv(x, 
                                     N_filters, 
                                     kernel=(kern_width,kern_width),
                                     n_pad_rows=kern_width-1, 
                                     n_pad_cols=kern_width-1)
  
  x = periodic_convolution(x, input_channels, kernel=(16, 16),
                           n_pad_rows=15, n_pad_cols=15, activation='linear')
  # project out non-solenoidal component
  if input_channels == 2:
    x = div_free_2D_layer(x)
  x = exp_filter_layer(x)
  return Model(input_vort, x)

def super_res_vel_v3(Nx_coarse, Ny_coarse, N_filters, N_grow=4, input_channels=2):
  """ As v1 but with circular filter (de-alias).  """
  input_vort = Input(shape=(Nx_coarse, Ny_coarse, input_channels))
   
  # an initial linear layer prior to Residual blocks 
  x = periodic_convolution(input_vort, N_filters, kernel=(4, 4),
                           n_pad_rows=3, n_pad_cols=3, activation='linear')
  
  # upsample and apply residual block however many times we need to rescale 
  # note we are keeping our kernel constant size -- perhaps not ideal
  # might want to scale this too 
  for _ in range(N_grow):
    x = UpSampling2D((2,2))(x)
    x = residual_block_periodic_conv(x, N_filters, kernel=(4,4),
                                     n_pad_rows=3, n_pad_cols=3)
  
  x = periodic_convolution(x, input_channels, kernel=(4, 4),
                           n_pad_rows=3, n_pad_cols=3, activation='linear')
  # project out non-solenoidal component
  x = div_free_2D_layer(x)
  x = circ_filter_layer(x)
  return Model(input_vort, x)

def super_res_vel_v3_noleray(Nx_coarse, Ny_coarse, N_filters, N_grow=4, input_channels=2):
  """ As v1 but with circular filter (de-alias).  """
  input_vort = Input(shape=(Nx_coarse, Ny_coarse, input_channels))
   
  # an initial linear layer prior to Residual blocks 
  x = periodic_convolution(input_vort, N_filters, kernel=(4, 4),
                           n_pad_rows=3, n_pad_cols=3, activation='linear')
  
  # upsample and apply residual block however many times we need to rescale 
  # note we are keeping our kernel constant size -- perhaps not ideal
  # might want to scale this too 
  for _ in range(N_grow):
    x = UpSampling2D((2,2))(x)
    x = residual_block_periodic_conv(x, N_filters, kernel=(4,4),
                                     n_pad_rows=3, n_pad_cols=3)
  
  x = periodic_convolution(x, input_channels, kernel=(4, 4),
                           n_pad_rows=3, n_pad_cols=3, activation='linear')
  # project out non-solenoidal component
  x = circ_filter_layer(x)
  return Model(input_vort, x)

def super_res_vel_v3_traj(Nx_coarse, Ny_coarse, Nt, N_filters, N_grow=4, input_channels=2):
  """ Modify v3 to deal with trajectory input -- but take only first entry """
  # add "None" for trajectory input 
  input_vort = Input(shape=(Nt, Nx_coarse, Ny_coarse, input_channels))  

  # Slice the input to use only the first time step.
  # Lambda layer to extract the first time step. We assume the first time step is along axis 1.
  first_step = Lambda(lambda x: x[:, 0, :, :, :])(input_vort)

  # an initial linear layer prior to Residual blocks 
  x = periodic_convolution(first_step, N_filters, kernel=(4, 4),
                           n_pad_rows=3, n_pad_cols=3, activation='linear')
  
  # upsample and apply residual block however many times we need to rescale 
  # note we are keeping our kernel constant size -- perhaps not ideal
  # might want to scale this too 
  for _ in range(N_grow):
    x = UpSampling2D((2,2))(x)
    x = residual_block_periodic_conv(x, N_filters, kernel=(4,4),
                                     n_pad_rows=3, n_pad_cols=3)
  
  x = periodic_convolution(x, input_channels, kernel=(4, 4),
                           n_pad_rows=3, n_pad_cols=3, activation='linear')
  # project out non-solenoidal component
  x = div_free_2D_layer(x)
  x = circ_filter_layer(x)

  # now artificially copy the output so that the shape matches the input (a trajectory)
  def repeat_time(x, time_steps):
    return jnp.repeat(x[:, jnp.newaxis, ...], time_steps, axis=1)

  repeat_shape = (Nt, int(Nx_coarse * 2 ** N_grow), int(Ny_coarse * 2 ** N_grow), input_channels)
  output_traj = Lambda(repeat_time, arguments={'time_steps': Nt}, output_shape=repeat_shape)(x)

  return Model(input_vort, output_traj)

def super_res_taira_leray(Nx_coarse, Ny_coarse):
  """ Create DSC/MS model from Fukami, Fukagata & Taira,
      for the sake of illutration we insist: 
     
      coarse_graining = 32 so
      Nx_coarse = Nx // 32 
      Ny_coarse = Ny // 32

      we will need to generalise -- in fact easy to do
  """ 
  # we first build the "skip connection" component
  # we follow the Fukami achitecture "in spirit" because we are doing more severe upsampling
  input_vel = Input(shape=(Nx_coarse, Ny_coarse, 2))

  # looking at the original JFM 2019 (Fukami, Fukagata, Taira) 
  # seems that the input image is first upsampled to the target size for all subsequent processing... 
  input_vel_upsampled = UpSampling2D((32, 32), interpolation='bicubic')(input_vel)

  down_1 = MaxPooling2D((8,8),padding='same')(input_vel_upsampled)
  x1 = periodic_convolution(down_1, 32, kernel=(3,3),
                            n_pad_rows=2, n_pad_cols=2, activation='gelu')
  x1 = periodic_convolution(x1, 32, kernel=(3,3),
                            n_pad_rows=2, n_pad_cols=2, activation='gelu')
  x1 = UpSampling2D((2,2))(x1)

  down_2 = MaxPooling2D((4,4),padding='same')(input_vel_upsampled)
  x2 = Concatenate(axis=-1)([x1, down_2]) 
  x2 = periodic_convolution(x2, 32, kernel=(3,3),
                            n_pad_rows=2, n_pad_cols=2, activation='gelu')
  x2 = periodic_convolution(x2, 32, kernel=(3,3),
                            n_pad_rows=2, n_pad_cols=2, activation='gelu')
  x2 = UpSampling2D((2,2))(x2)

  down_3 = MaxPooling2D((2,2),padding='same')(input_vel_upsampled)
  x3 = Concatenate(axis=-1)([x2, down_3]) 
  x3 = periodic_convolution(x3, 32, kernel=(3,3),
                            n_pad_rows=2, n_pad_cols=2, activation='gelu')
  x3 = periodic_convolution(x3, 32, kernel=(3,3),
                            n_pad_rows=2, n_pad_cols=2, activation='gelu')
  x3 = UpSampling2D((2,2))(x3)

  x4 = Concatenate(axis=-1)([x3, input_vel_upsampled])
  x4 = periodic_convolution(x4, 32, kernel=(3,3),
                            n_pad_rows=2, n_pad_cols=2, activation='gelu')
  x4 = periodic_convolution(x4, 32, kernel=(3,3),
                            n_pad_rows=2, n_pad_cols=2, activation='gelu')

  #Multi-scale model (Du et al., 2018)
  layer_1 = periodic_convolution(input_vel_upsampled, 16, kernel=(5,5),
                                 n_pad_rows=4, n_pad_cols=4, activation='gelu')
  x1m = periodic_convolution(layer_1, 8, kernel=(5,5),
                             n_pad_rows=4, n_pad_cols=4, activation='gelu')
  x1m = periodic_convolution(x1m, 8, kernel=(5,5),
                             n_pad_rows=4, n_pad_cols=4, activation='gelu')
  
  layer_2 = periodic_convolution(input_vel_upsampled, 16, kernel=(9,9),
                                 n_pad_rows=8, n_pad_cols=8, activation='gelu')
  x2m = periodic_convolution(layer_2, 8, kernel=(9,9),
                             n_pad_rows=8, n_pad_cols=8, activation='gelu')
  x2m = periodic_convolution(x2m, 8, kernel=(9,9),
                             n_pad_rows=8, n_pad_cols=8, activation='gelu')
  
  layer_3 = periodic_convolution(input_vel_upsampled, 16, kernel=(13,13),
                                 n_pad_rows=12, n_pad_cols=12, activation='gelu')
  x3m = periodic_convolution(layer_3, 8, kernel=(13,13),
                             n_pad_rows=12, n_pad_cols=12, activation='gelu')
  x3m = periodic_convolution(x3m, 8, kernel=(13,13),
                             n_pad_rows=12, n_pad_cols=12, activation='gelu')

  x_add = Concatenate(axis=-1)([x1m, x2m, x3m, input_vel_upsampled])
  x4m = periodic_convolution(x_add, 8, kernel=(7,7),
                             n_pad_rows=6, n_pad_cols=6, activation='gelu')
  x4m = periodic_convolution(x4m, 3, kernel=(5,5),
                             n_pad_rows=4, n_pad_cols=4, activation='gelu')

  # join both models ... 
  x_final = Concatenate(axis=-1)([x4, x4m])
  x_final = periodic_convolution(x_final, 2, kernel=(3,3),
                                 n_pad_rows=2, n_pad_cols=2, activation='linear')

  x_final = div_free_2D_layer(x_final)
  x_final = circ_filter_layer(x_final)
  return Model(input_vel, x_final)


def super_res_taira_v2_leray(Nx_coarse, Ny_coarse):
  """ Create DSC/MS model from Fukami, Fukagata & Taira,
      for the sake of illutration we insist: 
     
      coarse_graining = 32 so
      Nx_coarse = Nx // 32 
      Ny_coarse = Ny // 32

      Make changes to avoid upsampling the image to target resolution at input 
      Difference is we now use the output of the MS/SC model as input to the parallel filters,
      with a residual connection
  """ 
  # we first build the "skip connection" component
  # we follow the Fukami achitecture "in spirit" because we are doing more severe upsampling
  input_vel = Input(shape=(Nx_coarse, Ny_coarse, 2))

  # DSC
  down_1 = UpSampling2D((2,2))(input_vel)
  x1 = periodic_convolution(down_1, 32, kernel=(4,4),
                            n_pad_rows=3, n_pad_cols=3, activation='gelu')
  x1 = periodic_convolution(x1, 32, kernel=(4,4),
                            n_pad_rows=3, n_pad_cols=3, activation='gelu')
  x1 = UpSampling2D((2,2))(x1)

  down_2 = UpSampling2D((2,2))(down_1)
  x2 = Concatenate(axis=-1)([x1, down_2]) 
  x2 = periodic_convolution(x2, 32, kernel=(4,4),
                            n_pad_rows=3, n_pad_cols=3, activation='gelu')
  x2 = periodic_convolution(x2, 32, kernel=(4,4),
                            n_pad_rows=3, n_pad_cols=3, activation='gelu')
  x2 = UpSampling2D((2,2))(x2)

  down_3 = UpSampling2D((2,2))(down_2)
  x3 = Concatenate(axis=-1)([x2, down_3]) 
  x3 = periodic_convolution(x3, 32, kernel=(4,4),
                            n_pad_rows=3, n_pad_cols=3, activation='gelu')
  x3 = periodic_convolution(x3, 32, kernel=(4,4),
                            n_pad_rows=3, n_pad_cols=3, activation='gelu')
  x3 = UpSampling2D((2,2))(x3)

  down_4 = UpSampling2D((2,2))(down_3)
  x4 = Concatenate(axis=-1)([x3, down_4]) 
  x4 = periodic_convolution(x4, 32, kernel=(4,4),
                            n_pad_rows=3, n_pad_cols=3, activation='gelu')
  x4 = periodic_convolution(x4, 32, kernel=(4,4),
                            n_pad_rows=3, n_pad_cols=3, activation='gelu')
  x4 = UpSampling2D((2,2))(x4)

  x4 = periodic_convolution(x4, 32, kernel=(4,4),
                            n_pad_rows=3, n_pad_cols=3, activation='gelu')
  # this bit is now a velocity field which we will iteratively refine
  x4 = periodic_convolution(x4, 2, kernel=(4,4),
                            n_pad_rows=3, n_pad_cols=3, activation='linear')

  #Multi-scale model (Du et al., 2018)
  layer_1 = periodic_convolution(x4, 16, kernel=(4,4),
                                 n_pad_rows=3, n_pad_cols=3, activation='gelu')
  x1m = periodic_convolution(layer_1, 8, kernel=(4,4),
                             n_pad_rows=3, n_pad_cols=3, activation='gelu')
  x1m = periodic_convolution(x1m, 8, kernel=(4,4),
                             n_pad_rows=3, n_pad_cols=3, activation='gelu')
  
  layer_2 = periodic_convolution(x4, 16, kernel=(6,6),
                                 n_pad_rows=5, n_pad_cols=5, activation='gelu')
  x2m = periodic_convolution(layer_2, 8, kernel=(6,6),
                             n_pad_rows=5, n_pad_cols=5, activation='gelu')
  x2m = periodic_convolution(x2m, 8, kernel=(6,6),
                             n_pad_rows=5, n_pad_cols=5, activation='gelu')
  
  layer_3 = periodic_convolution(x4, 16, kernel=(8,8),
                                 n_pad_rows=7, n_pad_cols=7, activation='gelu')
  x3m = periodic_convolution(layer_3, 8, kernel=(8,8),
                             n_pad_rows=7, n_pad_cols=7, activation='gelu')
  x3m = periodic_convolution(x3m, 8, kernel=(8,8),
                             n_pad_rows=7, n_pad_cols=7, activation='gelu')

  x_add = Concatenate(axis=-1)([x1m, x2m, x3m])
  x4m = periodic_convolution(x_add, 8, kernel=(7,7),
                             n_pad_rows=6, n_pad_cols=6, activation='gelu')
  x4m = periodic_convolution(x4m, 3, kernel=(5,5),
                             n_pad_rows=4, n_pad_cols=4, activation='gelu')

  # join both models ... 
  x_final = Concatenate(axis=-1)([x4, x4m])
  x_final = periodic_convolution(x_final, 2, kernel=(4,4),
                                 n_pad_rows=3, n_pad_cols=3, activation='linear')

  x_final = div_free_2D_layer(x_final)
  x_final = circ_filter_layer(x_final)
  return Model(input_vel, x_final)
