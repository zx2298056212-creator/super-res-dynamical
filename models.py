""" Attempt to convert old tf keras model to jax backend """
import keras.backend as K
import keras.ops as kops
from keras.layers import Conv2D, Lambda
from keras.layers import Input, MaxPooling2D, Flatten, Dense, Reshape, UpSampling2D, AveragePooling2D
from keras.layers import BatchNormalization, Concatenate, Activation

from keras.models import Model

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


def dense_block_periodic_conv(x, n_blocks, growth_rate,
                              kernel=(1,1), strides=(1,1),
                              n_pad_rows=1, n_pad_cols=1,
                              activation='relu'):
  for j in range(n_blocks):
    xj = BatchNormalization()(x)
    xj = Activation(activation)(xj)
    xj = periodic_convolution(xj, growth_rate, kernel, strides=strides,
                              n_pad_rows=n_pad_rows, n_pad_cols=n_pad_cols)
    x = Concatenate(axis=-1)([x, xj])
  return x


def ae_densenet_v7(Nx, Ny, encoded_dim, return_encoder_decoder=False):
  """ 
  More feature maps, no FC 
  """
  input_vort = Input(shape=(Nx, Ny, 1))
   
  # Dense block (1) -- raw features on full field
  x = periodic_convolution(input_vort, 64, (8,8), strides=(1,1),
                           n_pad_rows=7, n_pad_cols=7, activation='gelu')
  x = dense_block_periodic_conv(x, 3, 32, (8,8), strides=(1,1),
                                n_pad_rows=7, n_pad_cols=7, activation='gelu')
  x = BatchNormalization()(x)
  x = Activation('gelu')(x)
  x = MaxPooling2D((2,2), padding='same')(x)

  # 64 x 64
  x = periodic_convolution(x, 32, (4,4), strides=(1,1),
                           n_pad_rows=3, n_pad_cols=3, activation='gelu')
  x = BatchNormalization()(x)
  x = Activation('gelu')(x)
  x = dense_block_periodic_conv(x, 3, 32, (4,4), strides=(1,1),
                                n_pad_rows=3, n_pad_cols=3, activation='gelu')
  x = MaxPooling2D((2,2), padding='same')(x)

  # 32 x 32
  x = periodic_convolution(x, 32, (4,4), strides=(1,1),
                           n_pad_rows=3, n_pad_cols=3, activation='gelu')
  x = BatchNormalization()(x)
  x = Activation('gelu')(x)
  x = dense_block_periodic_conv(x, 3, 32, (4,4), strides=(1,1),
                                n_pad_rows=3, n_pad_cols=3, activation='gelu')
  x = MaxPooling2D((2,2), padding='same')(x)

  
  # 16 x 16
  x = periodic_convolution(x, 32, (2,2), strides=(1,1),
                           n_pad_rows=1, n_pad_cols=1, activation='gelu')
  x = BatchNormalization()(x)
  x = Activation('gelu')(x)
  x = dense_block_periodic_conv(x, 3, 32, (2,2), strides=(1,1),
                                n_pad_rows=1, n_pad_cols=1, activation='gelu')
  x = MaxPooling2D((2,2))(x)

  # 8 x 8
  x = periodic_convolution(x, 32, (2,2), strides=(1,1), 
                           n_pad_rows=1, n_pad_cols=1, activation='gelu')
  x = BatchNormalization()(x)
  x = Activation('gelu')(x)
  x = dense_block_periodic_conv(x, 3, 32, (2,2), strides=(1,1),
                                n_pad_rows=1, n_pad_cols=1, activation='gelu')
  x = MaxPooling2D((2,1))(x)

  
  # 4 x 8 
  x = periodic_convolution(x, 32, (2,2), strides=(1,1), 
                           n_pad_rows=1, n_pad_cols=1, activation='gelu')
  x = BatchNormalization()(x)
  x = Activation('gelu')(x)
  x = dense_block_periodic_conv(x, 3, 32, (2,2), strides=(1,1),
                                n_pad_rows=1, n_pad_cols=1, activation='gelu')

  # Embedding
  encoded_final_feature_maps = periodic_convolution(x, encoded_dim // 32, (2,2), strides=(1,1), 
                                                    n_pad_rows=1, n_pad_cols=1, activation='gelu')
  x = BatchNormalization()(encoded_final_feature_maps)
  x = Activation('gelu')(x)

  # create two decoder blocks if also returning decoder
  if return_encoder_decoder == True:
    input_decoder = Input(shape=(*x.shape[1:],))
    decoder_inputs = [x, input_decoder]
    encoder_model = Model(input_vort, x)
  else:
    decoder_inputs = [x]
  decoder_outputs = []

  for decoder_input in decoder_inputs:
    # starting at 4 x 8 x m // 32
    x = UpSampling2D((2,1))(decoder_input)
    x = periodic_convolution(x, 32, (2,2), strides=(1,1), 
                             n_pad_rows=1, n_pad_cols=1, activation='gelu')
    x = BatchNormalization()(x)
    x = Activation('gelu')(x)
    x = dense_block_periodic_conv(x, 3, 32, (2,2), strides=(1,1),
                                n_pad_rows=1, n_pad_cols=1, activation='gelu')

    # now at 8 x 8 x ? 
    x = UpSampling2D((2,2))(x)
    x = periodic_convolution(x, 32, (2,2), strides=(1,1), 
                             n_pad_rows=1, n_pad_cols=1, activation='gelu')
    x = BatchNormalization()(x)
    x = Activation('gelu')(x)
    x = dense_block_periodic_conv(x, 3, 32, (2,2), strides=(1,1),
                                n_pad_rows=1, n_pad_cols=1, activation='gelu')

    # now at 16 x 16
    x = UpSampling2D((2,2))(x)
    x = periodic_convolution(x, 32, (4,4), strides=(1,1), 
                             n_pad_rows=3, n_pad_cols=3, activation='gelu')
    x = BatchNormalization()(x)
    x = Activation('gelu')(x)
    x = dense_block_periodic_conv(x, 3, 32, (4,4), strides=(1,1),
                                n_pad_rows=3, n_pad_cols=3, activation='gelu')
    
    # now at 32 x 32
    x = UpSampling2D((2,2))(x)
    x = periodic_convolution(x, 32, (4,4), strides=(1,1), 
                             n_pad_rows=3, n_pad_cols=3, activation='gelu')
    x = BatchNormalization()(x)
    x = Activation('gelu')(x)
    x = dense_block_periodic_conv(x, 3, 32, (4,4), strides=(1,1),
                                n_pad_rows=3, n_pad_cols=3, activation='gelu')
    
    # now at 64 x 64
    x = UpSampling2D((2,2))(x)
    x = periodic_convolution(x, 32, (8,8), strides=(1,1), 
                             n_pad_rows=7, n_pad_cols=7, activation='gelu')
    x = BatchNormalization()(x)
    x = Activation('gelu')(x)
    x = dense_block_periodic_conv(x, 3, 32, (8,8), strides=(1,1),
                                n_pad_rows=7, n_pad_cols=7, activation='gelu')

    # return to original configuration 
    x = periodic_convolution(x, 1, (8,8), strides=(1,1), n_pad_rows=7, n_pad_cols=7, activation='tanh')
    decoder_outputs.append(x)
  
  if return_encoder_decoder == True:
    return Model(input_vort, decoder_outputs[0]), encoder_model, Model(decoder_inputs[1], decoder_outputs[1])
  else:
    return Model(input_vort, decoder_outputs[0])
