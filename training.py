import tensorflow as tf
import os

import models  
import data_read_v2 as reader 

# Define constants
Nx = 128
Ny = 128
m = 128
norm_factor = 60
batch_size = 64
num_epochs = 500
lr = 1e-3
file_loc = 'gs://whirl_turbulence_cambridge/Re_400_whirl_01_27_2023/*'
model_dir = '/home/jp789/autoencoders_2023/models'
model_weight_name = 'weights_DNv7a5_Re400_m'+str(m)+'_lr'+str(lr)+'_epoch{epoch:04d}'
model_weight_path = os.path.join(model_dir, model_weight_name)

# Create a MirroredStrategy for distributing across GPUs
strategy = tf.distribute.MirroredStrategy()

# Define custom loss function
def mse_and_vortsq_loss(vort_true, vort_pred):
  mse = tf.losses.mean_squared_error(vort_true, vort_pred)
  vort_true_sq = tf.square(vort_true)
  vort_pred_sq = tf.square(vort_pred)
  mse_vort_sq = tf.reduce_mean(tf.square(vort_true_sq - vort_pred_sq))
  loss = 0.5 * mse + 0.5 * mse_vort_sq
  return loss

# Define learning rate schedule [has made some impact even with Adam in AE] 
def lr_schedule(epoch, initial_lr):
  return initial_lr 
  #if epoch < 100:
  #  return initial_lr
  #elif epoch < 250:
  #  return 0.1 * initial_lr
  #else:
  #  return 0.01 * initial_lr

# Load and compile model inside strategy scope 
with strategy.scope():
  dataset_dict = reader.read_xarray_tfdata_glob(file_loc, batch_size, 'complete', norm_factor, train_frac=0.9)
  model = models.ae_densenet_v7(Nx, Ny, m)
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
      loss=mse_and_vortsq_loss, #'mse',
      metrics=[tf.keras.losses.MeanSquaredError()]
  )
print("LOADING DONE!")

# Define callbacks
callbacks = [
      tf.keras.callbacks.ModelCheckpoint(
          model_weight_path, save_weights_only=True,
          save_best_only=True),
      tf.keras.callbacks.TensorBoard(log_dir=model_dir),
      tf.keras.callbacks.LearningRateScheduler(lambda epoch: lr_schedule(epoch, lr))
  ]


# Train model
with strategy.scope():
  history = model.fit(
      dataset_dict['train'],
      epochs=num_epochs,
      steps_per_epoch=dataset_dict['num_train_samples'] // batch_size,
      validation_data=dataset_dict['val'],
      callbacks=callbacks)
