import os
import tensorflow as tf
from tensorflow.keras.layers import Input, BatchNormalization, Activation
from tensorflow.keras.layers import TimeDistributed, Conv2D, Lambda, ConvLSTM2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from keras.losses import mean_absolute_error
from training_flags import FLAGS

tf.logging.set_verbosity(tf.logging.ERROR)

def base_layer(x, filters, kernel_size=5, strides=2, activation='relu', kernel_initializer='he_uniform',
               recurrent=False, convolutional=True, reg_lambda=0.00):
    """Base  layer for sequence-to-sequence regression from images.
    Includes a convolutional &/or a recurrent layer (ConvLSTM). These are followed by
    a batch normalization layer and an activation function.

    Convolutional layer: the input is a sequence of images meaning it has the shape
                         (batch_size, seq_len, width, height, channels). For each batch,
                         we want to obtain an output at each timestep so the same convolutional
                         layer should be applied to each timestep's image. For that a Conv2D layer
                         is wrapped around a TimeDistributed layer. Hence the output shape of the
                         convolutional layer is
                         (batch_size, seq_len, conv_out_dim, conv_out_dim, conv_filters)

    Recurrent layer: can be used when there is enough temporal correlation in the image sequence.
                     Uses ConvLSTMs, which replace matrix multiplications in the LSTMs gates by
                     convolution operations, which has been shown to perform better in image data (see:
                     Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting).
                     Because this layer is intended to either output the final sequence or the input to
                     the next base_layer the option return_sequences is set to True, which causes the
                     ConvLSTM2D layer to output the internal state at each timestep and the output to have
                     the shape (batch_size, seq_len, input_width, input_length, conv_lstm_filters).

    """

    assert convolutional or recurrent, \
        "At least one of 'convolutional' and 'recurrent' must be True"

    if convolutional is True:
        x = TimeDistributed(Conv2D(filters=filters,
                                   kernel_size=kernel_size,
                                   strides=strides,
                                   padding='same',
                                   kernel_regularizer=l2(reg_lambda),
                                   kernel_initializer=kernel_initializer))(x)

    if recurrent is True:
        x = ConvLSTM2D(filters=filters,
                       kernel_size=kernel_size,
                       return_sequences=True,
                       padding='same',
                       activation=None,
                       kernel_regularizer=l2(reg_lambda),
                       kernel_initializer=kernel_initializer)(x)

    bn = BatchNormalization()(x)
    layer_output = Activation(activation)(bn)

    return layer_output


def action_inference_model(input_sequence):
    """Add input options: kernel_size, filters, ...
    """
    encoder = base_layer(input_sequence, filters=128, strides=2)

    encoder = base_layer(encoder, filters=64, strides=2)

    encoder = base_layer(encoder, filters=64, strides=2)

    encoder = base_layer(encoder, filters=32, strides=1)

    encoder = base_layer(encoder, filters=16, strides=2)

    encoder = base_layer(encoder, filters=8, strides=1)

    encoder = base_layer(encoder, filters=4, strides=2)

    actions = TimeDistributed(Conv2D(filters=2,
                                     kernel_size=5,
                                     strides=2,
                                     padding='same',
                                     activation='linear',
                                     kernel_initializer='he_uniform'))(encoder)

    actions = Lambda(lambda x: tf.squeeze(x))(actions)

    return actions


def regularization(x, scale_factor=1):
    """
    A regularization term that penalizes close consecutive values.

    First the absolute difference between consecutive timesteps is obtained,
    i.e.,[x_1 - x_0, x_2 - x_1, ... , x_T - x_T-1]. Then the inverse exponential
    of each value is obtained and the values are summed.

    x: the input, of dimensions (batch_size, timesteps, ...)
    scale_factor: a factor that scales the exponential in the x axis
    """
    x_t = x[:, :-1, ...]
    x_tp1 = x[:, 1:, ...]
    return tf.reduce_sum(tf.exp(tf.abs(x_tp1 - x_t)) ** (-scale_factor))


def regularized_mean_absolute_error(y_true, y_pred, K=0.00000):
    K = K if FLAGS.K is None else FLAGS.K
    scale_factor = 300 if FLAGS.scale_factor is None else FLAGS.scale_factor
    return mean_absolute_error(y_true, y_pred) + K * regularization(y_pred, scale_factor=300)


def train_action_inference(inputs, targets, epochs=1, steps_per_epoch=1000, print_layer_sizes=True,
                           val_inputs=None, val_targets=None, validation_steps=None, save_path=None):

    model_input = Input(tensor=inputs)
    actions = action_inference_model(model_input)

    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)

    optimizer = Adam()
    model = Model(model_input, actions)

    model.compile(optimizer=optimizer,
                  loss=regularized_mean_absolute_error,
                  target_tensors=targets)

    if print_layer_sizes is True:
        for layer in model.layers:
            print(layer.name, ': ', layer.output_shape)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=epochs)
    ckpt = ModelCheckpoint(filepath=os.path.join(save_path, 'model_weights.h5'), monitor='val_loss',
                           save_best_only=True)

    history = model.fit(epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        callbacks=[es, ckpt],
                        validation_data=(val_inputs, val_targets),
                        validation_steps=validation_steps)

    return model, history


