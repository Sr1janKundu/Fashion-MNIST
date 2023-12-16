import tensorflow as tf
from keras import Input
from keras.layers import (Dense, Activation, ZeroPadding2D, BatchNormalization,
                          Flatten, Conv2D, AveragePooling2D, MaxPooling2D)
from keras.initializers import glorot_uniform
from keras.models import Model
import utils


def create_resnet50(input_shape, classes):
    """
    Stage-wise implementation of the architecture of the popular ResNet50:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> FLATTEN -> DENSE

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = utils.convolutional_block(X, f=3, filters=[64, 64, 256], s=1)
    X = utils.identity_block(X, 3, [64, 64, 256])
    X = utils.identity_block(X, 3, [64, 64, 256])

    # Stage 3
    # `convolutional_block` with correct values of `f`, `filters` and `s` for this stage
    X = utils.convolutional_block(X, f=3, filters=[128, 128, 512], s=2)
    # the 3 `identity_block` with correct values of `f` and `filters` for this stage
    X = utils.identity_block(X, 3, [128, 128, 512])
    X = utils.identity_block(X, 3, [128, 128, 512])
    X = utils.identity_block(X, 3, [128, 128, 512])

    # Stage 4
    # add `convolutional_block` with correct values of `f`, `filters` and `s` for this stage
    X = utils.convolutional_block(X, f=3, filters=[256, 256, 1024], s=2)
    # the 5 `identity_block` with correct values of `f` and `filters` for this stage
    X = utils.identity_block(X, 3, [256, 256, 1024])
    X = utils.identity_block(X, 3, [256, 256, 1024])
    X = utils.identity_block(X, 3, [256, 256, 1024])
    X = utils.identity_block(X, 3, [256, 256, 1024])
    X = utils.identity_block(X, 3, [256, 256, 1024])

    # Stage 5
    # add `convolutional_block` with correct values of `f`, `filters` and `s` for this stage
    X = utils.convolutional_block(X, f=3, filters=[512, 512, 2048], s=2)
    # the 2 `identity_block` with correct values of `f` and `filters` for this stage
    X = utils.identity_block(X, 3, [512, 512, 2048])
    X = utils.identity_block(X, 3, [512, 512, 2048])

    # AVGPOOL. Use "X = AveragePooling2D()(X)"
    X = AveragePooling2D((1, 1))(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X)

    # Compile model
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01), 
        loss = tf.keras.losses.CategoricalCrossentropy(), 
        metrics = ['accuracy'])

    return model