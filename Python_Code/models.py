import os
import tensorflow as tf
from keras.layers import (
    Dense, Conv2D, Flatten, Input, MaxPooling2D, UpSampling2D, 
    Concatenate, LeakyReLU, Dropout, BatchNormalization, Lambda, 
    Conv2DTranspose, ReLU
)
from keras.models import Model
from keras.applications import VGG16

# Configure TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def TernausNet16(
    input_shape=(256, 256, 1),
    pretrained_weights=True,
    out_channels=3,
    shift_invariant_mod=False,
    use_dropout=False
):
    """
    TernausNet16 model based on VGG16 encoder with U-Net decoder.
    
    Args:
        input_shape (tuple): Shape of input images (H, W, C)
        pretrained_weights (bool): Whether to use ImageNet pretrained weights
        out_channels (int): Number of output channels
        shift_invariant_mod (bool): Use shift-invariant pooling
        use_dropout (bool): Whether to apply dropout (currently unused)
        
    Returns:
        keras.Model: Compiled TernausNet16 model
        
    Note: Has 31,337,994 parameters
    """
    # Input validation
    assert tf.keras.backend.image_data_format() == 'channels_last', \
        "Expected channels_last data format"
    assert len(input_shape) == 3, \
        "Input images should be HxWxC format"
    assert input_shape[0] == input_shape[1], \
        "Input images should be square"
    
    # Define input layer
    input_layer = Input(input_shape)
    x = input_layer
    
    # Convert single channel to RGB if needed
    if input_shape[-1] != 3:
        x = Conv2D(
            filters=3,
            kernel_size=1,
            padding='same',
            name='input_channel_converter'
        )(x)
    
    # Load VGG16 encoder
    weights = 'imagenet' if pretrained_weights else None
    vgg16_model = VGG16(
        include_top=False,
        input_shape=x.shape[1:],
        weights=weights
    )
    
    # Define layers for skip connections
    skip_connection_layers = [
        'block1_conv2',
        'block2_conv2', 
        'block3_conv3',
        'block4_conv3',
        'block5_conv3'
    ]
    
    # Build encoder and collect skip connections
    skip_connections = []
    
    for layer in vgg16_model.layers:
        # Apply shift-invariant pooling if requested
        if shift_invariant_mod and layer.name.split('_')[1] == 'pool':
            x = MaxPooling2D(strides=1, padding='same')(x)
        else:
            x = layer(x)
        
        # Store outputs for skip connections
        if layer.name in skip_connection_layers:
            skip_connections.append(x)
    
    # Build decoder with skip connections
    decoder_filters = [256, 256, 256, 64, 32]
    
    for i, filters in enumerate(decoder_filters):
        # Double convolution block
        x = Conv2D(
            filters=filters * 2,
            kernel_size=3,
            padding='same',
            activation='relu',
            kernel_initializer='he_normal',
            name=f'decoder_conv_{i}_1'
        )(x)
        
        x = Dropout(0.5, name=f'decoder_dropout_{i}')(x)
        
        # Transpose convolution for upsampling
        x = Conv2DTranspose(
            filters=filters,
            kernel_size=4,
            strides=2,
            padding='same',
            activation='relu',
            kernel_initializer='he_normal',
            name=f'decoder_transpose_{i}'
        )(x)
        
        # Skip connection
        if skip_connections:
            skip_connection = skip_connections.pop()
            x = Concatenate(axis=-1, name=f'skip_concat_{i}')([x, skip_connection])
    
    # Final convolution layers
    x = Conv2D(
        filters=32,
        kernel_size=3,
        padding='same',
        activation='relu',
        kernel_initializer='he_normal',
        name='final_conv'
    )(x)
    
    # Output layer with softmax activation
    output = Conv2D(
        filters=4,
        kernel_size=1,
        padding='same',
        activation='softmax',
        name='output_conv'
    )(x)
    
    # Extract first 3 channels from 4-channel softmax output
    output_layer = Lambda(
        lambda x: x[..., :3],
        name='output_slice'
    )(output)
    
    # Create and return model
    model = Model(inputs=input_layer, outputs=output_layer, name='TernausNet16')
    
    return model