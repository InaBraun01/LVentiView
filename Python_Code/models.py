import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import keras
from keras.layers import Dense, Conv2D, Flatten, Input, MaxPooling2D, UpSampling2D, Concatenate, LeakyReLU, Dropout, BatchNormalization, Lambda, Conv2DTranspose, ReLU
from keras.models import Model

def TernausNet16(input_shape=(256,256,1), pretrained_weights=True, out_channels=3, shift_invariant_mod=False, use_dropout=False):
	#has 31,337,994 params

	assert keras.backend.image_data_format() == 'channels_last'
	assert len(input_shape) == 3 #input images should be HxWxC
	assert input_shape[0] == input_shape[1] #input images should be square

	h = input_layer = Input(input_shape)

	#make the input have three channels if it doesn't already:
	if input_shape[-1] != 3:
		h = Conv2D(3, 1, padding='same')(h) #converts greyscale to rgb using 1x1 convolution

	weights = 'imagenet' if pretrained_weights else None
	model = keras.applications.vgg16.VGG16(include_top=False, input_shape=h.shape[1:], weights=weights)
#	model = keras.applications.vgg16.VGG16(include_top=False, input_shape=h._shape_tuple()[1:], weights=weights)

	#make encoder and get inputs for skip connections:
	for_skip_names, for_skips = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3'], []
	layer_outputs = []
	for layer in model.layers:

		if layer.name.split('_')[1] == 'pool' and shift_invariant_mod == True:
			h = MaxPooling2D(strides=1, padding='same')(h)
		else:
			h = layer(h)
		if layer.name in for_skip_names:
			for_skips.append(h)

	#make decoder using skip connections:
	dbs = [256,256,256,64,32]
	for i in dbs:
		h = Conv2D(i*2, 3, padding='same', activation='relu', kernel_initializer='he_normal')(h)
		h = Dropout(0.5)(h)
		h = Conv2DTranspose(i, 4, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(h)
		h = Concatenate(axis=-1)([h,for_skips.pop()])



	h = Conv2D(32, 3, padding='same', activation='relu', kernel_initializer='he_normal')(h)
	output_layer = Conv2D(4, 1, padding='same', activation='softmax')(h)
	output_layer = Lambda(lambda x : x[...,:3])(output_layer)
	# output_layer = Conv2D(out_channels, 1, padding='same', activation='sigmoid')(h)

	model = Model(input_layer, output_layer)

	return model