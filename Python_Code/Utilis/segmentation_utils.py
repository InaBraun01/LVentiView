
import sys
import numpy as np
from scipy.ndimage import zoom,label
from scipy.ndimage.measurements import center_of_mass
import keras.backend as K
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from Python_Code.models import TernausNet16
import Python_Code.models_pytorch as pytorch


def produceSegAtRequiredRes(data, pixel_spacing, is_sax=True):

	dicom_data_shape = data.shape

	#zoom to the correct resolution (i.e. 1mmx1mm):
	data = zoom(data,(1, 1, pixel_spacing[1], pixel_spacing[2]), order=1)

	# normalize intensities:
	data = data - data.min()
	data = np.clip(data, 0, np.percentile(data, 99.5))
	data = data / data.max()

	#load the segmentation network:
	model = TernausNet16((256,256,1))
	if is_sax:
		model.load_weights('SegmentationModels/my_model.h5')
	else:
		model.load_weights('SegmentationModels/my_LAX_model.h5')

	#segment image, finding 256x256 window centered on LV
	pred, c1, c2 = getSeg(data, model) #returns the segmentation and the pixel co-orinates for the center of the LV

	pred = np.sum(pred*[[[[[1,2,3]]]]], axis=-1)

	K.clear_session() # delete the model

	return data, pred, c1, c2

def getImageAt(c1,c2,data,sz=256):
	#extract a square image of size sz with the top left corner at the given point, cropping and padding as required
	return np.pad(data+0,((0,0),(0,0),(sz//2,sz//2),(sz//2,sz//2)))[:,:,c1:c1+sz,c2:c2+sz]

def hardSoftMax(pred, threshold=0.3):
	
	content = np.sum(pred, axis=-1) > threshold
	pred_class = np.argmax(pred, axis=-1)
	pred *= 0
	for i in range(3):
		pred[pred_class==i,i]=1
	pred *= content[...,None]
	return pred

def getSeg(data, m, sz=256):
	# iterativley search for the 256x256 window centered on the LV

	_,_,c1,c2 = data.shape
	c1,c2 = c1//2,c2//2
	center_moved = True

	all_c1c2 = [(c1,c2)]

	center_moved_counter = -1
	while center_moved:
		center_moved_counter += 1
		center_moved = False
		roi = getImageAt(c1,c2,data).reshape((-1,sz,sz,1))

		pred = m.predict(roi)

		pred = hardSoftMax(pred)

		# imageio.imwrite('center_search_debug_%d.png' % (center_moved_counter,), np.mean(pred, axis=0))

		new_c1, new_c2 = center_of_mass(np.mean(pred, axis=0)[...,2])

		if np.isnan(new_c1) or np.isnan(new_c2):
			print(new_c1, new_c2)
			print('nothing found in center of image(s) for this series, aborting. Exclude the series or shift it to center on the heart')
			sys.exit()

		new_c1, new_c2 = int(np.round(new_c1)), int(np.round(new_c2))
		
		new_c1 = c1 + new_c1-sz//2
		new_c2 = c2 + new_c2-sz//2
		# print("%d --> %d, %d --> %d" %(c1,new_c1,c2,new_c2))
		if np.abs(c1 - new_c1) > 2 or np.abs(c2 - new_c2) > 2:
			center_moved = True
			c1 = new_c1
			c2 = new_c2

			if (c1,c2) in all_c1c2:
				#stuck in a loop
				all_c1c2 = all_c1c2[all_c1c2.index((c1,c2)):]
				print('averaging points in loop:')
				print(all_c1c2)
				c1, c2 = np.mean(all_c1c2, axis=0).astype('int')
				print(c1,c2)
				break

			all_c1c2.append((c1,c2))


	pred = np.pad(pred,((0,0),(c1,data.shape[2]-c1),(c2,data.shape[3]-c2),(0,0)))
	pred = pred[:,sz//2:-sz//2,sz//2:-sz//2]
	pred = pred.reshape(data.shape+(3,))


	return pred, c1, c2

def simpleShapeCorrection(msk):

	print(msk.shape, msk.min(), msk.max())

	#slicewise:
	for i in range(msk.shape[0]):
		for j in range(msk.shape[1]):

			### keep only largest cc for myo and LV-BP channels (i.e. green and blue):
			lvmyo = msk[i,j]==2
			labels, count = label(lvmyo) # value of 2 = green = LV-myo
			if count != 0:
				largest_cc = np.argmax( [np.sum(labels==k) for k in range(1, count + 1)] ) + 1
				msk[i,j] -= (1-(labels==largest_cc))*(lvmyo)*2

			lvbp = msk[i,j]==3
			labels, count = label(lvbp) # value of 3 = blue = LV-BP
			if count != 0:
				largest_cc = np.argmax( [np.sum(labels==k) for k in range(1, count + 1)] ) + 1
				msk[i,j] -= (1-(labels==largest_cc))*(lvbp)*2

			### remove any components smaller than 20pixels from the RV-BP channel:
			rvbp = msk[i,j]==1
			labels, count = label(rvbp) # value of 3 = blue = RV-BP
			for idx in range(1, count + 1):
				cc = labels==idx
				if np.sum(cc) < 20:
					msk[i,j] *= (1-cc)

	return msk

