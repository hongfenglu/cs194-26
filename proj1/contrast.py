import os
import time
import numpy as np
import skimage as sk
import skimage.io as skio
import multiprocessing
from skimage import feature
from matplotlib import pyplot as plt

def hist_eqaulization(img, plot_hist=False):
	im_out = img

	for i in range(3):
		img = im_out[:, :, i]
		hist, _ = np.histogram(img.flatten(), 256, [0,256])

		cdf = hist.cumsum()
		if plot_hist:
			cdf_normalized = cdf * hist.max() / cdf.max()
			plt.plot(cdf_normalized, color='b')
			plt.hist(img.flatten(), 256,[0, 256], color='r')
			plt.xlim([0, 256])
			plt.legend(('cdf', 'histogram'), loc='best')
			plt.show()

		cdf_mask = np.ma.masked_equal(cdf, 0)
		cdf_mask = (cdf_mask - cdf_mask.min()) * 255 / (cdf_mask.max() - cdf_mask.min())
		cdf = np.ma.filled(cdf_mask, 0).astype('uint8')
		im_out[:, :, i] = cdf[img]

	if plot_hist:
		for i in range(3):
			img = im_out[:, :, i]
			hist, _ = np.histogram(img.flatten(), 256, [0,256])
			cdf = hist.cumsum()
			cdf_normalized = cdf * hist.max() / cdf.max()
			plt.plot(cdf_normalized, color='b')
			plt.hist(img.flatten(), 256, [0, 256], color='r')
			plt.xlim([0, 256])
			plt.legend(('cdf', 'histogram'), loc='best')
			plt.show()

	skio.imshow(im_out)
	skio.show()
	return im_out



