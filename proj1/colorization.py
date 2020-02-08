# CS194-26 (CS294-26): Project 1 starter Python code
# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images
import os
import time
import itertools
import numpy as np
import skimage as sk
import skimage.io as skio
import multiprocessing
from skimage import feature
from joblib import Parallel, delayed
from matplotlib import pyplot as plt


def L2loss(image1, image2):
	return np.mean((image1-image2)**2)

def NCCloss(image1, image2):
	mean1 = np.mean(image1)
	mean2 = np.mean(image2)
	ncc = - np.sum((image2 - mean2)*(image1 - mean1)) #/ ((np.sum((image2-mean2)**2) * np.sum((image1-mean1)**2))**0.5)
	return ncc

def displace(image, displacement):
	image = np.roll(image, displacement[0], axis=0)
	image = np.roll(image, displacement[1], axis=1)
	return image

def computeLoss(im1, im2, d_pair, metric, edge_w):
	curr_im2 = np.roll(im2, d_pair[0], axis=0)
	curr_im2 = np.roll(curr_im2, d_pair[1], axis=1)
	loss = metric(im1[edge_w:-edge_w, edge_w:-edge_w], curr_im2[edge_w:-edge_w, edge_w:-edge_w])
	return loss

def align(im1, im2, d, metric):
	# Parallel search for the best displacement in the window of possible displacement
	d_pairs = [(i, j) for i in np.arange(-d, d) for j in np.arange(-d, d)] 
	num_cores = multiprocessing.cpu_count()
	edge_width = int(im1.shape[0]*0.05)
	losses = Parallel(n_jobs=num_cores)(delayed(computeLoss)(im1, im2, d_pair, metric, edge_width) for d_pair in d_pairs)
	best_displacement = d_pairs[np.argmin(losses)]
	return best_displacement

def iterative_align(im1, im2, pyramid_depth, metric):
	d = 15
	# initialize cumulative displacement
	cum_displacement = np.array([0, 0])
	level = 0
	while pyramid_depth >= 0:
		scale = 0.5**pyramid_depth
		im1_rescaled = sk.transform.rescale(im1, scale, anti_aliasing=True)
		im2_rescaled = sk.transform.rescale(im2, scale, anti_aliasing=True)
		im2_rescaled = displace(im2_rescaled, 2*cum_displacement)

		# compute the best displacement for the current level
		curr_displacement = align(im1_rescaled, im2_rescaled, d, metric)

		cum_displacement = cum_displacement*2 + curr_displacement
		print('current scale: {0}, cumulative displacement: {1}, '.format(scale, cum_displacement))
		d = 5
		pyramid_depth -= 1
	return cum_displacement

def colorize(im, imname, res, metric, level):
	ts = time.time()

	# convert to double (might want to do this later on to save memory)    
	im = sk.img_as_float(im)

	# compute the height of each part (just 1/3 of total)
	height = np.floor(im.shape[0] / 3.0).astype(np.int)

	# separate color channels
	b = im[:height]
	g = im[height: 2*height]
	r = im[2*height: 3*height]

	# align the images
	if res=='low':
		g_disp = align(b, g, 15, metric)
		r_disp = align(b, r, 15, metric)
		ag = displace(g, g_disp)
		ar = displace(r, r_disp)
	elif res=='high':
		# import ipdb; ipdb.set_trace()
		g_disp = iterative_align(b, g, level-1, metric)
		print('g displacement: ', g_disp)
		r_disp = iterative_align(b, r, level-1, metric)
		print('r displacement: ', r_disp)
		ag = displace(g, g_disp)
		ar = displace(r, r_disp)
	else:
		raise NotImplementedError
	# create a color image
	im_out = np.dstack([ar, ag, b])
	im_name = imname[:-4] + \
					'_{0}_{1}_{2}_{3}.jpg'.format(g_disp[0], g_disp[1], r_disp[0], r_disp[1])
	print('use time: ', time.time() - ts)

	# display the image
	# skio.imshow(im_out)
	# skio.show()
	return im_out, im_name



