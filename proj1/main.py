import os
import time
import argparse
import skimage.io as skio

from colorization import *
from contrast import *

def main(args):
	datadir = args.datadir
	outdir = args.outdir
	if args.contrast:
		imgs = os.listdir('./'+datadir)
		if not os.path.exists(os.path.join(outdir)):
			os.makedirs(os.path.join(outdir))

		for img in imgs:
			if "jpg" in img:
				impath = './{0}/{1}'.format(datadir, img)
				im = skio.imread(impath)
				im_out = hist_eqaulization(im, args.plot_hist)
				skio.imsave('./{0}/contrast_{1}'.format(outdir, img), im_out)
	else:
		if args.metric == 'L2loss':
			metric = L2loss
		elif args.metric == 'NCCloss':
			metric = NCCloss
		else:
			raise NotImplementedError

		if not os.path.exists(os.path.join(outdir)):
			os.makedirs(os.path.join(outdir))

		imgs = os.listdir('./'+datadir)
		print('Colorzing the following images:', imgs)
		for img in imgs:
			impath = './{0}/{1}'.format(datadir, img)
			# read in the image
			im = skio.imread(impath)
			print('====current image: {0}, shape: {1}===='.format(img, im.shape))

			if "jpg" in img:
				im_out, im_name = colorize(im, img, 'low', metric, args.level)
			elif "tif" in img:
				im_out, im_name = colorize(im, img, 'high', metric, args.level)

			fname = './{0}/{1}'.format(outdir, im_name)
			# save the image
			skio.imsave(fname, im_out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=5)
    parser.add_argument("--outdir", type=str, default='results')
    parser.add_argument("--datadir", type=str, default='data')
    parser.add_argument("--metric", type=str, help='NCCloss/L2loss', default='NCCloss')
    parser.add_argument("--contrast", action='store_true')
    parser.add_argument("--plot_hist", action='store_true')
    args = parser.parse_args()

    main(args)

