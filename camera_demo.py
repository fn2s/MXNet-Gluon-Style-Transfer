import os
import cv2
import numpy as np
import mxnet as mx
#import torch
#from torch.autograd import Variable
from mxnet import autograd, gluon
from mxnet.gluon import nn, Block, HybridBlock, Parameter, ParameterDict
import mxnet.ndarray as F

import net
from option import Options
import utils
from utils import StyleLoader, cudamem

def run_demo(args, mirror=False):
	if args.cuda:
		ctx = mx.gpu(0); os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
	else:
		ctx = mx.cpu(0)
	style_loader = StyleLoader(args.style_folder, args.style_size, ctx)
	style_model = net.Net(ngf=args.ngf)
	style_model.load_parameters(args.model, ctx=ctx)
#	model_dict = torch.load(args.model)
#	model_dict_clone = model_dict.copy()
#	for key, value in model_dict_clone.items():
#		if key.endswith(('running_mean', 'running_var')):
#			del model_dict[key]
#	style_model.load_state_dict(model_dict, False)
#	style_model.eval()
	# Define the codec and create VideoWriter object
	height = args.demo_size
	width = int(4.0/3*args.demo_size)
	swidth = int(width/4); sheight = int(height/4)
	if args.record:
		fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
		out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (2*width, height))
	cam = cv2.VideoCapture(0)
	cam.set(3, width); cam.set(4, height)
	key, idx = 0, 0
	wName = 'STYLIZED VIDEO  W:'+ str(width) +'  H:'+ str(height)
	cv2.namedWindow(wName,cv2.WINDOW_NORMAL)
	cv2.resizeWindow(wName, width*2, height)
	while True:
		# read frame
		idx += 1
		ret_val, img = cam.read()
		if mirror: 
			img = cv2.flip(img, 1)
		cimg = img.copy()
		img = np.array(img).transpose(2, 0, 1).astype(float)
		img = F.expand_dims(mx.nd.array(img, ctx=ctx), 0)
		# changing style s
		if idx%20 == 1:
			style_v = style_loader.get(int(idx/20))
			style_model.set_target(style_v)
		img = style_model(img)

		simg = np.squeeze(style_v.asnumpy())
		simg = simg.transpose(1, 2, 0).astype('uint8')
		img = F.clip(img[0], 0, 255).asnumpy()
		img = img.transpose(1, 2, 0).astype('uint8')

		# display
		simg = cv2.resize(simg,(swidth, sheight), interpolation = cv2.INTER_CUBIC)
		cimg[0:sheight,0:swidth,:]=simg
		img = np.concatenate((cimg,img),axis=1)
		cv2.imshow(wName, img)
		#cv2.imwrite('stylized/%i.jpg'%idx,img)
		key = cv2.waitKey(1)
		print("End of LOOP"); cudamem()
		if args.record:
			out.write(img)
		if key == 27: 
			break
	cam.release()
	if args.record:
		out.release()
	cv2.destroyAllWindows()

def main():
	# getting things ready
	args = Options().parse()
	if args.subcommand is None:
		raise ValueError("ERROR: specify the experiment type")
	# run demo
	run_demo(args, mirror=False)

if __name__ == '__main__':
	main()
