import os, click, cv2
import numpy as np
import mxnet as mx
import mxnet.ndarray as F
from net import Net
from utils import StyleLoader

@click.command(help="""
    Takes web-camera dev/video0 stream and stylizes it according to pretrained MXNet model. 
    By default it's 21-styles model from  https://github.com/StacyYang/MXNet-Gluon-Style-Transfer \n
    Output stream combines input & stylized video  as well as style image \n
    Press ESC key to stop   """)
@click.option('--cuda', is_flag=True, help='to use CUDA GPU, by default uses CPU')
@click.option('--record', is_flag=True, help='to write video to "output.mp4" file ')
def run_demo(cuda, record):
	if cuda:
		ctx = mx.gpu(0); os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
	else:
		ctx = mx.cpu(0)
	model = 'models/21styles.params'
	ngf = 128
	style_size = 512
	style_folder = 'images/styles/'
	demo_size = 480	# demo window height
	mirror = False
	style_loader = StyleLoader(style_folder, style_size, ctx)
	style_model = Net(ngf=ngf)
	style_model.load_parameters(model, ctx=ctx)
	# Define the codec and create VideoWriter object
	height = demo_size
	width = int(4.0/3*demo_size)
	swidth = int(width/4); sheight = int(height/4)
	if record:
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
		# changing styles
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
		key = cv2.waitKey(1)
		if record:
			out.write(img)
		if key == 27: # Esc
			break
	cam.release()
	if record:
		out.release()
	cv2.destroyAllWindows()

def main(): run_demo()

if __name__ == '__main__': main()
