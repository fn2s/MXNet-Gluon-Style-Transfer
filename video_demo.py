import os, click, cv2, json
import numpy as np
import mxnet as mx
import mxnet.ndarray as F
from skvideo.io import FFmpegWriter, vreader, ffprobe
from net import Net
from utils import StyleLoader
from transferAudio import transferAudio

@click.command(help="""
    VFILE: name of input video file placed in directory 'video' \n
    Takes video file and stylizes it according to pretrained MXNet model.     
    By default it's 21-styles model from  https://github.com/StacyYang/MXNet-Gluon-Style-Transfer \n
    Output stream combines input & stylized video  as well as style image """)
@click.argument('vfile')
@click.option('-c', '--cuda', is_flag=True, help='to use CUDA GPU, by default uses CPU')
@click.option('-r', '--record', is_flag=True, help='to write video to "video/VFILE-output21.mp4" file ')
def run_demo(cuda, record, vfile):
	model = 'models/21styles.params'
	ngf = 128
	style_size = 512
	style_folder = 'images/styles/'
	mirror = False
	vDir = './video/'
	vPath = vDir + vfile
	oFile = 'output21-'+ vfile
	wM, hM = 640, 480
	if cuda:
		ctx = mx.gpu(0); os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
	else:
		ctx = mx.cpu(0)
	style_loader = StyleLoader(style_folder, style_size, ctx)
	style_model = Net(ngf=ngf)
	style_model.load_parameters(model, ctx=ctx)
	metadata = ffprobe(vPath)
	fps = metadata["video"]["@avg_frame_rate"]
#	print(json.dumps(metadata["video"], indent=4))
	w, h = int(metadata["video"]["@width"]), int(metadata["video"]["@height"])
	downsize = h > hM
	if downsize :
		w = 2 * int(w * hM / h / 2); h = hM
#	downsize = w > wM
#	if downsize :
#		h = 2 * int(h * wM / w / 2); w = wM
	swidth = int(w/4); sheight = int(h/4)
	wName = vfile +'  STYLIZED VIDEO   fps:'+ fps + '  W:'+ str(w) +'  H:'+ str(h)
	if record:
		out = FFmpegWriter(vDir+oFile, inputdict={'-r': str(fps),'-s':'{}x{}'.format(2*w, h)},
					outputdict={'-r': str(fps),'-c:v': 'h264'})
	key, idx = 0, 0
	cv2.namedWindow(wName,cv2.WINDOW_NORMAL)
	cv2.resizeWindow(wName, 2*w, h)
	for img in vreader(vPath):
		idx += 1
		if downsize :
			img = cv2.resize(img, (w, h), interpolation = cv2.INTER_AREA)
		if mirror: 
			img = cv2.flip(img, 1)
		cimg = img.copy()
		img = np.array(img).transpose(2, 0, 1).astype(float)
		img = F.expand_dims(mx.nd.array(img, ctx=ctx), 0)
		# changing styles
		if idx%50 == 1:
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
		img = np.concatenate((cimg, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),axis=1)
		if record:
			out.writeFrame(img)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		cv2.imshow(wName, img)
		key = cv2.waitKey(1)
		if key == 27: # Esc
			break
	if record:
		out.close()
		transferAudio(vPath, vDir, oFile)
		print("Done OK. Created Stylised Video file", vDir+oFile)
		print ("fps :", fps, "    W:", w, " H:", h)
	cv2.destroyAllWindows()

def main(): run_demo()

if __name__ == '__main__': main()
