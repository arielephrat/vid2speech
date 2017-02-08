import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import sys
import audio_tools as aud
import audio_read as ar

SR = 8000
FPS = 25
SPF = int(SR/FPS)
FRAME_ROWS = 128
FRAME_COLS = 128
NFRAMES = 9 # size of input volume of frames
MARGIN = NFRAMES/2
COLORS = 1 # grayscale
CHANNELS = COLORS*NFRAMES
OVERLAP = 1.0/2
LPC_ORDER = 8
NET_OUT = int(1/OVERLAP)*(LPC_ORDER+1)
SEQ_LEN = 75
SAMPLE_LEN = SEQ_LEN-2*MARGIN
MAX_FRAMES = 1000*SAMPLE_LEN
DEFAULT_FACE_DIM = 292
CASC_PATH = 'haarcascade_frontalface_alt.xml'
datapath = '../dataset'
VIDDATA_PATH = 'viddata.npy'
AUDDATA_PATH = 'auddata.npy'

def process_video(vf, viddata, vidctr, faceCascade):
	temp_frames = np.zeros((SEQ_LEN*CHANNELS,FRAME_ROWS,FRAME_COLS),dtype="uint8")
	cap = cv2.VideoCapture(vf)
	faceCascade = cv2.CascadeClassifier(CASC_PATH)
	for i in np.arange(SEQ_LEN):
		ret,frame = cap.read()
		if ret==0:
			break
		frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		faces = faceCascade.detectMultiScale(
			frame,
			scaleFactor=1.05,
			minNeighbors=3,
			minSize=(200,200),
			flags = 2
		)
		if len(faces)==0 or len(faces)>1:
			print ('Face detection error in %s frame: %d'%(vf, i))							
			face = frame[199:-85,214:-214] # hard-coded face location
		else:
			for (x,y,w,h) in faces:
				face = frame[y:y+DEFAULT_FACE_DIM,x:x+DEFAULT_FACE_DIM]
		face = cv2.resize(face,(FRAME_COLS,FRAME_ROWS))
		face = np.expand_dims(face,axis=2)
		face = face.transpose(2,0,1)
		temp_frames[i*COLORS:i*COLORS+COLORS,:,:] = face	
	for i in np.arange(MARGIN,SAMPLE_LEN+MARGIN):
		viddata[vidctr,:,:,:] = temp_frames[COLORS*(i-MARGIN):COLORS*(i+MARGIN+1),:,:]
		vidctr = vidctr+1
	return vidctr

def process_audio(af, auddata, audctr):
	# audio processing
	(y,sr) = ar.audio_read(af,sr=SR)
	win_length = SPF
	hop_length = int(SPF*OVERLAP)
	[a,g,e] = aud.lpc_analysis(y,LPC_ORDER,window_step=hop_length,window_size=win_length)
	lsf = aud.lpc_to_lsf(a)
	lsf = lsf[(MARGIN)*int(1/OVERLAP):(SAMPLE_LEN+MARGIN)*int(1/OVERLAP),:]
	lsf_concat = np.concatenate((lsf[::2,:],lsf[1::2,:]),axis=1) # MAGIC NUMBERS for half overlap
	g = g[(MARGIN)*int(1/OVERLAP):(SAMPLE_LEN+MARGIN)*int(1/OVERLAP),:]		
	g_concat = np.concatenate((g[::2,:],g[1::2,:]),axis=1) # MAGIC NUMBERS for half overlap
	feat = np.concatenate((lsf_concat,g_concat),axis=1)
	auddata[audctr:audctr+SAMPLE_LEN,:] = feat
	audctr = audctr+SAMPLE_LEN
	return audctr

def show_progress(progress, step):
	progress += step
	sys.stdout.write("Processing progress: %d%%	\r"%(int(progress)))
	sys.stdout.flush()
	return progress

def main():
	viddata_path = join(datapath,VIDDATA_PATH)
	auddata_path = join(datapath,AUDDATA_PATH)
	if isfile(viddata_path) and isfile(auddata_path):
		print ('Data has already been processed and saved in %s'%(datapath))
	else:
		print ('Processing video and audio data...')
		viddata = np.zeros((MAX_FRAMES,CHANNELS,FRAME_ROWS,FRAME_COLS),dtype="uint8")
		auddata = np.zeros((MAX_FRAMES,NET_OUT),dtype="float32")
		vidfiles = [f for f in listdir(datapath) if isfile(join(datapath, f)) and f.endswith(".mpg")]
		faceCascade = cv2.CascadeClassifier(CASC_PATH)
		vidctr = 0
		audctr = 0
		progress = 0
		progress_step = 100./len(vidfiles)
		progress = show_progress(progress, 0)
		for vf in vidfiles:
			vidctr = process_video(join(datapath,vf), viddata, vidctr, faceCascade)
			audctr = process_audio(join(datapath,vf.replace('.mpg','.wav')), auddata, audctr)
			progress = show_progress(progress, progress_step)
		progress = show_progress(progress, progress_step)
		assert vidctr==audctr
		print ('Done processing. Saving data to disk...')
		viddata = viddata[:vidctr,:,:,:]
		auddata = auddata[:audctr,:]
		np.save(viddata_path, viddata)
		np.save(auddata_path, auddata)
		print ('Done')
	
if __name__ == "__main__":
	main()