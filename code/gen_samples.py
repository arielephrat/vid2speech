import numpy as np
import os
from os import listdir
from os.path import isfile, join
import sys
import audio_tools as aud
import moviepy.editor as mpy
import argparse
from scipy.io.wavfile import write

VIDPATH = '../dataset/'
RESPATH = '../results/'
SR = 8000
FPS = 25
SPF = int(SR/FPS)
NFRAMES = 9
MARGIN = NFRAMES/2
OVERLAP = 1.0/2
LPC_ORDER = 8
NET_OUT = int(1/OVERLAP)*(LPC_ORDER+1)
SEQ_LEN = 75
SAMPLE_LEN = SEQ_LEN-2*MARGIN
STEP = int(SAMPLE_LEN*1/OVERLAP)
TRAIN_PER = 0.8
N_SAMPLES = 5

def get_args():
    parser = argparse.ArgumentParser(description="Generate samples of reconstructed speech from test set")
    parser.add_argument(
        '--respath',
        type=str,
        default=RESPATH,
        help='path to saved network predictions'
    )
    args = parser.parse_args()
    return args

def get_lsf(Y_pred):
    lsf_tepr = Y_pred[:,:-2]
    lsf_tepr2 = np.zeros((lsf_tepr.shape[0]*2,lsf_tepr.shape[1]/2))
    lsf_tepr2[::2,:] = lsf_tepr[:,:LPC_ORDER]
    lsf_tepr2[1::2,:] = lsf_tepr[:,LPC_ORDER:]
    g_tepr = Y_pred[:,-2:]
    g_tepr2 = np.zeros((g_tepr.shape[0]*2,1))
    g_tepr2[::2,:] = g_tepr[:,:1]
    g_tepr2[1::2,:] = g_tepr[:,1:]
    g_tepr2[g_tepr2<0] = 0.0
    return lsf_tepr2, g_tepr2

def synthesize(lpc,g):
    x = aud.lpc_synthesis(lpc,g,None,window_step=SPF/2)
    return x

def main(): 
    args = get_args()   
    respath = args.respath
    sample_path = join(respath,'samples/')
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    vidfiles = [f for f in listdir(VIDPATH) if isfile(join(VIDPATH, f)) and f.endswith(".mpg")]
    if isfile(join(respath,'Yte_pred.npy')):
        Y_pred = np.load(join(respath,'Yte_pred.npy'))
    elif isfile(join(respath,'aud_pred.npy')):
        Y_pred = np.load(join(respath,'aud_pred.npy'))
    else:
        print ('No results found.')
        sys.exit()
    lsf,g = get_lsf(Y_pred)
    # Choose N_SAMPLES random test videos to reconstruct
    if respath==RESPATH:
        n_test = int((1-TRAIN_PER)*len(vidfiles))
        n_samples = min(n_test-1, N_SAMPLES)
        offset = int(TRAIN_PER*len(vidfiles))
    else:
        n_test = len(vidfiles)
        n_samples = min(N_SAMPLES, len(vidfiles)-1)
        offset = 0
    ind = np.random.choice(n_test,n_samples,replace=False)
    for i in ind:
        vf = VIDPATH+vidfiles[i+offset]
        clip = mpy.VideoFileClip(vf)
        lpc_i = aud.lsf_to_lpc(lsf[i*STEP:(i+1)*STEP,:])
        g_i = g[i*STEP:(i+1)*STEP,:]
        x = synthesize(lpc_i,g_i)
        diff = 24000-len(x)
        pad = np.zeros((diff/2))
        x = np.hstack((np.hstack((pad,x)),pad))
        name = vidfiles[i+offset].replace('.mpg','_pred')
        write(sample_path+name+'.wav',SR,x)
        clip.write_videofile(sample_path+name+'.avi',
                         codec='mpeg4',
                         audio =sample_path+name+'.wav',
                         audio_fps=SR)

if __name__ == "__main__":
    main()

