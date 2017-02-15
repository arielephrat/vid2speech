import keras
import numpy as np
import sys
from os.path import isfile, join
import h5py
import os.path
import train
import argparse

RESPATH = '../pretrained_results/'
DATAPATH = '../dataset/'

def get_args():
    parser = argparse.ArgumentParser(description="Use pretrained model to get predictions for data in ../dataset/")
    parser.add_argument(
    	'--weight_path',
        type=str,
        default=None,
        help='path to saved pretrained model weights'
    )
    args = parser.parse_args()
    return args

def standardize_data(viddata, auddata):
	viddata = viddata.astype('float32')
	viddata /= 255
	viddata_mean = np.mean(viddata)
	viddata = viddata-viddata_mean
	auddata_means = np.mean(auddata,axis=0) 
	auddata_stds = np.std(auddata, axis=0)
	auddata_norm = ((auddata-auddata_means)/auddata_stds)
	return viddata, auddata_norm, auddata_means, auddata_stds

def main():
	args = get_args()
	weight_path = args.weight_path
	if not os.path.exists(RESPATH):
		os.makedirs(RESPATH)
	viddata, auddata = train.load_data(DATAPATH)
	net_out = auddata.shape[1]
	viddata, auddata_norm, auddata_means, auddata_stds = standardize_data(viddata, auddata)
	model = train.build_model(net_out)
	model.compile(loss='mse', optimizer='adam')
	model.load_weights(weight_path)
	aud_pred = train.predict(model, viddata, auddata_means, auddata_stds)
	np.save(join(RESPATH,'aud_pred.npy'),aud_pred)

if __name__ == "__main__":
	main()
