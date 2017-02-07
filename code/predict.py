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
    parser.add_argument('--weight_path',
        type=str,
        default=None,
        help='path to saved pretrained model weights')
    args = parser.parse_args()
    return args

def main():
	args = get_args()
	weight_path = args.weight_path
	if not os.path.exists(RESPATH):
		os.makedirs(RESPATH)
	(Xtr,Ytr), (Xte, Yte) = train.load_data(DATAPATH)
	net_out = Ytr.shape[1]
	Xtr, Ytr_norm, Xte, Yte_norm, Y_means, Y_stds = train.standardize_data(Xtr, Ytr, Xte, Yte)
	model = train.build_model(net_out)
	model.compile(loss='mse', optimizer='adam')
	model.load_weights(weight_path)
	Ytr_pred, Yte_pred = train.predict(model, Xtr, Xte, Y_means, Y_stds)
	train.savedata(Ytr, Ytr_pred, Yte, Yte_pred, respath = RESPATH)

if __name__ == "__main__":
	main()
