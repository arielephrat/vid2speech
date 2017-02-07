from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import keras.backend as K
import numpy as np
import sys
from os.path import isfile, join
import shutil
import scipy.io, scipy.stats
import librosa.display, librosa.output
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import h5py
import os.path
import glob
import audio_tools as aud

FRAME_ROWS = 128
FRAME_COLS = 128
NFRAMES = 9
MARGIN = NFRAMES/2
COLORS = 1
CHANNELS = COLORS*NFRAMES
TRAIN_PER = 0.8
LR = 0.01
nb_pool = 2
BATCH_SIZE = 32
DROPOUT = 0.25 
DROPOUT2 = 0.5
EPOCHS = 20
FINETUNE_EPOCHS = 5
activation_func2 = 'tanh'

respath = '../results/'
weight_path = join(respath,'weights/')
datapath = '../dataset/'

def load_data(datapath):
	viddata_path = join(datapath,'viddata.npy')
	auddata_path = join(datapath,'auddata.npy')
	if isfile(viddata_path) and isfile(auddata_path):
		print ('Loading data...')
		viddata = np.load(viddata_path)
		auddata = np.load(auddata_path)
		vidctr = len(auddata)
		print ('Done.')
	else:
		print ('Preprocessed data not found.')
		sys.exit()

	Xtr = viddata[:int(vidctr*TRAIN_PER),:,:,:]
	Ytr = auddata[:int(vidctr*TRAIN_PER),:]
	Xte = viddata[int(vidctr*TRAIN_PER):,:,:,:]
	Yte = auddata[int(vidctr*TRAIN_PER):,:]
	return (Xtr, Ytr), (Xte, Yte)

def build_model(net_out):
	model = Sequential()
	model.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal', input_shape=(CHANNELS, FRAME_ROWS, FRAME_COLS)))
	model.add(BatchNormalization())
	model.add(LeakyReLU())	
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
	model.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal'))
	model.add(BatchNormalization())
	model.add(LeakyReLU())	
	model.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal'))
	model.add(BatchNormalization())
	model.add(LeakyReLU())	
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
	model.add(Dropout(DROPOUT))
	model.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal'))
	model.add(BatchNormalization())
	model.add(LeakyReLU())	
	model.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal'))
	model.add(BatchNormalization())
	model.add(LeakyReLU())	
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
	model.add(Dropout(DROPOUT))
	model.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal'))
	model.add(BatchNormalization())
	model.add(LeakyReLU())	
	model.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal'))
	model.add(BatchNormalization())
	model.add(LeakyReLU())	
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
	model.add(Dropout(DROPOUT))
	model.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal'))
	model.add(BatchNormalization())
	model.add(LeakyReLU())	
	model.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal'))
	model.add(BatchNormalization())
	model.add(Activation(activation_func2))	
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
	model.add(Dropout(DROPOUT2))
	model.add(Flatten())
	model.add(Dense(512, init='he_normal'))
	model.add(BatchNormalization())
	model.add(Activation(activation_func2))	
	model.add(Dropout(DROPOUT2))
	model.add(Dense(512, init='he_normal'))
	model.add(BatchNormalization())
	model.add(Dense(net_out))
	return model

def savedata(Ytr, Ytr_pred, Yte, Yte_pred, respath=respath):
	np.save(join(respath,'Ytr.npy'),Ytr)
	np.save(join(respath,'Ytr_pred.npy'),Ytr_pred)
	np.save(join(respath,'Yte.npy'),Yte)
	np.save(join(respath,'Yte_pred.npy'),Yte_pred)

def standardize_data(Xtr, Ytr, Xte, Yte):
	Xtr = Xtr.astype('float32')
	Xte = Xte.astype('float32')
	Xtr /= 255
	Xte /= 255
	xtrain_mean = np.mean(Xtr)
	Xtr = Xtr-xtrain_mean
	Xte = Xte-xtrain_mean
	Y_means = np.mean(Ytr,axis=0) 
	Y_stds = np.std(Ytr, axis=0)
	Ytr_norm = ((Ytr-Y_means)/Y_stds)
	Yte_norm = ((Yte-Y_means)/Y_stds)
	return Xtr, Ytr_norm, Xte, Yte_norm, Y_means, Y_stds

def train_net(model, Xtr, Ytr_norm, Xte, Yte_norm, batch_size=BATCH_SIZE, epochs=EPOCHS, finetune=False):
	if finetune:
		newest = max(glob.iglob(weight_path+'*.hdf5'), key=os.path.getctime)
		model.load_weights(newest)
		lr = LR/10
	else:
		lr = LR
	adam = Adam(lr=lr)
	model.compile(loss='mean_squared_error', optimizer=adam)
	checkpointer = ModelCheckpoint(filepath=weight_path+'weights.{epoch:02d}-{val_loss:.4f}.hdf5',
		monitor='val_loss', verbose=1, save_best_only=True)
	history = model.fit(Xtr, Ytr_norm, batch_size=batch_size, nb_epoch=epochs,
		verbose=1, validation_data=(Xte, Yte_norm),callbacks=[checkpointer])
	newest = max(glob.iglob(weight_path+'*.hdf5'), key=os.path.getctime)
	model.load_weights(newest)
	return model

def predict(model, Xtr, Xte, Y_means, Y_stds, batch_size=BATCH_SIZE):
	Ytr_pred = model.predict(Xtr, batch_size=batch_size, verbose=1)
	Yte_pred = model.predict(Xte, batch_size=batch_size, verbose=1)
	Ytr_pred = (Ytr_pred*Y_stds+Y_means)
	Yte_pred = (Yte_pred*Y_stds+Y_means)
	return Ytr_pred, Yte_pred

def main():
	if not os.path.exists(weight_path):
		os.makedirs(weight_path)
	(Xtr,Ytr), (Xte, Yte) = load_data(datapath)
	net_out = Ytr.shape[1]
	Xtr, Ytr_norm, Xte, Yte_norm, Y_means, Y_stds = standardize_data(Xtr, Ytr, Xte, Yte)
	model = build_model(net_out)
	model = train_net(model, Xtr, Ytr_norm, Xte, Yte_norm)
	model = train_net(model, Xtr, Ytr_norm, Xte, Yte_norm, epochs=FINETUNE_EPOCHS, finetune=True)
	Ytr_pred, Yte_pred = predict(model, Xtr, Xte, Y_means, Y_stds)
	savedata(Ytr, Ytr_pred, Yte, Yte_pred)

if __name__ == "__main__":
	main()
