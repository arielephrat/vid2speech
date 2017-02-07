# vid2speech

This is the code for the paper  
**[Vid2speech: Speech Reconstruction from Silent Video](http://www.cs.huji.ac.il/~peleg/papers/arXiv1701.00495-Vid2Speech.pdf)**
<br>
[Ariel Ephrat](http://www.cs.huji.ac.il/~arielephrat/) and
[Shmuel Peleg](http://www.cs.huji.ac.il/~peleg/)
<br>
to appear at [ICASSP 2017](http://www.ieee-icassp2017.org/)

If you find this code useful for your research, please cite

```
@inproceedings{ephrat2017vid2speech,
  title     = {Vid2Speech: speech reconstruction from silent video},
  author    = {Ariel Ephrat and Shmuel Peleg},
  booktitle = {2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2017},
}
```

## Requirements
The code depends on keras, h5py, numpy and librosa, all of which can be easily installed using pip:
```shell
pip install keras h5py numpy librosa
```  
Keras was used with the TensorFlow backend. 

## Prepare the dataset
Download one speaker's videos from the [GRID Corpus](http://spandh.dcs.shef.ac.uk/gridcorpus/), and save the videos directly in the ```/dataset``` folder.  
This code has been tested on the high quality videos of speakers 2 (male) and 4 (female).

Next, strip the audio part of each video and save as the same filename with extension ```.mpg``` replaced with ```.wav```.  
The supplied ```strip_audio.sh``` script can be used:
```shell
cd dataset
sh strip_audio.sh
```

## Preprocess data and train the network
```shell
cd ../code
python process_data.py
python train.py
```
Training one entire GRID speaker (1000 videos) with the supplied settings takes ~12 hours on one Titan Black GPU.

## Generate video samples with reconstructed audio
```shell
python gen_samples.py
```
Samples will appear in ```../results/samples/```

## Use pre-trained model to predict and generate samples
```shell
python predict.py --weight_path <path_to_weights>
python gen_samples.py --respath '../pretrained_results'
```
Weights for a pre-trained model of speaker 2 are supplied in
```
pretrained_weights/s2.hdf5
```

#### Please be in touch with arielephrat@cs.huji.ac.il with any questions or bug reports. Enjoy!