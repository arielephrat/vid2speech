# License: BSD 3-clause
# Authors: Kyle Kastner

import numpy as np
# Do numpy version check for  <= 1.9 ! Something crazy going on in 1.10
#if int(np.__version__.split(".")[1]) >= 10:
#   raise ValueError("Only numpy <= 1.9 currently supported! There is a "
#                     "numerical error in one of the numpy 1.10 routines. "
#                     "Hopefully this will be debugged an corrected soon. "
#                     "For the intrepid, the error can be seen by running"
#                     "run_phase_reconstruction()")
from numpy.lib.stride_tricks import as_strided
import scipy.signal as sg
from scipy.cluster.vq import vq
from scipy import linalg, fftpack
from numpy.testing import assert_almost_equal
from scipy.linalg import svd
from scipy.io import wavfile
from scipy.signal import firwin
import zipfile
import tarfile
import os
import copy
try:
    import urllib.request as urllib  # for backwards compatibility
except ImportError:
    import urllib2 as urllib


def download(url, server_fname, local_fname=None, progress_update_percentage=5,
             bypass_certificate_check=False):
    """
    An internet download utility modified from
    http://stackoverflow.com/questions/22676/
    how-do-i-download-a-file-over-http-using-python/22776#22776
    """
    if bypass_certificate_check:
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        u = urllib.urlopen(url, context=ctx)
    else:
        u = urllib.urlopen(url)
    if local_fname is None:
        local_fname = server_fname
    full_path = local_fname
    meta = u.info()
    with open(full_path, 'wb') as f:
        try:
            file_size = int(meta.get("Content-Length"))
        except TypeError:
            print("WARNING: Cannot get file size, displaying bytes instead!")
            file_size = 100
        print("Downloading: %s Bytes: %s" % (server_fname, file_size))
        file_size_dl = 0
        block_sz = int(1E7)
        p = 0
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break
            file_size_dl += len(buffer)
            f.write(buffer)
            if (file_size_dl * 100. / file_size) > p:
                status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl *
                                               100. / file_size)
                print(status)
                p += progress_update_percentage


def fetch_sample_speech_tapestry():
    url = "https://www.dropbox.com/s/qte66a7haqspq2g/tapestry.wav?dl=1"
    wav_path = "tapestry.wav"
    if not os.path.exists(wav_path):
        download(url, wav_path)
    fs, d = wavfile.read(wav_path)
    d = d.astype('float32') / (2 ** 15)
    # file is stereo? - just choose one channel
    return fs, d


def fetch_sample_music():
    url = "http://www.music.helsinki.fi/tmt/opetus/uusmedia/esim/"
    url += "a2002011001-e02-16kHz.wav"
    wav_path = "test.wav"
    if not os.path.exists(wav_path):
        download(url, wav_path)
    fs, d = wavfile.read(wav_path)
    d = d.astype('float32') / (2 ** 15)
    # file is stereo - just choose one channel
    d = d[:, 0]
    return fs, d


def fetch_sample_speech_fruit(n_samples=None):
    url = 'https://dl.dropboxusercontent.com/u/15378192/audio.tar.gz'
    wav_path = "audio.tar.gz"
    if not os.path.exists(wav_path):
        download(url, wav_path)
    tf = tarfile.open(wav_path)
    wav_names = [fname for fname in tf.getnames()
                 if ".wav" in fname.split(os.sep)[-1]]
    speech = []
    print("Loading speech files...")
    for wav_name in wav_names[:n_samples]:
        f = tf.extractfile(wav_name)
        fs, d = wavfile.read(f)
        d = d.astype('float32') / (2 ** 15)
        speech.append(d)
    return fs, speech


def fetch_sample_speech_eustace(n_samples=None):
    """
    http://www.cstr.ed.ac.uk/projects/eustace/download.html
    """
    # data
    url = "http://www.cstr.ed.ac.uk/projects/eustace/down/eustace_wav.zip"
    wav_path = "eustace_wav.zip"
    if not os.path.exists(wav_path):
        download(url, wav_path)

    # labels
    url = "http://www.cstr.ed.ac.uk/projects/eustace/down/eustace_labels.zip"
    labels_path = "eustace_labels.zip"
    if not os.path.exists(labels_path):
        download(url, labels_path)

    # Read wavfiles
    # 16 kHz wav
    zf = zipfile.ZipFile(wav_path, 'r')
    wav_names = [fname for fname in zf.namelist()
                 if ".wav" in fname.split(os.sep)[-1]]
    fs = 16000
    speech = []
    print("Loading speech files...")
    for wav_name in wav_names[:n_samples]:
        wav_str = zf.read(wav_name)
        d = np.frombuffer(wav_str, dtype=np.int16)
        d = d.astype('float32') / (2 ** 15)
        speech.append(d)

    zf = zipfile.ZipFile(labels_path, 'r')
    label_names = [fname for fname in zf.namelist()
                   if ".lab" in fname.split(os.sep)[-1]]
    labels = []
    print("Loading label files...")
    for label_name in label_names[:n_samples]:
        label_file_str = zf.read(label_name)
        labels.append(label_file_str)
    return fs, speech


def stft(X, fftsize=128, step="half", mean_normalize=True, real=False,
         compute_onesided=True):
    """
    Compute STFT for 1D real valued input X
    """
    if real:
        local_fft = np.fft.rfft
        cut = -1
    else:
        local_fft = np.fft.fft
        cut = None
    if compute_onesided:
        cut = fftsize // 2
    if mean_normalize:
        X -= X.mean()
    if step == "half":
        X = halfoverlap(X, fftsize)
    else:
        X = overlap(X, fftsize, step)
    size = fftsize
    win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))
    X = X * win[None]
    X = local_fft(X)[:, :cut]
    return X


def istft(X, fftsize=128, mean_normalize=True, real=False,
          compute_onesided=True):
    """
    Compute ISTFT for STFT transformed X
    """
    if real:
        local_ifft = np.fft.irfft
        X_pad = np.zeros((X.shape[0], X.shape[1] + 1)) + 0j
        X_pad[:, :-1] = X
        X = X_pad
    else:
        local_ifft = np.fft.ifft
    if compute_onesided:
        X_pad = np.zeros((X.shape[0], 2 * X.shape[1])) + 0j
        X_pad[:, :fftsize // 2] = X
        X_pad[:, fftsize // 2:] = 0
        X = X_pad
    X = local_ifft(X).astype("float64")
    X = invert_halfoverlap(X)
    if mean_normalize:
        X -= np.mean(X)
    return X


def mdct_slow(X, dctsize=128):
    M = dctsize
    N = 2 * dctsize
    N_0 = (M + 1) / 2
    X = halfoverlap(X, N)
    X = sine_window(X)
    n, k = np.meshgrid(np.arange(N), np.arange(M))
    # Use transpose due to "samples as rows" convention
    tf = np.cos(np.pi * (n + N_0) * (k + 0.5) / M).T
    return np.dot(X, tf)


def imdct_slow(X, dctsize=128):
    M = dctsize
    N = 2 * dctsize
    N_0 = (M + 1) / 2
    N_4 = N / 4
    n, k = np.meshgrid(np.arange(N), np.arange(M))
    # inverse *is not* transposed
    tf = np.cos(np.pi * (n + N_0) * (k + 0.5) / M)
    X_r = np.dot(X, tf) / N_4
    X_r = sine_window(X_r)
    X = invert_halfoverlap(X_r)
    return X


def rolling_mean(X, window_size):
    w = 1.0 / window_size * np.ones((window_size))
    return np.correlate(X, w, 'valid')


def voiced_unvoiced(X, window_size=256, window_step=128, copy=True):
    """
    Voiced unvoiced detection from a raw signal

    Based on code from:
        https://www.clear.rice.edu/elec532/PROJECTS96/lpc/code.html

    Other references:
        http://www.seas.ucla.edu/spapl/code/harmfreq_MOLRT_VAD.m

    Parameters
    ----------
    X : ndarray
        Raw input signal

    window_size : int, optional (default=256)
        The window size to use, in samples.

    window_step : int, optional (default=128)
        How far the window steps after each calculation, in samples.

    copy : bool, optional (default=True)
        Whether to make a copy of the input array or allow in place changes.
    """
    X = np.array(X, copy=copy)
    if len(X.shape) < 2:
        X = X[None]
    n_points = X.shape[1]
    n_windows = n_points // window_step
    # Padding
    pad_sizes = [(window_size - window_step) // 2,
                 window_size - window_step // 2]
    # TODO: Handling for odd window sizes / steps
    X = np.hstack((np.zeros((X.shape[0], pad_sizes[0])), X,
                   np.zeros((X.shape[0], pad_sizes[1]))))

    clipping_factor = 0.6
    b, a = sg.butter(10, np.pi * 9 / 40)
    voiced_unvoiced = np.zeros((n_windows, 1))
    period = np.zeros((n_windows, 1))
    for window in range(max(n_windows - 1, 1)):
        XX = X.ravel()[window * window_step + np.arange(window_size)]
        XX *= sg.hamming(len(XX))
        XX = sg.lfilter(b, a, XX)
        left_max = np.max(np.abs(XX[:len(XX) // 3]))
        right_max = np.max(np.abs(XX[-len(XX) // 3:]))
        clip_value = clipping_factor * np.min([left_max, right_max])
        XX_clip = np.clip(XX, clip_value, -clip_value)
        XX_corr = np.correlate(XX_clip, XX_clip, mode='full')
        center = np.argmax(XX_corr)
        right_XX_corr = XX_corr[center:]
        prev_window = max([window - 1, 0])
        if voiced_unvoiced[prev_window] > 0:
            # Want it to be harder to turn off than turn on
            strength_factor = .29
        else:
            strength_factor = .3
        start = np.where(right_XX_corr < .3 * XX_corr[center])[0]
        # 20 is hardcoded but should depend on samplerate?
        start = np.max([20, start[0]])
        search_corr = right_XX_corr[start:]
        index = np.argmax(search_corr)
        second_max = search_corr[index]
        if (second_max > strength_factor * XX_corr[center]):
            voiced_unvoiced[window] = 1
            period[window] = start + index - 1
        else:
            voiced_unvoiced[window] = 0
            period[window] = 0
    return np.array(voiced_unvoiced), np.array(period)


def lpc_analysis(X, order=8, window_step=128, window_size=2 * 128,
                 emphasis=0.9, voiced_start_threshold=.9,
                 voiced_stop_threshold=.6, truncate=False, copy=True):
    """
    Extract LPC coefficients from a signal

    Based on code from:
        http://labrosa.ee.columbia.edu/matlab/sws/

    Parameters
    ----------
    X : ndarray
        Signals to extract LPC coefficients from

    order : int, optional (default=8)
        Order of the LPC coefficients. For speech, use the general rule that the
        order is two times the expected number of formants plus 2.
        This can be formulated as 2 + 2 * (fs // 2000). For approx. signals
        with fs = 7000, this is 8 coefficients - 2 + 2 * (7000 // 2000).

    window_step : int, optional (default=128)
        The size (in samples) of the space between each window

    window_size : int, optional (default=2 * 128)
        The size of each window (in samples) to extract coefficients over

    emphasis : float, optional (default=0.9)
        The emphasis coefficient to use for filtering

    voiced_start_threshold : float, optional (default=0.9)
        Upper power threshold for estimating when speech has started

    voiced_stop_threshold : float, optional (default=0.6)
        Lower power threshold for estimating when speech has stopped

    truncate : bool, optional (default=False)
        Whether to cut the data at the last window or do zero padding.

    copy : bool, optional (default=True)
        Whether to copy the input X or modify in place

    Returns
    -------
    lp_coefficients : ndarray
        lp coefficients to describe the frame

    per_frame_gain : ndarray
        calculated gain for each frame

    residual_excitation : ndarray
        leftover energy which is not described by lp coefficents and gain

    voiced_frames : ndarray
        array of [0, 1] values which holds voiced/unvoiced decision for each
        frame.

    References
    ----------
    D. P. W. Ellis (2004), "Sinewave Speech Analysis/Synthesis in Matlab",
    Web resource, available: http://www.ee.columbia.edu/ln/labrosa/matlab/sws/
    """
    X = np.array(X, copy=copy)
    if len(X.shape) < 2:
        X = X[None]

    n_points = X.shape[1]
    n_windows = n_points // window_step
    if not truncate:
        pad_sizes = [(window_size - window_step) // 2,
                     window_size - window_step // 2]
        # TODO: Handling for odd window sizes / steps
        X = np.hstack((np.zeros((X.shape[0], pad_sizes[0])), X,
                       np.zeros((X.shape[0], pad_sizes[1]))))
    else:
        pad_sizes = [0, 0]
        X = X[0, :n_windows * window_step]

    lp_coefficients = np.zeros((n_windows, order + 1))
    per_frame_gain = np.zeros((n_windows, 1))
    residual_excitation = np.zeros(
        ((n_windows - 1) * window_step + window_size))
    # Pre-emphasis high-pass filter
    X = sg.lfilter([1, -emphasis], 1, X)
    # stride_tricks.as_strided?
    autocorr_X = np.zeros((n_windows, 2 * window_size - 1))

    for window in range(max(n_windows - 1, 1)):
        XX = X.ravel()[window * window_step + np.arange(window_size)]
        if np.max(XX) > 0:
            WXX = XX * sg.hanning(window_size)
            autocorr_X[window] = np.correlate(WXX, WXX, mode='full')
            center = np.argmax(autocorr_X[window])
            RXX = autocorr_X[window,
                             np.arange(center, window_size + order)]
            R = linalg.toeplitz(RXX[:-1])
            solved_R = linalg.pinv(R).dot(RXX[1:])
            filter_coefs = np.hstack((1, -solved_R))
            residual_signal = sg.lfilter(filter_coefs, 1, WXX)
            gain = np.sqrt(np.mean(residual_signal ** 2))
            lp_coefficients[window] = filter_coefs
            per_frame_gain[window] = gain
            assign_range = window * window_step + np.arange(window_size)
            residual_excitation[assign_range] += residual_signal / gain
    # Throw away first part in overlap mode for proper synthesis
    residual_excitation = residual_excitation[pad_sizes[0]:]
    return lp_coefficients, per_frame_gain, residual_excitation


def lpc_synthesis(lp_coefficients, per_frame_gain, residual_excitation=None,
                  voiced_frames=None, window_step=128, emphasis=0.9):
    """
    Synthesize a signal from LPC coefficients

    Based on code from:
        http://labrosa.ee.columbia.edu/matlab/sws/
        http://web.uvic.ca/~tyoon/resource/auditorytoolbox/auditorytoolbox/synlpc.html

    Parameters
    ----------
    lp_coefficients : ndarray
        Linear prediction coefficients

    per_frame_gain : ndarray
        Gain coefficients

    residual_excitation : ndarray or None, optional (default=None)
        Residual excitations. If None, this will be synthesized with white noise

    voiced_frames : ndarray or None, optional (default=None)
        Voiced frames. If None, all frames assumed to be voiced.

    window_step : int, optional (default=128)
        The size (in samples) of the space between each window

    emphasis : float, optional (default=0.9)
        The emphasis coefficient to use for filtering

    overlap_add : bool, optional (default=True)
        What type of processing to use when joining windows

    copy : bool, optional (default=True)
       Whether to copy the input X or modify in place

    Returns
    -------
    synthesized : ndarray
        Sound vector synthesized from input arguments

    References
    ----------
    D. P. W. Ellis (2004), "Sinewave Speech Analysis/Synthesis in Matlab",
    Web resource, available: http://www.ee.columbia.edu/ln/labrosa/matlab/sws/
    """
    # TODO: Incorporate better synthesis from
    # http://eecs.oregonstate.edu/education/docs/ece352/CompleteManual.pdf
    window_size = 2 * window_step
    [n_windows, order] = lp_coefficients.shape

    n_points = (n_windows + 1) * window_step
    n_excitation_points = n_points + window_step + window_step // 2

    random_state = np.random.RandomState(1999)
    if residual_excitation is None:
        # Need to generate excitation
        if voiced_frames is None:
            # No voiced/unvoiced info, so just use randn
            voiced_frames = np.ones((lp_coefficients.shape[0], 1))
        residual_excitation = np.zeros((n_excitation_points))
        f, m = lpc_to_frequency(lp_coefficients, per_frame_gain)
        t = np.linspace(0, 1, window_size, endpoint=False)
        hanning = sg.hanning(window_size)
        for window in range(n_windows):
            window_base = window * window_step
            index = window_base + np.arange(window_size)
            if voiced_frames[window]:
                sig = np.zeros_like(t)
                cycles = np.cumsum(f[window][0] * t)
                sig += sg.sawtooth(cycles, 0.001)
                residual_excitation[index] += hanning * sig
            residual_excitation[index] += hanning * 0.01 * random_state.randn(
                window_size)
    else:
        n_excitation_points = residual_excitation.shape[0]
        n_points = n_excitation_points + window_step + window_step // 2
    residual_excitation = np.hstack((residual_excitation,
                                     np.zeros(window_size)))
    if voiced_frames is None:
        voiced_frames = np.ones_like(per_frame_gain)

    synthesized = np.zeros((n_points))
    for window in range(n_windows):
        window_base = window * window_step
        oldbit = synthesized[window_base + np.arange(window_step)]
        w_coefs = lp_coefficients[window]
        if not np.all(w_coefs):
            # Hack to make lfilter avoid
            # ValueError: BUG: filter coefficient a[0] == 0 not supported yet
            # when all coeffs are 0
            w_coefs = [1]
        g_coefs = voiced_frames[window] * per_frame_gain[window]
        index = window_base + np.arange(window_size)
        newbit = g_coefs * sg.lfilter([1], w_coefs,
                                      residual_excitation[index])
        synthesized[index] = np.hstack((oldbit, np.zeros(
            (window_size - window_step))))
        synthesized[index] += sg.hanning(window_size) * newbit
    synthesized = sg.lfilter([1], [1, -emphasis], synthesized)
    return synthesized


def soundsc(X, copy=True):
    """
    Approximate implementation of soundsc from MATLAB without the audio playing.

    Parameters
    ----------
    X : ndarray
        Signal to be rescaled

    copy : bool, optional (default=True)
        Whether to make a copy of input signal or operate in place.

    Returns
    -------
    X_sc : ndarray
        (-1, 1) scaled version of X as float32, suitable for writing
        with scipy.io.wavfile
    """
    X = np.array(X, copy=copy)
    X = (X - X.min()) / (X.max() - X.min())
    X = .9 * X
    X = 2 * X - 1
    return X.astype('float32')


def lpc_to_frequency(lp_coefficients, per_frame_gain):
    """
    Extract resonant frequencies and magnitudes from LPC coefficients and gains.
    Parameters
    ----------
    lp_coefficients : ndarray
        LPC coefficients, such as those calculated by ``lpc_analysis``

    per_frame_gain : ndarray
       Gain calculated for each frame, such as those calculated
       by ``lpc_analysis``

    Returns
    -------
    frequencies : ndarray
       Resonant frequencies calculated from LPC coefficients and gain. Returned
       frequencies are from 0 to 2 * pi

    magnitudes : ndarray
       Magnitudes of resonant frequencies

    References
    ----------
    D. P. W. Ellis (2004), "Sinewave Speech Analysis/Synthesis in Matlab",
    Web resource, available: http://www.ee.columbia.edu/ln/labrosa/matlab/sws/
    """
    n_windows, order = lp_coefficients.shape

    frame_frequencies = np.zeros((n_windows, (order - 1) // 2))
    frame_magnitudes = np.zeros_like(frame_frequencies)

    for window in range(n_windows):
        w_coefs = lp_coefficients[window]
        g_coefs = per_frame_gain[window]
        roots = np.roots(np.hstack(([1], w_coefs[1:])))
        # Roots doesn't return the same thing as MATLAB... agh
        frequencies, index = np.unique(
            np.abs(np.angle(roots)), return_index=True)
        # Make sure 0 doesn't show up...
        gtz = np.where(frequencies > 0)[0]
        frequencies = frequencies[gtz]
        index = index[gtz]
        magnitudes = g_coefs / (1. - np.abs(roots))
        sort_index = np.argsort(frequencies)
        frame_frequencies[window, :len(sort_index)] = frequencies[sort_index]
        frame_magnitudes[window, :len(sort_index)] = magnitudes[sort_index]
    return frame_frequencies, frame_magnitudes


def sinusoid_analysis(X, input_sample_rate, resample_block=128, copy=True):
    """
    Contruct a sinusoidal model for the input signal.

    Parameters
    ----------
    X : ndarray
        Input signal to model

    input_sample_rate : int
        The sample rate of the input signal

    resample_block : int, optional (default=128)
       Controls the step size of the sinusoidal model

    Returns
    -------
    frequencies_hz : ndarray
       Frequencies for the sinusoids, in Hz.

    magnitudes : ndarray
       Magnitudes of sinusoids returned in ``frequencies``

    References
    ----------
    D. P. W. Ellis (2004), "Sinewave Speech Analysis/Synthesis in Matlab",
    Web resource, available: http://www.ee.columbia.edu/ln/labrosa/matlab/sws/
    """
    X = np.array(X, copy=copy)
    resample_to = 8000
    if input_sample_rate != resample_to:
        if input_sample_rate % resample_to != 0:
            raise ValueError("Input sample rate must be a multiple of 8k!")
        # Should be able to use resample... ?
        # resampled_count = round(len(X) * resample_to / input_sample_rate)
        # X = sg.resample(X, resampled_count, window=sg.hanning(len(X)))
        X = sg.decimate(X, input_sample_rate // resample_to)
    step_size = 2 * round(resample_block / input_sample_rate * resample_to / 2.)
    a, g, e = lpc_analysis(X, order=8, window_step=step_size,
                           window_size=2 * step_size)
    f, m = lpc_to_frequency(a, g)
    f_hz = f * resample_to / (2 * np.pi)
    return f_hz, m


def slinterp(X, factor, copy=True):
    """
    Slow-ish linear interpolation of a 1D numpy array. There must be some
    better function to do this in numpy.

    Parameters
    ----------
    X : ndarray
        1D input array to interpolate

    factor : int
        Integer factor to interpolate by

    Return
    ------
    X_r : ndarray
    """
    sz = np.product(X.shape)
    X = np.array(X, copy=copy)
    X_s = np.hstack((X[1:], [0]))
    X_r = np.zeros((factor, sz))
    for i in range(factor):
        X_r[i, :] = (factor - i) / float(factor) * X + (i / float(factor)) * X_s
    return X_r.T.ravel()[:(sz - 1) * factor + 1]


def sinusoid_synthesis(frequencies_hz, magnitudes, input_sample_rate=16000,
                       resample_block=128):
    """
    Create a time series based on input frequencies and magnitudes.

    Parameters
    ----------
    frequencies_hz : ndarray
        Input signal to model

    magnitudes : int
        The sample rate of the input signal

    input_sample_rate : int, optional (default=16000)
        The sample rate parameter that the sinusoid analysis was run with

    resample_block : int, optional (default=128)
       Controls the step size of the sinusoidal model

    Returns
    -------
    synthesized : ndarray
        Sound vector synthesized from input arguments

    References
    ----------
    D. P. W. Ellis (2004), "Sinewave Speech Analysis/Synthesis in Matlab",
    Web resource, available: http://www.ee.columbia.edu/ln/labrosa/matlab/sws/
    """
    rows, cols = frequencies_hz.shape
    synthesized = np.zeros((1 + ((rows - 1) * resample_block),))
    for col in range(cols):
        mags = slinterp(magnitudes[:, col], resample_block)
        freqs = slinterp(frequencies_hz[:, col], resample_block)
        cycles = np.cumsum(2 * np.pi * freqs / float(input_sample_rate))
        sines = mags * np.cos(cycles)
        synthesized += sines
    return synthesized


def compress(X, n_components, window_size=128):
    """
    Compress using the DCT

    Parameters
    ----------
    X : ndarray, shape=(n_samples,)
        The input signal to compress. Should be 1-dimensional

    n_components : int
        The number of DCT components to keep. Setting n_components to about
        .5 * window_size can give compression with fairly good reconstruction.

    window_size : int
        The input X is broken into windows of window_size, each of which are
        then compressed with the DCT.

    Returns
    -------
    X_compressed : ndarray, shape=(num_windows, window_size)
       A 2D array of non-overlapping DCT coefficients. For use with uncompress

    Reference
    ---------
    http://nbviewer.ipython.org/github/craffel/crucialpython/blob/master/week3/stride_tricks.ipynb
    """
    if len(X) % window_size != 0:
        append = np.zeros((window_size - len(X) % window_size))
        X = np.hstack((X, append))
    num_frames = len(X) // window_size
    X_strided = X.reshape((num_frames, window_size))
    X_dct = fftpack.dct(X_strided, norm='ortho')
    if n_components is not None:
        X_dct = X_dct[:, :n_components]
    return X_dct


def uncompress(X_compressed, window_size=128):
    """
    Uncompress a DCT compressed signal (such as returned by ``compress``).

    Parameters
    ----------
    X_compressed : ndarray, shape=(n_samples, n_features)
        Windowed and compressed array.

    window_size : int, optional (default=128)
        Size of the window used when ``compress`` was called.

    Returns
    -------
    X_reconstructed : ndarray, shape=(n_samples)
        Reconstructed version of X.
    """
    if X_compressed.shape[1] % window_size != 0:
        append = np.zeros((X_compressed.shape[0],
                           window_size - X_compressed.shape[1] % window_size))
        X_compressed = np.hstack((X_compressed, append))
    X_r = fftpack.idct(X_compressed, norm='ortho')
    return X_r.ravel()


def sine_window(X):
    """
    Apply a sinusoid window to X.

    Parameters
    ----------
    X : ndarray, shape=(n_samples, n_features)
        Input array of samples

    Returns
    -------
    X_windowed : ndarray, shape=(n_samples, n_features)
        Windowed version of X.
    """
    i = np.arange(X.shape[1])
    win = np.sin(np.pi * (i + 0.5) / X.shape[1])
    row_stride = 0
    col_stride = win.itemsize
    strided_win = as_strided(win, shape=X.shape,
                             strides=(row_stride, col_stride))
    return X * strided_win


def kaiserbessel_window(X, alpha=6.5):
    """
    Apply a Kaiser-Bessel window to X.

    Parameters
    ----------
    X : ndarray, shape=(n_samples, n_features)
        Input array of samples

    alpha : float, optional (default=6.5)
        Tuning parameter for Kaiser-Bessel function. alpha=6.5 should make
        perfect reconstruction possible for DCT.

    Returns
    -------
    X_windowed : ndarray, shape=(n_samples, n_features)
        Windowed version of X.
    """
    beta = np.pi * alpha
    win = sg.kaiser(X.shape[1], beta)
    row_stride = 0
    col_stride = win.itemsize
    strided_win = as_strided(win, shape=X.shape,
                             strides=(row_stride, col_stride))
    return X * strided_win


def overlap(X, window_size, window_step):
    """
    Create an overlapped version of X

    Parameters
    ----------
    X : ndarray, shape=(n_samples,)
        Input signal to window and overlap

    window_size : int
        Size of windows to take

    window_step : int
        Step size between windows

    Returns
    -------
    X_strided : shape=(n_windows, window_size)
        2D array of overlapped X
    """
    if window_size % 2 != 0:
        raise ValueError("Window size must be even!")
    # Make sure there are an even number of windows before stridetricks
    append = np.zeros((window_size - len(X) % window_size))
    X = np.hstack((X, append))
    num_frames = len(X) // window_step - 1
    row_stride = X.itemsize * window_step
    col_stride = X.itemsize
    X_strided = as_strided(X, shape=(num_frames, window_size),
                           strides=(row_stride, col_stride))
    return X_strided


def halfoverlap(X, window_size):
    """
    Create an overlapped version of X using 50% of window_size as overlap.

    Parameters
    ----------
    X : ndarray, shape=(n_samples,)
        Input signal to window and overlap

    window_size : int
        Size of windows to take

    Returns
    -------
    X_strided : shape=(n_windows, window_size)
        2D array of overlapped X
    """
    if window_size % 2 != 0:
        raise ValueError("Window size must be even!")
    window_step = window_size // 2
    # Make sure there are an even number of windows before stridetricks
    append = np.zeros((window_size - len(X) % window_size))
    X = np.hstack((X, append))
    num_frames = len(X) // window_step - 1
    row_stride = X.itemsize * window_step
    col_stride = X.itemsize
    X_strided = as_strided(X, shape=(num_frames, window_size),
                           strides=(row_stride, col_stride))
    return X_strided


def invert_halfoverlap(X_strided):
    """
    Invert ``halfoverlap`` function to reconstruct X

    Parameters
    ----------
    X_strided : ndarray, shape=(n_windows, window_size)
        X as overlapped windows

    Returns
    -------
    X : ndarray, shape=(n_samples,)
        Reconstructed version of X
    """
    # Hardcoded 50% overlap! Can generalize later...
    n_rows, n_cols = X_strided.shape
    X = np.zeros((((int(n_rows // 2) + 1) * n_cols),)).astype(X_strided.dtype)
    start_index = 0
    end_index = n_cols
    window_step = n_cols // 2
    for row in range(X_strided.shape[0]):
        X[start_index:end_index] += X_strided[row]
        start_index += window_step
        end_index += window_step
    return X


def csvd(arr):
    """
    Do the complex SVD of a 2D array, returning real valued U, S, VT

    http://stemblab.github.io/complex-svd/
    """
    C_r = arr.real
    C_i = arr.imag
    block_x = C_r.shape[0]
    block_y = C_r.shape[1]
    K = np.zeros((2 * block_x, 2 * block_y))
    # Upper left
    K[:block_x, :block_y] = C_r
    # Lower left
    K[:block_x, block_y:] = C_i
    # Upper right
    K[block_x:, :block_y] = -C_i
    # Lower right
    K[block_x:, block_y:] = C_r
    return svd(K, full_matrices=False)


def icsvd(U, S, VT):
    """
    Invert back to complex values from the output of csvd

    U, S, VT = csvd(X)
    X_rec = inv_csvd(U, S, VT)
    """
    K = U.dot(np.diag(S)).dot(VT)
    block_x = U.shape[0] // 2
    block_y = U.shape[1] // 2
    arr_rec = np.zeros((block_x, block_y)) + 0j
    arr_rec.real = K[:block_x, :block_y]
    arr_rec.imag = K[:block_x, block_y:]
    return arr_rec


def overlap_compress(X, n_components, window_size):
    """
    Overlap (at 50% of window_size) and compress X.

    Parameters
    ----------
    X : ndarray, shape=(n_samples,)
        Input signal to compress

    n_components : int
        number of DCT components to keep

    window_size : int
        Size of windows to take

    Returns
    -------
    X_dct : ndarray, shape=(n_windows, n_components)
        Windowed and compressed version of X
    """
    X_strided = halfoverlap(X, window_size)
    X_dct = fftpack.dct(X_strided, norm='ortho')
    if n_components is not None:
        X_dct = X_dct[:, :n_components]
    return X_dct


# Evil voice is caused by adding double the zeros before inverse DCT...
# Very cool bug but makes sense
def overlap_uncompress(X_compressed, window_size):
    """
    Uncompress X as returned from ``overlap_compress``.

    Parameters
    ----------
    X_compressed : ndarray, shape=(n_windows, n_components)
        Windowed and compressed version of X

    window_size : int
        Size of windows originally used when compressing X

    Returns
    -------
    X_reconstructed : ndarray, shape=(n_samples,)
        Reconstructed version of X
    """
    if X_compressed.shape[1] % window_size != 0:
        append = np.zeros((X_compressed.shape[0], window_size -
                           X_compressed.shape[1] % window_size))
        X_compressed = np.hstack((X_compressed, append))
    X_r = fftpack.idct(X_compressed, norm='ortho')
    return invert_halfoverlap(X_r)


def lpc_to_lsf(all_lpc):
    if len(all_lpc.shape) < 2:
        all_lpc = all_lpc[None]
    order = all_lpc.shape[1] - 1
    all_lsf = np.zeros((len(all_lpc), order))
    for i in range(len(all_lpc)):
        lpc = all_lpc[i]
        lpc1 = np.append(lpc, 0)
        lpc2 = lpc1[::-1]
        sum_filt = lpc1 + lpc2
        diff_filt = lpc1 - lpc2

        if order % 2 != 0:
            deconv_diff, _ = sg.deconvolve(diff_filt, [1, 0, -1])
            deconv_sum = sum_filt
        else:
            deconv_diff, _ = sg.deconvolve(diff_filt, [1, -1])
            deconv_sum, _ = sg.deconvolve(sum_filt, [1, 1])

        roots_diff = np.roots(deconv_diff)
        roots_sum = np.roots(deconv_sum)
        angle_diff = np.angle(roots_diff[::2])
        angle_sum = np.angle(roots_sum[::2])
        lsf = np.sort(np.hstack((angle_diff, angle_sum)))
        if len(lsf) != 0:
            all_lsf[i] = lsf
    return np.squeeze(all_lsf)


def lsf_to_lpc(all_lsf):
    if len(all_lsf.shape) < 2:
        all_lsf = all_lsf[None]
    order = all_lsf.shape[1]
    all_lpc = np.zeros((len(all_lsf), order + 1))
    for i in range(len(all_lsf)):
        lsf = all_lsf[i]
        zeros = np.exp(1j * lsf)
        sum_zeros = zeros[::2]
        diff_zeros = zeros[1::2]
        sum_zeros = np.hstack((sum_zeros, np.conj(sum_zeros)))
        diff_zeros = np.hstack((diff_zeros, np.conj(diff_zeros)))
        sum_filt = np.poly(sum_zeros)
        diff_filt = np.poly(diff_zeros)

        if order % 2 != 0:
            deconv_diff = sg.convolve(diff_filt, [1, 0, -1])
            deconv_sum = sum_filt
        else:
            deconv_diff = sg.convolve(diff_filt, [1, -1])
            deconv_sum = sg.convolve(sum_filt, [1, 1])

        lpc = .5 * (deconv_sum + deconv_diff)
        # Last coefficient is 0 and not returned
        all_lpc[i] = lpc[:-1]
    return np.squeeze(all_lpc)


def herz_to_mel(freqs):
    """
    Based on code by Dan Ellis

    http://labrosa.ee.columbia.edu/matlab/tf_agc/
    """
    f_0 = 0  # 133.33333
    f_sp = 200 / 3.  # 66.66667
    bark_freq = 1000.
    bark_pt = (bark_freq - f_0) / f_sp
    # The magic 1.0711703 which is the ratio needed to get from 1000 Hz
    # to 6400 Hz in 27 steps, and is *almost* the ratio between 1000 Hz
    # and the preceding linear filter center at 933.33333 Hz
    # (actually 1000/933.33333 = 1.07142857142857 and
    # exp(log(6.4)/27) = 1.07117028749447)
    if not isinstance(freqs, np.ndarray):
        freqs = np.array(freqs)[None]
    log_step = np.exp(np.log(6.4) / 27)
    lin_pts = (freqs < bark_freq)
    mel = 0. * freqs
    mel[lin_pts] = (freqs[lin_pts] - f_0) / f_sp
    mel[~lin_pts] = bark_pt + np.log(freqs[~lin_pts] / bark_freq) / np.log(
        log_step)
    return mel


def mel_to_herz(mel):
    """
    Based on code by Dan Ellis

    http://labrosa.ee.columbia.edu/matlab/tf_agc/
    """
    f_0 = 0  # 133.33333
    f_sp = 200 / 3.  # 66.66667
    bark_freq = 1000.
    bark_pt = (bark_freq - f_0) / f_sp
    # The magic 1.0711703 which is the ratio needed to get from 1000 Hz
    # to 6400 Hz in 27 steps, and is *almost* the ratio between 1000 Hz
    # and the preceding linear filter center at 933.33333 Hz
    # (actually 1000/933.33333 = 1.07142857142857 and
    # exp(log(6.4)/27) = 1.07117028749447)
    if not isinstance(mel, np.ndarray):
        mel = np.array(mel)[None]
    log_step = np.exp(np.log(6.4) / 27)
    lin_pts = (mel < bark_pt)

    freqs = 0. * mel
    freqs[lin_pts] = f_0 + f_sp * mel[lin_pts]
    freqs[~lin_pts] = bark_freq * np.exp(np.log(log_step) * (
        mel[~lin_pts] - bark_pt))
    return freqs


def mel_freq_weights(n_fft, fs, n_filts=None, width=None):
    """
    Based on code by Dan Ellis

    http://labrosa.ee.columbia.edu/matlab/tf_agc/
    """
    min_freq = 0
    max_freq = fs // 2
    if width is None:
        width = 1.
    if n_filts is None:
        n_filts = int(herz_to_mel(max_freq) / 2) + 1
    else:
        n_filts = int(n_filts)
        assert n_filts > 0
    weights = np.zeros((n_filts, n_fft))
    fft_freqs = np.arange(n_fft // 2) / n_fft * fs
    min_mel = herz_to_mel(min_freq)
    max_mel = herz_to_mel(max_freq)
    partial = np.arange(n_filts + 2) / (n_filts + 1) * (max_mel - min_mel)
    bin_freqs = mel_to_herz(min_mel + partial)
    bin_bin = np.round(bin_freqs / fs * (n_fft - 1))
    for i in range(n_filts):
        fs_i = bin_freqs[i + np.arange(3)]
        fs_i = fs_i[1] + width * (fs_i - fs_i[1])
        lo_slope = (fft_freqs - fs_i[0]) / float(fs_i[1] - fs_i[0])
        hi_slope = (fs_i[2] - fft_freqs) / float(fs_i[2] - fs_i[1])
        weights[i, :n_fft // 2] = np.maximum(
            0, np.minimum(lo_slope, hi_slope))
    # Constant amplitude multiplier
    weights = np.diag(2. / (bin_freqs[2:n_filts + 2]
                      - bin_freqs[:n_filts])).dot(weights)
    weights[:, n_fft // 2:] = 0
    return weights


def time_attack_agc(X, fs, t_scale=0.5, f_scale=1.):
    """
    AGC based on code by Dan Ellis

    http://labrosa.ee.columbia.edu/matlab/tf_agc/
    """
    # 32 ms grid for FFT
    n_fft = 2 ** int(np.log(0.032 * fs) / np.log(2))
    f_scale = float(f_scale)
    window_size = n_fft
    window_step = window_size // 2
    X_freq = stft(X, window_size, mean_normalize=False)
    fft_fs = fs / window_step
    n_bands = max(10, 20 / f_scale)
    mel_width = f_scale * n_bands / 10.
    f_to_a = mel_freq_weights(n_fft, fs, n_bands, mel_width)
    f_to_a = f_to_a[:, :n_fft // 2]
    audiogram = np.abs(X_freq).dot(f_to_a.T)
    fbg = np.zeros_like(audiogram)
    state = np.zeros((audiogram.shape[1],))
    alpha = np.exp(-(1. / fft_fs) / t_scale)
    for i in range(len(audiogram)):
        state = np.maximum(alpha * state, audiogram[i])
        fbg[i] = state

    sf_to_a = np.sum(f_to_a, axis=0)
    E = np.diag(1. / (sf_to_a + (sf_to_a == 0)))
    E = E.dot(f_to_a.T)
    E = fbg.dot(E.T)
    E[E <= 0] = np.min(E[E > 0])
    ts = istft(X_freq / E, window_size, mean_normalize=False)
    return ts, X_freq, E


def hebbian_kmeans(X, n_clusters=10, n_epochs=10, W=None, learning_rate=0.01,
                   batch_size=100, random_state=None, verbose=True):
    """
    Modified from existing code from R. Memisevic
    See http://www.cs.toronto.edu/~rfm/code/hebbian_kmeans.py
    """
    if W is None:
        if random_state is None:
            random_state = np.random.RandomState()
        W = 0.1 * random_state.randn(n_clusters, X.shape[1])
    else:
        assert n_clusters == W.shape[0]
    X2 = (X ** 2).sum(axis=1, keepdims=True)
    last_print = 0
    for e in range(n_epochs):
        for i in range(0, X.shape[0], batch_size):
            X_i = X[i: i + batch_size]
            X2_i = X2[i: i + batch_size]
            D = -2 * np.dot(W, X_i.T)
            D += (W ** 2).sum(axis=1, keepdims=True)
            D += X2_i.T
            S = (D == D.min(axis=0)[None, :]).astype("float").T
            W += learning_rate * (
                np.dot(S.T, X_i) - S.sum(axis=0)[:, None] * W)
        if verbose:
            if e == 0 or e > (.05 * n_epochs + last_print):
                last_print = e
                print("Epoch %i of %i, cost %.4f" % (
                    e + 1, n_epochs, D.min(axis=0).sum()))
    return W


def complex_to_real_view(arr_c):
    # Inplace view from complex to r, i as separate columns
    assert arr_c.dtype in [np.complex64, np.complex128]
    shp = arr_c.shape
    dtype = np.float64 if arr_c.dtype == np.complex128 else np.float32
    arr_r = arr_c.ravel().view(dtype=dtype).reshape(shp[0], 2 * shp[1])
    return arr_r


def real_to_complex_view(arr_r):
    # Inplace view from real, image as columns to complex
    assert arr_r.dtype not in [np.complex64, np.complex128]
    shp = arr_r.shape
    dtype = np.complex128 if arr_r.dtype == np.float64 else np.complex64
    arr_c = arr_r.ravel().view(dtype=dtype).reshape(shp[0], shp[1] // 2)
    return arr_c


def complex_to_abs(arr_c):
    return np.abs(arr_c)


def complex_to_angle(arr_c):
    return np.angle(arr_c)


def abs_and_angle_to_complex(arr_abs, arr_angle):
    # abs(f_c2 - f_c) < 1E-15
    return arr_abs * np.exp(1j * arr_angle)


def polyphase_core(x, m, f):
    # x = input data
    # m = decimation rate
    # f = filter
    # Hack job - append zeros to match decimation rate
    if x.shape[0] % m != 0:
        x = np.append(x, np.zeros((m - x.shape[0] % m,)))
    if f.shape[0] % m != 0:
        f = np.append(f, np.zeros((m - f.shape[0] % m,)))
    polyphase = p = np.zeros((m, (x.shape[0] + f.shape[0]) / m), dtype=x.dtype)
    p[0, :-1] = np.convolve(x[::m], f[::m])
    # Invert the x values when applying filters
    for i in range(1, m):
        p[i, 1:] = np.convolve(x[m - i::m], f[i::m])
    return p


def polyphase_single_filter(x, m, f):
    return np.sum(polyphase_core(x, m, f), axis=0)


def polyphase_lowpass(arr, downsample=2, n_taps=50, filter_pad=1.1):
    filt = firwin(downsample * n_taps, 1 / (downsample * filter_pad))
    filtered = polyphase_single_filter(arr, downsample, filt)
    return filtered


def window(arr, window_size, window_step=1, axis=0):
    """
    Directly taken from Erik Rigtorp's post to numpy-discussion.
    <http://www.mail-archive.com/numpy-discussion@scipy.org/msg29450.html>

    <http://stackoverflow.com/questions/4936620/using-strides-for-an-efficient-moving-average-filter>
    """
    if window_size < 1:
        raise ValueError("`window_size` must be at least 1.")
    if window_size > arr.shape[-1]:
        raise ValueError("`window_size` is too long.")

    orig = list(range(len(arr.shape)))
    trans = list(range(len(arr.shape)))
    trans[axis] = orig[-1]
    trans[-1] = orig[axis]
    arr = arr.transpose(trans)

    shape = arr.shape[:-1] + (arr.shape[-1] - window_size + 1, window_size)
    strides = arr.strides + (arr.strides[-1],)
    strided = as_strided(arr, shape=shape, strides=strides)

    if window_step > 1:
        strided = strided[..., ::window_step, :]

    orig = list(range(len(strided.shape)))
    trans = list(range(len(strided.shape)))
    trans[-2] = orig[-1]
    trans[-1] = orig[-2]
    trans = trans[::-1]
    strided = strided.transpose(trans)
    return strided


def unwindow(arr, window_size, window_step=1, axis=0):
    # undo windows by broadcast
    if axis != 0:
        raise ValueError("axis != 0 currently unsupported")
    shp = arr.shape
    unwindowed = np.tile(arr[:, None, ...], (1, window_step, 1, 1))
    unwindowed = unwindowed.reshape(shp[0] * window_step, *shp[1:])
    return unwindowed.mean(axis=1)


def angle_to_sin_cos(arr_angle):
    return np.hstack((np.sin(arr_angle), np.cos(arr_angle)))


def sin_cos_to_angle(arr_sin, arr_cos):
    return np.arctan2(arr_sin, arr_cos)


def xcorr_offset(x1, x2):
    """
    Under MSR-LA License

    Based on MATLAB implementation from Spectrogram Inversion Toolbox

    References
    ----------
    D. Griffin and J. Lim. Signal estimation from modified
    short-time Fourier transform. IEEE Trans. Acoust. Speech
    Signal Process., 32(2):236-243, 1984.

    Malcolm Slaney, Daniel Naar and Richard F. Lyon. Auditory
    Model Inversion for Sound Separation. Proc. IEEE-ICASSP,
    Adelaide, 1994, II.77-80.

    Xinglei Zhu, G. Beauregard, L. Wyse. Real-Time Signal
    Estimation from Modified Short-Time Fourier Transform
    Magnitude Spectra. IEEE Transactions on Audio Speech and
    Language Processing, 08/2007.
    """
    x1 = x1 - x1.mean()
    x2 = x2 - x2.mean()
    frame_size = len(x2)
    half = frame_size // 2
    corrs = np.convolve(x1.astype('float32'), x2[::-1].astype('float32'))
    corrs[:half] = -1E30
    corrs[-half:] = -1E30
    offset = corrs.argmax() - len(x1)
    return offset


def invert_spectrogram(X_s, step, calculate_offset=True, set_zero_phase=True):
    """
    Under MSR-LA License

    Based on MATLAB implementation from Spectrogram Inversion Toolbox

    References
    ----------
    D. Griffin and J. Lim. Signal estimation from modified
    short-time Fourier transform. IEEE Trans. Acoust. Speech
    Signal Process., 32(2):236-243, 1984.

    Malcolm Slaney, Daniel Naar and Richard F. Lyon. Auditory
    Model Inversion for Sound Separation. Proc. IEEE-ICASSP,
    Adelaide, 1994, II.77-80.

    Xinglei Zhu, G. Beauregard, L. Wyse. Real-Time Signal
    Estimation from Modified Short-Time Fourier Transform
    Magnitude Spectra. IEEE Transactions on Audio Speech and
    Language Processing, 08/2007.
    """
    size = int(X_s.shape[1] // 2)
    wave = np.zeros((X_s.shape[0] * step + size))
    # Getting overflow warnings with 32 bit...
    wave = wave.astype('float64')
    total_windowing_sum = np.zeros((X_s.shape[0] * step + size))
    win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))

    est_start = int(size // 2) - 1
    est_end = est_start + size
    for i in range(X_s.shape[0]):
        wave_start = int(step * i)
        wave_end = wave_start + size
        if set_zero_phase:
            #spectral_slice = X_s[i].real + np.random.random(X_s[i].shape)*1.j
            spectral_slice = X_s[i].real + 0j
        else:
            # already complex
            spectral_slice = X_s[i]

        # Don't need fftshift due to different impl.
        wave_est = np.real(np.fft.ifft(spectral_slice))[::-1]
        if calculate_offset and i > 0:
            offset_size = size - step
            if offset_size <= 0:
                print("WARNING: Large step size >50\% detected! "
                      "This code works best with high overlap - try "
                      "with 75% or greater")
                offset_size = step
            offset = xcorr_offset(wave[wave_start:wave_start + offset_size],
                                  wave_est[est_start:est_start + offset_size])
        else:
            offset = 0
        wave[wave_start:wave_end] += win * wave_est[
            est_start - offset:est_end - offset]
        total_windowing_sum[wave_start:wave_end] += win
    wave = np.real(wave) / (total_windowing_sum + 1E-6)
    return wave


def iterate_invert_spectrogram(X_s, fftsize, step, n_iter=10, verbose=False):
    """
    Under MSR-LA License

    Based on MATLAB implementation from Spectrogram Inversion Toolbox

    References
    ----------
    D. Griffin and J. Lim. Signal estimation from modified
    short-time Fourier transform. IEEE Trans. Acoust. Speech
    Signal Process., 32(2):236-243, 1984.

    Malcolm Slaney, Daniel Naar and Richard F. Lyon. Auditory
    Model Inversion for Sound Separation. Proc. IEEE-ICASSP,
    Adelaide, 1994, II.77-80.

    Xinglei Zhu, G. Beauregard, L. Wyse. Real-Time Signal
    Estimation from Modified Short-Time Fourier Transform
    Magnitude Spectra. IEEE Transactions on Audio Speech and
    Language Processing, 08/2007.
    """
    reg = np.max(X_s) / 1E8
    X_best = copy.deepcopy(X_s)
    for i in range(n_iter):
        if verbose:
            print("Runnning iter %i" % i)
        if i == 0:
            X_t = invert_spectrogram(X_best, step, calculate_offset=True,
                                     set_zero_phase=True)
        else:
            # Calculate offset was False in the MATLAB version
            # but in mine it massively improves the result
            # Possible bug in my impl?
            X_t = invert_spectrogram(X_best, step, calculate_offset=True,
                                     set_zero_phase=False)
        est = stft(X_t, fftsize=fftsize, step=step, compute_onesided=False)
        phase = est / np.maximum(reg, np.abs(est))
        X_best = X_s * phase[:len(X_s)]
    X_t = invert_spectrogram(X_best, step, calculate_offset=True,
                             set_zero_phase=False)
    return np.real(X_t)


def implot(arr, scale=None, title="", cmap="gray"):
    import matplotlib.pyplot as plt
    if scale is "specgram":
        # plotting part
        mag = 10. * np.log10(np.abs(arr))
        # Transpose so time is X axis, and invert y axis so
        # frequency is low at bottom
        mag = mag.T[::-1, :]
    else:
        mag = arr
    f, ax = plt.subplots()
    ax.matshow(mag, cmap=cmap)
    plt.axis("off")
    x1 = mag.shape[0]
    y1 = mag.shape[1]

    def autoaspect(x_range, y_range):
        """
        The aspect to make a plot square with ax.set_aspect in Matplotlib
        """
        mx = max(x_range, y_range)
        mn = min(x_range, y_range)
        if x_range <= y_range:
            return mx / float(mn)
        else:
            return mn / float(mx)
    asp = autoaspect(x1, y1)
    ax.set_aspect(asp)
    plt.title(title)


def test_lpc_to_lsf():
    # Matlab style vectors for testing
    # lsf = [0.7842 1.5605 1.8776 1.8984 2.3593]
    # a = [1.0000  0.6149  0.9899  0.0000  0.0031 -0.0082];
    lsf = [[0.7842, 1.5605, 1.8776, 1.8984, 2.3593],
           [0.7842, 1.5605, 1.8776, 1.8984, 2.3593]]
    a = [[1.0000, 0.6149, 0.9899, 0.0000, 0.0031, -0.0082],
         [1.0000, 0.6149, 0.9899, 0.0000, 0.0031, -0.0082]]
    a = np.array(a)
    lsf = np.array(lsf)
    lsf_r = lpc_to_lsf(a)
    assert_almost_equal(lsf, lsf_r, decimal=4)
    a_r = lsf_to_lpc(lsf)
    assert_almost_equal(a, a_r, decimal=4)
    lsf_r = lpc_to_lsf(a[0])
    assert_almost_equal(lsf[0], lsf_r, decimal=4)
    a_r = lsf_to_lpc(lsf[0])
    assert_almost_equal(a[0], a_r, decimal=4)


def test_lpc_analysis_truncate():
    # Test that truncate doesn't crash and actually truncates
    [a, g, e] = lpc_analysis(np.random.randn(85), order=8, window_step=80,
                             window_size=80, emphasis=0.9, truncate=True)
    assert(a.shape[0] == 1)


def test_feature_build():
    samplerate, X = fetch_sample_music()
    # MATLAB wavread does normalization
    X = X.astype('float32') / (2 ** 15)
    wsz = 256
    wst = 128
    a, g, e = lpc_analysis(X, order=8, window_step=wst,
                           window_size=wsz, emphasis=0.9,
                           copy=True)
    v, p = voiced_unvoiced(X, window_size=wsz,
                           window_step=wst)
    c = compress(e, n_components=64)
    # First component of a is always 1
    combined = np.hstack((a[:, 1:], g, c[:a.shape[0]]))
    features = np.zeros((a.shape[0], 2 * combined.shape[1]))
    start_indices = v * combined.shape[1]
    start_indices = start_indices.astype('int32')
    end_indices = (v + 1) * combined.shape[1]
    end_indices = end_indices.astype('int32')
    for i in range(features.shape[0]):
        features[i, start_indices[i]:end_indices[i]] = combined[i]


def test_mdct_and_inverse():
    fs, X = fetch_sample_music()
    X_dct = mdct_slow(X)
    X_r = imdct_slow(X_dct)
    assert np.all(np.abs(X_r[:len(X)] - X) < 1E-3)
    assert np.abs(X_r[:len(X)] - X).mean() < 1E-6


def test_all():
    test_lpc_analysis_truncate()
    test_feature_build()
    test_lpc_to_lsf()
    test_mdct_and_inverse()


def run_lpc_example():
    # ae.wav is from
    # http://www.linguistics.ucla.edu/people/hayes/103/Charts/VChart/ae.wav
    # Partially following the formant tutorial here
    # http://www.mathworks.com/help/signal/ug/formant-estimation-with-lpc-coefficients.html

    samplerate, X = fetch_sample_music()

    c = overlap_compress(X, 200, 400)
    X_r = overlap_uncompress(c, 400)
    wavfile.write('lpc_uncompress.wav', samplerate, soundsc(X_r))

    print("Calculating sinusoids")
    f_hz, m = sinusoid_analysis(X, input_sample_rate=16000)
    Xs_sine = sinusoid_synthesis(f_hz, m)
    orig_fname = 'lpc_orig.wav'
    sine_fname = 'lpc_sine_synth.wav'
    wavfile.write(orig_fname, samplerate, soundsc(X))
    wavfile.write(sine_fname, samplerate, soundsc(Xs_sine))

    lpc_order_list = [8, ]
    dct_components_list = [200, ]
    window_size_list = [400, ]
    # Seems like a dct component size of ~2/3rds the step
    # (1/3rd the window for 50% overlap) works well.
    for lpc_order in lpc_order_list:
        for dct_components in dct_components_list:
            for window_size in window_size_list:
                # 50% overlap
                window_step = window_size // 2
                a, g, e = lpc_analysis(X, order=lpc_order,
                                       window_step=window_step,
                                       window_size=window_size, emphasis=0.9,
                                       copy=True)
                print("Calculating LSF")
                lsf = lpc_to_lsf(a)
                # Not window_size - window_step! Need to implement overlap
                print("Calculating compression")
                c = compress(e, n_components=dct_components,
                             window_size=window_step)
                co = overlap_compress(e, n_components=dct_components,
                                      window_size=window_step)
                block_excitation = uncompress(c, window_size=window_step)
                overlap_excitation = overlap_uncompress(co,
                                                        window_size=window_step)
                a_r = lsf_to_lpc(lsf)
                f, m = lpc_to_frequency(a_r, g)
                block_lpc = lpc_synthesis(a_r, g, block_excitation,
                                          emphasis=0.9,
                                          window_step=window_step)
                overlap_lpc = lpc_synthesis(a_r, g, overlap_excitation,
                                            emphasis=0.9,
                                            window_step=window_step)
                v, p = voiced_unvoiced(X, window_size=window_size,
                                       window_step=window_step)
                noisy_lpc = lpc_synthesis(a_r, g, voiced_frames=v,
                                          emphasis=0.9,
                                          window_step=window_step)
                if dct_components is None:
                    dct_components = window_size
                noisy_fname = 'lpc_noisy_synth_%iwin_%ilpc_%idct.wav' % (
                    window_size, lpc_order, dct_components)
                block_fname = 'lpc_block_synth_%iwin_%ilpc_%idct.wav' % (
                    window_size, lpc_order, dct_components)
                overlap_fname = 'lpc_overlap_synth_%iwin_%ilpc_%idct.wav' % (
                    window_size, lpc_order, dct_components)
                wavfile.write(noisy_fname, samplerate, soundsc(noisy_lpc))
                wavfile.write(block_fname, samplerate,
                              soundsc(block_lpc))
                wavfile.write(overlap_fname, samplerate,
                              soundsc(overlap_lpc))


def run_fft_vq_example():
    def _pre(list_of_data):
        n_fft = 512
        f_c = np.vstack([stft(dd, n_fft) for dd in list_of_data])
        f_mag = complex_to_abs(f_c)
        f_phs = complex_to_angle(f_c)
        f_sincos = angle_to_sin_cos(f_phs)
        f_r = np.hstack((f_mag, f_sincos))
        return f_r, n_fft

    def preprocess_train(list_of_data, random_state):
        f_r, n_fft = _pre(list_of_data)
        clusters = f_r
        return clusters

    def apply_preprocess(list_of_data, clusters):
        f_r, n_fft = _pre(list_of_data)
        memberships, distances = vq(f_r, clusters)
        vq_r = clusters[memberships]
        f_mag = vq_r[:, :n_fft // 2]
        f_sincos = vq_r[:, n_fft // 2:]
        extent = f_sincos.shape[1] // 2
        f_phs = sin_cos_to_angle(f_sincos[:, :extent], f_sincos[:, extent:])
        vq_c = abs_and_angle_to_complex(f_mag, f_phs)
        d_k = istft(vq_c, fftsize=n_fft)
        return d_k

    random_state = np.random.RandomState(1999)

    # Doesn't work yet for unknown reasons...
    fs, d = fetch_sample_music()
    sub = int(.8 * d.shape[0])
    d1 = [d[:sub]]
    d2 = [d[sub:]]

    """
    fs, d = fetch_sample_speech_fruit()
    d1 = d[::8] + d[1::8] + d[2::8] + d[3::8] + d[4::8] + d[5::8] + d[6::8]
    d2 = d[7::8]
    # make sure d1 and d2 aren't the same!
    assert [len(di) for di in d1] != [len(di) for di in d2]
    """

    clusters = preprocess_train(d1, random_state)
    # Training data
    vq_d1 = apply_preprocess(d1, clusters)
    vq_d2 = apply_preprocess(d2, clusters)
    assert [i != j for i, j in zip(vq_d2.ravel(), vq_d2.ravel())]

    fix_d1 = np.concatenate(d1)
    fix_d2 = np.concatenate(d2)

    wavfile.write("fft_train_no_agc.wav", fs, soundsc(fix_d1))
    wavfile.write("fft_test_no_agc.wav", fs, soundsc(fix_d2))
    wavfile.write("fft_vq_train_no_agc.wav", fs, soundsc(vq_d1, fs))
    wavfile.write("fft_vq_test_no_agc.wav", fs, soundsc(vq_d2, fs))

    agc_d1, freq_d1, energy_d1 = time_attack_agc(fix_d1, fs, .5, 5)
    agc_d2, freq_d2, energy_d2 = time_attack_agc(fix_d2, fs, .5, 5)
    agc_vq_d1, freq_vq_d1, energy_vq_d1 = time_attack_agc(vq_d1, fs, .5, 5)
    agc_vq_d2, freq_vq_d2, energy_vq_d2 = time_attack_agc(vq_d2, fs, .5, 5)

    wavfile.write("fft_train_agc.wav", fs, soundsc(agc_d1))
    wavfile.write("fft_test_agc.wav", fs, soundsc(agc_d2))
    wavfile.write("fft_vq_train_agc.wav", fs, soundsc(agc_vq_d1, fs))
    wavfile.write("fft_vq_test_agc.wav", fs, soundsc(agc_vq_d2))


def run_dct_vq_example():
    def _pre(list_of_data):
        # Temporal window setting is crucial! - 512 seems OK for music, 256
        # fruit perhaps due to samplerates
        n_dct = 512
        f_r = np.vstack([mdct_slow(dd, n_dct) for dd in list_of_data])
        return f_r, n_dct

    def preprocess_train(list_of_data, random_state):
        f_r, n_dct = _pre(list_of_data)
        clusters = f_r
        return clusters

    def apply_preprocess(list_of_data, clusters):
        f_r, n_dct = _pre(list_of_data)
        f_clust = f_r
        memberships, distances = vq(f_clust, clusters)
        vq_r = clusters[memberships]
        d_k = imdct_slow(vq_r, n_dct)
        return d_k

    random_state = np.random.RandomState(1999)

    # This doesn't work very well due to only taking a sample from the end as
    # test
    fs, d = fetch_sample_music()
    sub = int(.8 * d.shape[0])
    d1 = [d[:sub]]
    d2 = [d[sub:]]

    """
    fs, d = fetch_sample_speech_fruit()
    d1 = d[::8] + d[1::8] + d[2::8] + d[3::8] + d[4::8] + d[5::8] + d[6::8]
    d2 = d[7::8]
    # make sure d1 and d2 aren't the same!
    assert [len(di) for di in d1] != [len(di) for di in d2]
    """

    clusters = preprocess_train(d1, random_state)
    # Training data
    vq_d1 = apply_preprocess(d1, clusters)
    vq_d2 = apply_preprocess(d2, clusters)
    assert [i != j for i, j in zip(vq_d2.ravel(), vq_d2.ravel())]

    fix_d1 = np.concatenate(d1)
    fix_d2 = np.concatenate(d2)

    wavfile.write("dct_train_no_agc.wav", fs, soundsc(fix_d1))
    wavfile.write("dct_test_no_agc.wav", fs, soundsc(fix_d2))
    wavfile.write("dct_vq_train_no_agc.wav", fs, soundsc(vq_d1))
    wavfile.write("dct_vq_test_no_agc.wav", fs, soundsc(vq_d2))

    """
    import matplotlib.pyplot as plt
    plt.specgram(vq_d2, cmap="gray")
    plt.figure()
    plt.specgram(fix_d2, cmap="gray")
    plt.show()
    """

    agc_d1, freq_d1, energy_d1 = time_attack_agc(fix_d1, fs, .5, 5)
    agc_d2, freq_d2, energy_d2 = time_attack_agc(fix_d2, fs, .5, 5)
    agc_vq_d1, freq_vq_d1, energy_vq_d1 = time_attack_agc(vq_d1, fs, .5, 5)
    agc_vq_d2, freq_vq_d2, energy_vq_d2 = time_attack_agc(vq_d2, fs, .5, 5)

    wavfile.write("dct_train_agc.wav", fs, soundsc(agc_d1))
    wavfile.write("dct_test_agc.wav", fs, soundsc(agc_d2))
    wavfile.write("dct_vq_train_agc.wav", fs, soundsc(agc_vq_d1))
    wavfile.write("dct_vq_test_agc.wav", fs, soundsc(agc_vq_d2))


def run_phase_reconstruction_example():
    fs, d = fetch_sample_speech_tapestry()
    # actually gives however many components you say! So double what .m file
    # says
    fftsize = 512
    step = 64
    X_s = np.abs(stft(d, fftsize=fftsize, step=step, real=False,
                      compute_onesided=False))
    X_t = iterate_invert_spectrogram(X_s, fftsize, step, verbose=True)

    """
    import matplotlib.pyplot as plt
    plt.specgram(d, cmap="gray")
    plt.savefig("1.png")
    plt.close()
    plt.imshow(X_s, cmap="gray")
    plt.savefig("2.png")
    plt.close()
    """

    wavfile.write("phase_original.wav", fs, soundsc(d))
    wavfile.write("phase_reconstruction.wav", fs, soundsc(X_t))


def run_phase_vq_example():
    def _pre(list_of_data):
        # Temporal window setting is crucial! - 512 seems OK for music, 256
        # fruit perhaps due to samplerates
        n_fft = 256
        step = 32
        f_r = np.vstack([np.abs(stft(dd, n_fft, step=step, real=False,
                                compute_onesided=False))
                         for dd in list_of_data])
        return f_r, n_fft, step

    def preprocess_train(list_of_data, random_state):
        f_r, n_fft, step = _pre(list_of_data)
        clusters = copy.deepcopy(f_r)
        return clusters

    def apply_preprocess(list_of_data, clusters):
        f_r, n_fft, step = _pre(list_of_data)
        f_clust = f_r
        # Nondeterministic ?
        memberships, distances = vq(f_clust, clusters)
        vq_r = clusters[memberships]
        d_k = iterate_invert_spectrogram(vq_r, n_fft, step, verbose=True)
        return d_k

    random_state = np.random.RandomState(1999)

    fs, d = fetch_sample_speech_fruit()
    d1 = d[::9]
    d2 = d[7::8][:5]
    # make sure d1 and d2 aren't the same!
    assert [len(di) for di in d1] != [len(di) for di in d2]

    clusters = preprocess_train(d1, random_state)
    fix_d1 = np.concatenate(d1)
    fix_d2 = np.concatenate(d2)
    vq_d2 = apply_preprocess(d2, clusters)

    wavfile.write("phase_train_no_agc.wav", fs, soundsc(fix_d1))
    wavfile.write("phase_vq_test_no_agc.wav", fs, soundsc(vq_d2))

    agc_d1, freq_d1, energy_d1 = time_attack_agc(fix_d1, fs, .5, 5)
    agc_d2, freq_d2, energy_d2 = time_attack_agc(fix_d2, fs, .5, 5)
    agc_vq_d2, freq_vq_d2, energy_vq_d2 = time_attack_agc(vq_d2, fs, .5, 5)

    """
    import matplotlib.pyplot as plt
    plt.specgram(agc_vq_d2, cmap="gray")
    #plt.title("Fake")
    plt.figure()
    plt.specgram(agc_d2, cmap="gray")
    #plt.title("Real")
    plt.show()
    """

    wavfile.write("phase_train_agc.wav", fs, soundsc(agc_d1))
    wavfile.write("phase_test_agc.wav", fs, soundsc(agc_d2))
    wavfile.write("phase_vq_test_agc.wav", fs, soundsc(agc_vq_d2))


if __name__ == "__main__":
    """
    Trying to run all examples will seg fault on my laptop - probably memory!
    Comment individually
    """
    # run_phase_reconstruction_example()
    run_phase_vq_example()
    # run_dct_vq_example()
    # run_fft_vq_example()
    # run_lpc_example()
    test_all()
