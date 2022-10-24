import numpy as np
from numpy import linalg as la
from scipy.stats import pearsonr
from scipy import signal

def train_trca(eeg, fs, num_fbs):
    num_targs = eeg.shape[0]
    num_chans = eeg.shape[1]
    num_smpls = eeg.shape[2]

    weight = np.zeros((num_targs, num_chans, num_fbs))
    train_temp = np.zeros((num_targs, num_chans, num_smpls, num_fbs))
    for targ_i in range(num_targs):
        for fb_i in range(num_fbs):
            traindata = filterbank(eeg[targ_i, :, :, :], fs, fb_i)
            w_tmp = ftrca(traindata)
            weight[targ_i, :, fb_i] = w_tmp[:, 1]
            train_temp[targ_i, :, :, fb_i] = np.average(traindata, axis=2)

    return weight, train_temp

def test_trca(eeg, weight, template, fs, num_fbs, is_ensemble):
    num_targs = eeg.shape[0]
    fb_coefs = pow(np.array(range(1, num_fbs + 1, 1)), -1.25) + 0.25

    outclass = np.zeros(num_targs)
    for targ_i in range(num_targs):
        corr = np.zeros((num_targs, num_fbs))
        for fb_i in range(num_fbs):
            testdata = filterbank(eeg[targ_i, :, :], fs, fb_i)
            for class_i in range(num_targs):
                traindata = template[class_i, :, :, fb_i]
                if is_ensemble:
                    w = weight[:, :, fb_i]
                else:
                    w = weight[class_i, :, fb_i]
                corr[class_i, fb_i], _ = pearsonr(np.dot(w, testdata), np.dot(w, traindata))

        rho = np.dot(fb_coefs, corr.T)
        outclass[targ_i] = np.argmax(rho)

    return outclass

def ftrca(x):
    num_trials = x.shape[2]

    for trial_i in range(num_trials):
        x1 = x[:, :, trial_i]
        x[:, :, trial_i] = x1 - x1.mean(axis=1, keepdims=True)

    SX = np.sum(x, axis=2)
    S = np.dot(SX, SX.T)
    QX = x.reshape(x.shape[0], -1)
    Q = np.dot(QX, QX.T)
    W, V = la.eig(np.dot(la.inv(Q), S))

    idx = W.argsort()[::-1]
    V = V[:, idx]

    return V

def filterbank(x, fs, fb_i):
    num_chans = x.shape[0]
    num_smpls = x.shape[1]
    if x.ndim > 2:
        num_trials = x.shape[2]
    else:
        num_trials = 1
    fs = fs / 2

    passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
    stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]
    Wp = [passband[fb_i] / fs, 90 / fs]
    Ws = [stopband[fb_i] / fs, 100 / fs]
    gpass = 3
    gstop = 40
    Rp = 0.5
    [N, Wn] = signal.cheb1ord(Wp, Ws, gpass, gstop)
    [B, A] = signal.cheby1(N, Rp, Wn, 'bandpass')

    if num_trials == 1:
        y = np.zeros((num_chans, num_smpls))
        for ch_i in range(num_chans):
            y[ch_i, :] = signal.filtfilt(B, A, x[ch_i, :])
    else:
        y = np.zeros((num_chans, num_smpls, num_trials))
        for trial_i in range(num_trials):
            for ch_i in range(num_chans):
                y[ch_i, :, trial_i] = signal.filtfilt(B, A, x[ch_i, :, trial_i])

    return y