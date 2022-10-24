import scipy.io
import numpy as np
from trca import train_trca, test_trca, ftrca, filterbank
import matplotlib.pyplot as plt

filename = 'data/sample.mat'
len_gaze_s = 0.5
len_delay_s = 0.13
num_fbs = 2
is_ensemble = 0
alpha_ci = 0.05
fs = 250
len_shift_s = 0.5

list_freqs = []
for i in range(5):
    for j in range(8):
        list_freqs.append(8+0.2*i+1*j)
num_targs = len(list_freqs)
list_labels = np.array(range(0,num_targs,1))

len_gaze_smpl = round(len_gaze_s*fs)
len_delay_smpl = round(len_delay_s*fs)
len_sel_s = len_gaze_s + len_shift_s
ci = 100*(1-alpha_ci)
data_segment_smpl = list(range(len_delay_smpl, len_delay_smpl+len_gaze_smpl))

matdata = scipy.io.loadmat(filename)
#matdata.keys()
eeg = matdata['eeg']
eeg_seg = eeg[:, :, data_segment_smpl, :]


num_trials = eeg_seg.shape[3]

accuracy = np.array(range(num_trials))
itr = np.array(range(num_trials))
for loocv_i in range(num_trials ):
    traindata = np.delete(eeg_seg, loocv_i, 3)
    weight, train_temp = train_trca(traindata, fs, num_fbs)

    #fig, ax = plt.subplots(4)
    #ax[0].plot(traindata[1 ,1 ,: ,:])
    #ax[1].plot(train_temp[1 ,1 ,: ,1])
    #ax[2].plot(np.dot(weight[1 ,: ,1], train_temp[1 ,: ,: ,1]))

    testdata = eeg_seg[: ,: ,: ,loocv_i]
    result = test_trca(testdata, weight, train_temp, fs, num_fbs, is_ensemble)
    #ax[3].plot(testdata[1 ,1 ,:])
    # print(result)

    # print(list_labels==result)
    is_correct = list_labels == result
    accuracy[loocv_i] = is_correct.mean( ) *100

print(accuracy)

