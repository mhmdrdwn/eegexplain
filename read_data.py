import mne
from tqdm import tqdm
from statsmodels.tsa.stattools import grangercausalitytests
import random
import numpy as np
from scipy.signal import savgol_filter
from scipy import signal
import cv2
from numpy import *
from scipy.stats import iqr
from scipy.integrate import simps
from scipy.stats import entropy
from scipy.signal import medfilt
from scipy.stats import differential_entropy, norm
from scipy import fftpack
import antropy as ant
from sklearn.model_selection import train_test_split
    
import scipy.io
import numpy as np
import os

def read_save():
    path = "mci_ad_dataset/CONTROL/"
    arr = []
    for folder in os.listdir(path):
        if "normal" in folder:
            arr = []
            for file in os.listdir(path+folder):
                mat = scipy.io.loadmat(path+folder+"/"+file)["export"]
                if mat.shape[-1] == 19:
                    arr.extend(list(mat))
            if len(arr) > 0:
                arr = np.vstack(arr)
            else:
                arr = np.array(arr)
        else:
            mat = scipy.io.loadmat(path+folder)['segmenty']
        np.save("mci_ad_dataset_npy/"+str(folder[:-4]), arr) 
    path = "mci_ad_dataset/AD/"
    arr = []
    for folder in os.listdir(path):
        arr = []
        for file in os.listdir(path+folder):
            mat = scipy.io.loadmat(path+folder+"/"+file)["export"]
            if mat.shape[-1] == 19:
                arr.extend(list(mat))
        if len(arr) > 0:
            arr = np.vstack(arr)
        else:
            arr = np.array(arr)
        np.save("mci_ad_dataset_npy/"+str(folder), arr)
    path = "mci_ad_dataset/MCI/"
    arr = []
    for folder in os.listdir(path):
        arr = []
        for file in os.listdir(path+folder):
            mat = scipy.io.loadmat(path+folder+"/"+file)["export"]
            if mat.shape[-1] == 19:
                arr.extend(list(mat))
        if len(arr) > 0:
            arr = np.vstack(arr)
        else:
            arr = np.array(arr)
        np.save("mci_ad_dataset_npy/"+str(folder), arr)

        
def get_label(file):
    if "AD" in file:
        return [1]
    elif "MCI" in file:
        return [2]
    else:
        return [0]

def calcGrangerCausality(eegData, ii, jj):
    all_gc = []
    for epoch_idx, epoch in enumerate(eegData):
        #if epoch_idx%2 == 0:
        X = np.vstack([epoch[ii,:],epoch[jj,:]]).T
        gc = grangercausalitytests(X, [2], addconst=True, verbose=False)[2][0]['ssr_ftest'][1]
        all_gc.append(gc)
            
    return all_gc
    
import numpy as np

def calCorr(eegData, i, j):    
    all_corr = []
    for epoch_idx, epoch in enumerate(eegData):
        corr1 = np.correlate(epoch[i], epoch[j])[0]
        all_corr.append(corr1)
    all_corr = np.array(all_corr)
    return all_corr


def break_array2chunks(x, epoch_duration):
    size = epoch_duration
    step = int(epoch_duration*0.5)
    trunc = x[x.shape[0]%size]
    x = [x[i : i + size] for i in range(0, len(x), step)]
    return x

def break_all_array(x, size):
    chunk_array = []
    for ch_idx in range(x.shape[1]):
        x_ch = x[:, ch_idx]
        chunck = break_array2chunks(x_ch, size)
        chunk_array.append(chunck[:-2])    
    chunk_array = np.array(chunk_array)
    return chunk_array

def stack_arrays(x, g, y):
    out_x = np.moveaxis(np.array(x[0]), 0, 1)
    out_g = np.moveaxis(np.array(g[0]), -1, 0)
    out_y = [y[0] for _ in range(out_x.shape[0])]
    for arr, gr, yl in zip(x[1:], g[1:], y[1:]):
        arr = np.array(arr)
        arr = np.moveaxis(arr, 0, 1)
        out_x = np.concatenate((out_x, arr), axis=0)
        gr = np.array(gr)
        gr = np.moveaxis(gr, -1, 0)
        out_g = np.concatenate((out_g, gr), axis=0)
        out_y.extend([yl for _ in range(arr.shape[0])])
    out_y = np.array(out_y)
    out_x = np.moveaxis(out_x, 1, 2)
    out_x = np.moveaxis(out_x, 2, -1)
    return out_x, out_g, out_y

def build_data(raw_data, size, cal_conn=None,  bands=True):
    
    eeg_data = []
    fs = 128
    
    all_data_features = []
    data_labels = []
    data_graphs = []
    
    for file in tqdm(raw_data):
        #node features
        sample_features = []
        data_features = []
        filtered_eeg = []
        
        try:
            data = mne.io.read_raw(file, verbose=False, preload=True)
            data.filter(l_freq=0.5, h_freq=45, verbose=False)#.resample(sfreq=128)
            data_epochs = mne.make_fixed_length_epochs(data, duration=2.5, overlap=1.25, verbose=False)
            data_epochs = data_epochs.get_data()[:, :-2 , :]
            data_epochs = np.moveaxis(data_epochs, 0, 1)
            data_epochs = detrend(data_epochs, type="constant", axis=-1)
            
            # freq domain features
            # delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz), and gamma (30–45 Hz).
            freq_ranges = [[0.5, 4], [4, 8], [8, 13], [13, 30], [30, 45]]

            for freq_band in freq_ranges:
                #data_epochs_new = np.moveaxis(data_epochs, 0, 1)
                sos = signal.butter(4, (freq_band[0], freq_band[1]), 'bandpass', fs=128, output='sos')
                data_epochs_new = signal.sosfilt(sos, data_epochs)
                data_epochs_new = break_all_array(data_epochs_new, size)
                data_epochs_new = np.moveaxis(data_epochs_new, 0, 1)
                filtered_eeg.append(data_epochs_new)                    
        
            #get label
            label = get_label(file)
            data_labels.append(label)
            filtered_eeg = np.array(filtered_eeg)

            if cal_conn=="gc":
                band_c = []
                for band_idx in range(len(freq_ranges)):
                    c = []
                    for i in range(19):
                        c1 = []
                        for j in range(19):
                            c1.append(calcGrangerCausality(filtered_eeg[band_idx, :, :, :], i, j))
                        #c1 = np.array(c1)
                        c.append(c1)
                    #c = np.array(c)
                    band_c.append(c)
                #band_c = np.array(band_c)
                data_graphs.append(band_c) 

            elif cal_conn=="corr":
                band_c = []
                for band_idx in range(len(freq_ranges)):
                    c = []
                    for i in range(19):
                        c1 = []
                        for j in range(19):
                            c1.append(calCorr(filtered_eeg[band_idx, :, :, :], i, j))
                        #c1 = np.array(c1)
                        c.append(c1)
                    #c = np.array(c)
                    band_c.append(c)
                #band_c = np.array(band_c)
                data_graphs.append(band_c) 

            data_features = []
            # freq domain features
            for freq_band_eeg in filtered_eeg:
                epoch_features = []
                for filtered_epoch in freq_band_eeg:
                    ch_features = []
                    for filtered_ch in filtered_epoch:
                        #ch_features = []

                        # differential entropy
                        ch_features.append(np.abs(differential_entropy(filtered_ch)))

                        # Permutation entropy
                        entropy = ant.perm_entropy(filtered_ch, normalize=True)
                        ch_features.append(entropy)

                        # Spectral entropy
                        entropy = ant.spectral_entropy(filtered_ch, sf=200, method='welch', normalize=True)
                        ch_features.append(entropy)

                        # Singular value decomposition entropy
                        entropy = ant.svd_entropy(filtered_ch, normalize=True)
                        ch_features.append(entropy)

                        # Hjorth mobility and complexity
                        mobility = ant.hjorth_params(filtered_ch)
                        ch_features.append(mobility[0])
                        ch_features.append(mobility[1])

                        # Number of zero-crossings
                        zero_cross_count = ant.num_zerocross(filtered_ch)
                        ch_features.append(zero_cross_count)

                        # Petrosian fractal dimension
                        ch_features.append(ant.petrosian_fd(filtered_ch))
                        # Katz fractal dimension
                        ch_features.append(ant.katz_fd(filtered_ch))
                        # Higuchi fractal dimension
                        ch_features.append(ant.higuchi_fd(filtered_ch))
                        # Detrended fluctuation analysis
                        ch_features.append(ant.detrended_fluctuation(filtered_ch))

                        epoch_features.append(ch_features)
                        ch_features = []
                    sample_features.append(epoch_features)
                    epoch_features = []

                data_features.append(sample_features)
                sample_features = []
            all_data_features.append(data_features)
        except:
            pass
    return all_data_features, data_graphs, data_labels


def tr_test_split(test_size):
    path = "mci_ad_dataset_npy/"
    control_subjects_files, AD_subjects_files, MCI_subjects_files = [], [], []
    for file in os.listdir(path):
        if file[:2]=="AD":
            control_subjects_files.append(path+file)
        elif file[:3]=="MCI":
            MCI_subjects_files.append(path+file)
        else:
            AD_subjects_files.append(path+file)

    train_data_files = control_subjects_files + AD_subjects_files + MCI_subjects_files
    train_data_files, test_data_files = train_test_split(train_data_files, test_size=test_size, random_state=100)
    
    return train_data_files, test_data_files