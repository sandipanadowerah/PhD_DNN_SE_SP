#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn 
import numpy as np
import librosa as lb
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import soundfile as sf

import collections
import random

import pandas as pd
import librosa
import pickle

from beamformer import delaysum as ds
from beamformer import util

STFT_MIN = 1e-6
STFT_MAX = 1e3


def pad_to_maxlen(x, y):
    diff = len(x) - len(y)
    if diff > 0 :
        y = np.concatenate([y, np.zeros(diff)])
    elif diff < 0 :
        x = np.concatenate([x, np.zeros(abs(diff))])

    return x, y

class UnetDataset(Dataset):
    

    def __init__(self, dirpath, array_filepath, feature_dim=257):
        
        
        self.y_dir = os.path.join(dirpath, 'Noisy')
        self.s_dir = os.path.join(dirpath, 'Target')
        self.n_dir = os.path.join(dirpath, 'Noise')
        self.s_dry_dir = os.path.join(dirpath, 'dry_target')
        self.n_dry_dir = os.path.join(dirpath, 'dry_noise')

        self.rir_dir = os.path.join(dirpath, 'RIR')
        self.stft_mask_dir = os.path.join(dirpath, 'ds_mask_tunet')
        
        if os.path.exists(self.stft_mask_dir) == False:
            os.mkdir(self.stft_mask_dir)
        
        self.array_filepath = array_filepath        
        self.feature_dim = feature_dim
        
        self.data_files = sorted(self.get_fileid_list(self.array_filepath))
        
       
    #stft normparam
    def get_fileid_list(self, array_file):
        list_ = []
        for i in open(array_file, 'r'):
            list_.append(int(i.strip()))
        return list_
    
    def load_wav_data(self, rir_id, wav_dirpath, nch=3, stype = 'y'):
        wav_data = []
        for i in range(1,nch+1):
            if stype == 's':
                str_ = str(rir_id) +'_target_Ch-%d.wav'%i
            elif stype == 'n':
                str_ = str(rir_id) + '_robovox_Ch-%d.wav'%i
            elif stype == 'y':
                str_ = str(rir_id) + '_Mix-robovox_Ch-%d.wav'%i                    

            fpath = os.path.join(wav_dirpath,str_)
            y, sr = sf.read(fpath)
            wav_data.append(y)
        
        wav_data = np.transpose(np.asanyarray(wav_data))

        return wav_data, sr
    
    def stft_nchannel(self, y, win_len=512, win_hop=256, n_channel=3, center=False):
        # Input data parameters
        n_freq = int(win_len / 2 + 1)
        n_frames = int(1 + np.floor((len(y) - win_len) / win_hop))

        y_stft = np.zeros((n_freq, n_frames, n_channel), 'complex')

        if n_channel == 1:
            y_stft = librosa.core.stft(np.ascontiguousarray(y), n_fft=win_len, hop_length=win_hop, center=False)
        else:
            for i_ch in range(n_channel):
                y_stft[:, :, i_ch] = librosa.core.stft(np.ascontiguousarray(y[:, i_ch]), n_fft=win_len, hop_length=win_hop, center=False)


        return np.clip(y_stft, STFT_MIN, STFT_MAX)
    
    def ideal_ratio_mask(self, s_stft, n_stft):
        mask = np.square(np.abs(s_stft))/(np.square(np.abs(s_stft)) + np.square(np.abs(n_stft)))

        return mask

    def get_theta_phi(self, mic_xyz, source_position):
        phi = []
        for (x,y,z) in mic_xyz:
            #print(x,y,z)
            dy = y - source_position[1]
            dx = x - source_position[0]

            phi.append(np.rad2deg(np.arctan(dy/dx)))

        return phi

    def delay_and_sum_beamforming(self, y_audio, phi, win_len = 512, hop_len = 256, sr=16000):
        SAMPLING_FREQUENCY = sr
        FFT_LENGTH = win_len
        FFT_SHIFT = hop_len
        MIC_ANGLE_VECTOR = np.array([0, 120, 360])
        LOOK_DIRECTION = phi
        MIC_DIAMETER = 0.1

        complex_spectrum, _ = util.get_3dim_spectrum_from_data(y_audio, FFT_LENGTH, FFT_SHIFT, FFT_LENGTH)
        complex_spectrum =  np.clip(complex_spectrum, STFT_MIN, STFT_MAX)

        ds_beamformer = ds.delaysum(MIC_ANGLE_VECTOR, MIC_DIAMETER, 
                                                      sampling_frequency=SAMPLING_FREQUENCY, 
                                                      fft_length=FFT_LENGTH, fft_shift=FFT_SHIFT)

        beamformer = ds_beamformer.get_sterring_vector(LOOK_DIRECTION)

        enhanced_speech = ds_beamformer.apply_beamformer(beamformer, complex_spectrum)

        norm_wav = enhanced_speech / np.max(np.abs(enhanced_speech))

        len_diff = y_audio.shape[0] - norm_wav.shape[0]

        if len_diff < 0:   
            norm_wav = norm_wav[:y_audio.shape[0],]
        elif len_diff > 0:    
            norm_wav = np.concatenate((norm_wav, np.zeros(abs(len_diff))))

        norm_stft = self.stft_nchannel(norm_wav, n_channel=1)

        return norm_stft 
    
    def get_data(self, index):
        y, sr = self.load_wav_data(index, self.y_dir, stype='y') 
        s, sr = self.load_wav_data(index, self.s_dir, stype='s')
        n, sr = self.load_wav_data(index, self.n_dir, stype='n')

        s_dry, sr = sf.read(os.path.join(self.s_dry_dir, str(index)+'.wav'))
        n_dry, sr = sf.read(os.path.join(self.n_dry_dir, str(index)+'.wav'))

        y_stft = self.stft_nchannel(y)
        s_stft = self.stft_nchannel(s)
        n_stft = self.stft_nchannel(n)

        mask = self.ideal_ratio_mask(abs(s_stft), abs(n_stft))
        rir_info = np.load(os.path.join(self.rir_dir, str(index)+'_info.npy'), allow_pickle=True).any()
        theta = self.get_theta_phi(rir_info['mics_xyz'], rir_info['sous_xyz'][0])
        #print(theta)
        ds_stft = self.delay_and_sum_beamforming(y, theta)
                
        #np.save(os.path.join(self.stft_mask_dir, str(index)+'.npy'), {'mask':mask, 'ds_stft':ds_stft}, allow_pickle=True)
       
        y_stft = np.concatenate((abs(y_stft), np.expand_dims(abs(ds_stft), axis=2)), axis=2)
        y_stft = torch.from_numpy(abs(y_stft))
        s_stft = torch.from_numpy(abs(s_stft))
        mask = torch.from_numpy(mask)
        s_dry, y = pad_to_maxlen(s_dry, y[:,0])

        s_dry = torch.from_numpy(np.asanyarray(s_dry)).unsqueeze(1)
        y = torch.from_numpy(np.asanyarray(y)).unsqueeze(1)
        
        
        return y_stft, s_stft, mask, s_dry, y
        
    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        
        y_stft, s_stft, mask, s_dry, y = self.get_data(self.data_files[index])
        
        sample = {'y_stft':y_stft, 's_stft': s_stft, 'mask': mask, 's_dry': s_dry, 'y':y }
        
        return sample



def collate_fn_speech_enhancement(batch):

    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.Mapping):
        #print(len(batch))

        y_stft = [d['y_stft'] for d in batch]
        s_stft = [d['s_stft'] for d in batch]
        mask = [d['mask'] for d in batch]
        s_dry = [d['s_dry'] for d in batch]
        y = [d['y'] for d in batch]

        #print(type(mask[0]))

        #y_stft = torch.from_numpy(abs(np.asanyarray(y_stft)))
        #s_stft = torch.from_numpy(abs(np.asanyarray(s_stft)))
        #mask = torch.from_numpy(mask[0])
        
        
        return y_stft[0], s_stft[0], mask[0], s_dry[0], y[0]

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))




def get_UnetDataset(dir_path, array_filepath):
    return UnetDataset(dir_path, array_filepath)
    






