# Unet with Beamforming

# y = s + n
# noisy_speech = clean_speech + noise speech

# for mvdr beamforming input is noisy speech, y and theta which is inverse tan between source and mic position

Input : y_stft along with y_mvdr_stft (output stft from mvdr beamforming)

1 x 3 x FrameLength X (win_len/2 +1)

Output : predicted mask, pred_mask

1 x 1 x FrameLength X (win_len/2 +1)



# data_utils.py : preprocessing with beamforming in get_data() function

# mask_utils.py : masking, multichannel_weiner_filter_current, multichannel_weiner_filter_previous

# unet.py : U_Net, U_Net_Channel_Attention, U_Net_Attention

# loss_functions.py : MSE loss with mask, MSE loss with clean speech stft and clean speech estimate

# training.py : training module for unet

# beamformer directory have functions related to mvdr beamforming estimation

# config.json : 

{
    "model": {
        "in_channel": 3,
        "learning_rate": 0.001,
        "weight_decay": 1e-6,
        "num_epochs": 100,
        "use_cuda": 1,
	"start_epoch":0
    },

    "data": {
        "test_array": "test_array_robovox.txt",
        "train_array": "train_array_robovox.txt",
        "dirpath": "/dataset/LibriSpeech/audio/train-reverb-robovox/WAV",
        "model_dir": "/u_net_masking/unet/",
        "checkpoint_model_path": "",
	"checkpoint_optimizer_path":"",
        "checkpoint_interval": 5
    }


}











