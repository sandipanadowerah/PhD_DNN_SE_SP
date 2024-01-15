#!/bin/bash

export PATH=/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/sdowerah/anaconda/bin:$PATH

source activate robovox

python training_channel_attention_m1.py config_ch_attention_m1.json
