#!/bin/bash

export PATH=/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/sdowerah/anaconda/bin:$PATH

source activate robovox

python training_frame_attention.py config_sfn_frame_attention.json
