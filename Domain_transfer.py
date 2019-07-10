"""
  Written by Pengfei Sun. The code is used for domain transfer.
  From larget scale general vocabulary to domain specific vocabulary. 
"""

import tensorflow as tf
import json

# load the original encoder dict
encoder_path = "./models/encoder.json"
with open(encoder_path, 'r') as read_file:
     encoder_dict = json.load(read_file)

#domain dictionary


# create reduced_decoder
