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
domain_dict = ["I", "You",  "My",  "They",  "It",  "Am", "Are", "Need", "Feel", "Is",  "Hungry",  
               "Help", "Tired", "Not", "How", "Okay", "Very", "Thirsty", "Comfortable", "Right",
               "Please", "Hope", "Clean", "Glasses", "Nurse", "Closer", "Bring", "What", "Where", 
               "Tell", "That", "Going", "Music", "Like", "Outside", "Do", "Have", "Faith", 
               "Success", "Coming", "Good", "Bad", "Here", "Family", "Hello", "Goodbye", 
               "Computer", "Yes", "Up", "No"]



# create reduced_decoder
