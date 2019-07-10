"""
  Written by Pengfei Sun. The code is used for domain transfer.
  From larget scale general vocabulary to domain specific vocabulary. 
"""

import tensorflow as tf
from functools import lru_cache
import json

@lru_cache()
def bytes_to_unicode():
    """
       Returns list of utf-8 byte and corresponding list of unicode strings
    """
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + \
         list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n  = 0
    for b in range(2**8):
        if b not in bs:
           bs.append(b)
           cs.append(2**8+n)
           n += 1
    cs = [chr(n) for n in cs]
    
    return dict(zip(bs, cs))

def decode(tokens, decoder):
    """
       Input digit tokens index the word sequences. tokens: int
       decoder is the dictionary using index as key and word as the value.
    """
  #  byte_encoder = bytes_to_unicode()
    byte_decoder = {v:k for k, v in bytes_to_unicode().items()}
    text = ''.join(decoder[token] for token in tokens)
    text = bytearray([byte_decoder[c] for c in text]).decode('utf-8', errors="replace")
    
    return text 

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





# create reduced_decoder
