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
  
 def byte_decoder():
     """
        Reversion of bytes_to_unicode
     """
     return {v:k for k, v in bytes_to_unicode().items()}
     
def decode(tokens, decoder):
    """
       Input digit tokens index the word sequences. tokens: int
       decoder is the dictionary using index as key and word as the value.
    """
  #  byte_encoder = bytes_to_unicode()
    byte_decoder = byte_decoder()
    text = ''.join(decoder[token] for token in tokens)
    text = bytearray([byte_decoder[c] for c in text]).decode('utf-8', errors="replace")
    
    return text 

def create_domain_dict(encoder_path, domain_dict):
   # load the original encoder dict
   # encoder_path = "./models/encoder.json"
   with open(encoder_path, 'r') as read_file:
          encoder_dict = json.load(read_file)
   
  # create reduced_decoder
   reduced_dict = {}
   byte_decoder = byte_decoder()
   for sub_text in domain_dict:
       for keys, index_seq in encoder_dict.items():
           keys = bytearray([byte_decoder[c] for c in keys]).decode('utf-8', errors='replace')
           if sub_text == keys or sub_text.lower()==keys or \
              (" " + sub_text==keys) or (" " + sub_text.lower()==keys):
              # dict {k:v}
              reduced_dict[index_seq] = keys
  
   return reduced_dict
  
