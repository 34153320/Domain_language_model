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
 
def get_pairs(word):
    """Return set of symbol pairs in a word.
       Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def byte_decoder():
     """
        Reversion of bytes_to_unicode
     """
     return {v:k for k, v in bytes_to_unicode().items()}
    
class Encoder:
    def __init__(self, encoder, bpe_merges):
        # bpe_merges : reduced version 
        self.encoder = encoder
        self.decoder = {v:k for k, v in self.encoder.items()}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = byte_decoder()
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    def bpe(self, token):
        if token in self.cache:
             return self.cache[token]
        word  = tuple(token)
        pairs = get_pairs(word)
        
        if not pairs:
            return token
        while True:
              bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
              if bigram not in self.bpe_ranks:
                 break
              first, second = bigram
              new_word = []
              i = 0
              while i < len(word):
                   try:
                       j = word.index(first, i)
                       new_word.extend(word[i:j])
                       i = j
                   except:
                       new_word.extend(word[i:])
                       break
                   
                   if word[i] == first and i < len(word)-1 and word[i+1]==second:
                      new_word.append(first+second)
                      i += 2
                   else:
                      new_word.append(word[i])
                      i += 1
              new_word = tuple(new_word)
              word  = new_word
              if len(word) == 1:
                 break
              else:
                 pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        
        return word
                     
    def encode(self, text):
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
            
        return bpe_tokens
            
    def decode(self, tokens, decoder):
    """
       Input digit tokens index the word sequences. tokens: int
       decoder is the dictionary using index as key and word as the value.
    """
     #  byte_encoder = bytes_to_unicode()
       byte_decoder = byte_decoder()
       text = ''.join(decoder[token] for token in tokens)
       text = bytearray([byte_decoder[c] for c in text]).decode('utf-8', errors="replace")
       return text 

def create_encoder(models_dir, domain_dict):
    with open(models_dir + 'encoder.json', 'r') as read_file:
         complete_encoder = json.load(read_file)
    with open(models_dir + 'vocab.bpe', 'r', encoding="utf-8") as read_file:
         bpe_data = read_file.read()
    
    reduce_encoder = {}
    byte_decoder = byte_decoder()
    for sub_text in domain_dict:
       for keys, index_seq in complete_encoder.items():
           keys = bytearray([byte_decoder[c] for c in keys]).decode('utf-8', errors='replace')
           if sub_text == keys or sub_text.lower()==keys or \
              (" " + sub_text==keys) or (" " + sub_text.lower()==keys):
              # dict {k:v}
              reduced_encoder[keys] = index_seq
    
              
    return Encoder(
         encoder    = reduced_encoder,  # using the reduced encoder dictionary
         bpe_merges = bpe_merges, 
         )
    
  
