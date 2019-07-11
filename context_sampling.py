# sampling sentences based on providing contexts

import fire
import json
import os
import numpy as np
import tensorflow as tf
import Transformer, generate, Domain_transfer

def pretrain_model(
    model_name="pretrained",
    seed = 1234, 
    nsamples=1,
    batch_size=1,
    length = 10,
    temperature=1, 
    top_k = 9,
    models_dir='models',
):
    model_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0
    
    


