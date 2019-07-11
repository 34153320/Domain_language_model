import tensorflow as tf
import Transformer
from domain_transfer import create_domain_dict

"""
   Written by Pengfei Sun. Part of the codes are borrowed from openai/GPT-2.
"""
domain_dict = ["I", "You",  "My",  "They",  "It",  "Am", "Are", "Need", "Feel", "Is",  "Hungry",  
               "Help", "Tired", "Not", "How", "Okay", "Very", "Thirsty", "Comfortable", "Right",
               "Please", "Hope", "Clean", "Glasses", "Nurse", "Closer", "Bring", "What", "Where", 
               "Tell", "That", "Going", "Music", "Like", "Outside", "Do", "Have", "Faith", 
               "Success", "Coming", "Good", "Bad", "Here", "Family", "Hello", "Goodbye", 
               "Computer", "Yes", "Up", "No"]

def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return tf.cond(
       tf.equal(k, 0),
       lambda: logits,
       lambda: _top_k(),
    )

create_domain_dict(encoder_path, domain_dict)

def sample_sequence(*, hparams, length, reduced_dict_index, 
                    start_token=None, batch_size=None, context=None, 
                    temperature=1, top_k=0):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token)
      
    
    # mask dict is too large
    # dict_mask = np.zeros((hparams.n_vocab, hparams.n_vocab))
   
    def step(hparams, tokens, past=None):
        lm_output = Transformer.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)

        logits = lm_output['logits'][:, :, :hparams.n_vocab] # keep output dimension the same
        # tf.gather collect the corresponding logits 
 
        logits = logits * dict_mask
        presents = lm_output['present']
        presents.set_shape(Transformer.past_shape(hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

    """
       For domain transfer, the general langauage model maintains the structure. 
       Change the final softmax to adapte to specific domain knowledge.
    """
    with tf.name_scope('sample_sequence'):
        def body(past, prev, output):
            next_outputs = step(hparams, prev, past=past)
            
            
            logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)
            
            
            logits = top_k_logits(logits, k=top_k)
            samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
            return [
                next_outputs['presents'] if past is None else tf.concat([past, next_outputs['presents']], axis=-2),
                samples,
                tf.concat([output, samples], axis=1)
            ]

        past, prev, output = body(None, context, context)
        def cond(*args):
            return True

        memo_, curre_, tokens = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length - 1,
            loop_vars=[
                past,
                prev,
                output
            ],
            shape_invariants=[
                tf.TensorShape(Transformer.past_shape(hparams=hparams, batch_size=batch_size)),
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([batch_size, None]),
            ],
            back_prop=False,
        )

        # tokens: outputs, curre_: current word choice, memo_: the previous word sequence.
        return tokens, curre_, memo_  

   
reduce_dict = create_domain_dict(encoder_path, domain_dict)

