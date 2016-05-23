from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import tensorflow as tf

from six.moves import xrange
from six.moves import zip
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope



from tensorflow.models.rnn.translate import data_utils


"""
##The model is based on the standard seq2seq and anti-LM model from the paper "A Diversity-Promoting Objective Function for Neural Conversation Models"(2016) by Li et al.
##The following code mostly mimicks seq2seq_model.py from tensorflow library(https://www.tensorflow.org/)
"""

def MMIloss(logits, targets, weights, lam, gam,
                             average_across_timesteps=True,
                             softmax_loss_function=None, name=None):
    
"""lam is lambda value(diversity penalty) of the object, gam is gamma value(length penalty) of the object
(see section 4.5.1 of Li et al)"""


  if len(targets) != len(logits) or len(weights) != len(logits):
    raise ValueError("Lengths of logits, weights, and targets must be the same "
                     "%d, %d, %d." % (len(logits), len(weights), len(targets)))
  
  with ops.op_scope(logits + targets + weights, name,
                    "sequence_loss_by_example"):
    log_perp_list = []
    for logit, target, weight in zip(logits, targets, weights):
        if softmax_loss_function is None:
        
            target = array_ops.reshape(target, [-1])
            crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
            logit, target)
        else:
            crossent = softmax_loss_function(logit, target)
        log_perp_list.append(crossent * weight)
    log_perps = math_ops.add_n(log_perp_list)
    if average_across_timesteps:
        total_size = math_ops.add_n(weights)
        total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
        log_perps /= total_size
        
       
    final_perps= log_perps - (lam)*lm_perps + (gam)*len(targets)   
    return final_perps

def MMIloss_with_bucket(encoder_inputs, decoder_inputs, targets, weights,
                       buckets, seq2seq, softmax_loss_function=None,
                       per_example_loss=False, name=None):
    
     
    if len(encoder_inputs) < buckets[-1][0]:
        raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                       "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
    if len(targets) < buckets[-1][1]:
        raise ValueError("Length of targets (%d) must be at least that of last"
                     "bucket (%d)." % (len(targets), buckets[-1][1]))
    if len(weights) < buckets[-1][1]:
        raise ValueError("Length of weights (%d) must be at least that of last"
                     "bucket (%d)." % (len(weights), buckets[-1][1]))

    all_inputs = encoder_inputs + decoder_inputs + targets + weights
    losses = []
    outputs = []
    with ops.op_scope(all_inputs, name, "model_with_buckets"):
        for j, bucket in enumerate(buckets):
            with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                         reuse=True if j > 0 else None):
                bucket_outputs, _ = seq2seq(encoder_inputs[:bucket[0]],
                                    decoder_inputs[:bucket[1]])
                outputs.append(bucket_outputs)
                if per_example_loss:
                    losses.append(MMIloss(
              outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
              softmax_loss_function=softmax_loss_function))
                
    return outputs, losses


