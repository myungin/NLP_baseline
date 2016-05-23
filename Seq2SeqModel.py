from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import tensorflow as tf

from six.moves import xrange
from six.moves import zip

import MMIloss

class Model(object):
    
"""

1.This model uses bucket due to inconsitency between input seqeunce length and target sequence length
2.This model enables to use sampled softmax(described in "http://arxiv.org/abs/1412.2007") if we want
3.model_type can be either 'standard SEQ2SEQ' or 'MMI-antiLM'

"""
    
    def __init__(self, source_vocab_size, target_vocab_size, buckets, size, num_layers, max_gradient_norm=1, batch_size=256, learning_rate=0.1, learning_rate_decay_factor =1, sampled_softmax_use=True, num_samples=512, standard_SEQ2SEQ=True):
        
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step=tf.Variable(0,  trainable=False)
        
        #sampled softmax 
        if sampled_softmax_use:
            output_projcetion = None
            softmax_loss_fuction = None
            
            if num_samples > 0 and num_samples < self.target_vocab_size:
                
                with tf.device("/cpu:0"):
                    w=tf.get_variable("proj_w", [size, self.target_vocab_size])
                    wt=tf.transpose(w)
                    b= tf.get_variable("proj_b", [self.target_vocab_size])
                    
                output_projection = (w, b)
                
                def sampled_loss(inputs, labels):
                    
                    with tf.deivce("/cpu:0"):
                        labels= tf.reshape(labels, [-1, 1])
                        return tf.nn.sampled_softmax_loss(wt, b, inputs, labels, num_samples, self.target_vocab_size)
                
                
                softmax_loss_function =  sampled_loss
                
        #creating multi-layered(deep) LSTM cell for our model
        
        mycell= tf.nn.rnn_cell.BasicLSTMCell(size)
        
        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([mycell] * num_layers)
        else:
            cell = mycell
          
        #seq2seq function(Note: we do not use attention based one)
        
        def seq2seq_func(encoder_inputs, decoder_inputs, do_decode=True):
            
            return tf.nn.seq2seq.embedding_rnn_seq2seq(encoder_inputs, decoder_inputs, cell, source_vocab_size, target_vocab_size, embedding_size=size, output_projection=output_projcetion, feed_previous=do_decode)
        
        #feeding for inputs
        
        self.encoder_inputs= []
        self.decoder_inputs= []
        self.target_inputs= []
        
        for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="encoder{0}".format(i)))
        for i in xrange(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                name="weight{0}".format(i)))

        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i + 1] for i in xrange(len(self.decoder_inputs) - 1)]
        
        
        self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
        self.encoder_inputs, self.decoder_inputs, targets,
        self.target_weights, buckets, lambda x, y: seq2seq_func(x, y),
        softmax_loss_function=softmax_loss_function)
        
        if not standard_SEQ2SEQ:
            self.outputs, self.losses = MMIloss_with_buckets(
        self.encoder_inputs, self.decoder_inputs, targets,
        self.target_weights, buckets, lambda x, y: seq2seq_func(x, y),
        softmax_loss_function=softmax_loss_function)
            
            
        
        
        # If we use output projection, we need to project outputs for decoding.
        if output_projection is not None:
            for b in xrange(len(buckets)):
                self.outputs[b] = [tf.matmul(output, output_projection[0]) + output_projection[1] for output in self.outputs[b]]
                
        
        
        params = tf.trainable_variables()
        
        self.saver = tf.train.saver(tf.all_Variables())
    
    #Running a step of the model by feeding the inputs
    def step(self, session, encoder_inputs, decoder_inputs, target_weights, bucket_id):
        
        input_feed = {} #input feeding dictionary
        
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]
            
        
        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)
            
        output_feed = [self.losses[bucket_id]]  
        for l in xrange(decoder_size):  
            output_feed.append(self.outputs[bucket_id][l])
        
        
        outputs= session.run(output_feed, input_feed)
        
        
        return outputs[0], outputs[1:]
    
    
     def get_batch(self, data, bucket_id):
            
        encoder_size, decoder_size = self.buckets[bucke_id]
            
        encoder_inputs, decoder_inputs = [], []
            
        #Get a random batch of encoder and decoder inputs from data.
        # pad them if needed and GO to decoder.
        for _ in xrange(self.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])
                
            encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(encoder_input+ encoder_pad))
                
        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

  
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in xrange(self.batch_size):
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
            return batch_encoder_inputs, batch_decoder_inputs, batch_weights  
          
       
    
        
        
        
        
        
        
        


           