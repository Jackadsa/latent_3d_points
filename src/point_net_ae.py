'''
Created on January 26, 2017

@author: optas
'''

import time
import tensorflow as tf
import os.path as osp

from tflearn.layers.conv import conv_1d
from tflearn.layers.core import fully_connected

from . in_out import create_dir
from . autoencoder import AutoEncoder, VarAutoEncoder
from . general_utils import apply_augmentations
from . tf_sinkhorn import ground_distance_tf, sinkhorn_knopp_tf

from tensorflow.python.framework import ops

try:    
    from .. external.structural_losses.tf_nndistance import nn_distance
except:
    print('Chamfer Losses cannot be loaded. Please install them first.')
try:
    from .. external.structural_losses.tf_approxmatch import approx_match, match_cost
except:
    print('EMD Losses cannot be loaded. Please install them first.')
    
import ot
import numpy as np
    

class PointNetVarAutoEncoder(VarAutoEncoder):
    '''
    An Auto-Encoder for point-clouds.
    '''

    def __init__(self, name, configuration, graph=None):
        c = configuration
        self.configuration = c

        VarAutoEncoder.__init__(self, name, graph, configuration)
        self.encoder = c.encoder(self.x, **c.encoder_args)

        with tf.variable_scope(name):
            self.z_mu, self.z_log_var, self.z = self.encoder
            self.bottleneck_size = int(self.z.get_shape()[1])
            layer = c.decoder(self.z, **c.decoder_args)
            
            
            if c.exists_and_is_not_none('close_with_tanh'):
                tf.nn.tanh(layer)*tf.stack([tf.fill(tf.shape(layer)[:-1],np.float32(100)), tf.fill(tf.shape(layer)[:-1],np.float32(100))],axis=-1)

            #self.x_reconstr = tf.reshape(layer, [-1, self.n_output[0], self.n_output[1]])
            self.x_reconstr = layer
            
            #self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=c.saver_max_to_keep)

            self._create_loss()
            self._setup_optimizer()

            # GPU configuration
            if hasattr(c, 'allow_gpu_growth'):
                growth = c.allow_gpu_growth
            else:
                growth = True

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = growth

            # Summaries
            self.merged_summaries = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(osp.join(configuration.train_dir, 'summaries'), self.graph)

            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=c.saver_max_to_keep)
            
            # Initializing the tensor flow variables
            self.init = tf.global_variables_initializer()

            # Launch the session
            self.sess = tf.Session(config=config)
            self.sess.run(self.init)

    def _create_loss(self):
        c = self.configuration

        if c.loss == 'chamfer':
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(self.x_reconstr, self.gt)
            self.loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
        elif c.loss == 'emd':
            match = approx_match(self.x_reconstr, self.gt)
            self.loss = tf.reduce_mean(match_cost(self.x_reconstr, self.gt, match))
        
        elif c.loss == 'pot_tf':
            in_locations = self.gt[:,:,1:]
            
            @tf.custom_gradient
            def return_loss(x_out):     
                ground_distance, ground_dist_gradient = ground_distance_tf(in_locations,x_out)
            #self.out_weights = tf.placeholder(tf.int32,shape=([None] + self.n_output)[:-1])
                self.out_weights = tf.fill(tf.shape(x_out)[:-1],1./self.n_output[0])
                match = sinkhorn_knopp_tf(self.gt[:,:,0], self.out_weights, ground_distance, c.sinkhorn_reg, numItermax=c.numItermax, stopThr=c.stopThr)
                self.ground_distance = ground_distance
                self.ground_grad = ground_dist_gradient
                self.match = match
                
                def grad(dL):
                    ground_dist_gradient_perm = tf.transpose(ground_dist_gradient,[0,3,1,2])
                    loss_grad_temp = tf.matrix_diag_part(tf.matmul(tf.tile(tf.expand_dims(match,1),[1,2,1,1]),ground_dist_gradient_perm,transpose_a = True))
                    return tf.transpose(loss_grad_temp,[0,2,1])  
                
                return tf.trace(tf.matmul(match,ground_distance,transpose_b=True)), grad
            

            self.recon_loss = tf.reduce_mean(return_loss(self.x_reconstr),axis=0)


        reg_losses = self.graph.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if c.exists_and_is_not_none('w_reg_alpha'):
            w_reg_alpha = c.w_reg_alpha
        else:
            w_reg_alpha = 1.0

        for rl in reg_losses:
            self.loss +=  (w_reg_alpha * rl)
        
        self.kl_loss = tf.reduce_mean(- 0.5 * self.beta * tf.reduce_sum(1 + self.z_log_var - tf.square(self.z_mu) - tf.exp(self.z_log_var), axis=-1))
        self.loss = self.kl_loss + self.recon_loss

    def _setup_optimizer(self):
        c = self.configuration
        self.lr = c.learning_rate
        if hasattr(c, 'exponential_decay'):
            self.lr = tf.train.exponential_decay(c.learning_rate, self.epoch, c.decay_steps, decay_rate=0.5, staircase=True, name="learning_rate_decay")
            self.lr = tf.maximum(self.lr, 1e-5)
            tf.summary.scalar('learning_rate', self.lr)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        
        
#         if c.loss is "pot_tf":
#             self.train_step = self.optimizer.minimize(self.loss, grad_loss=self.loss_gradient)
    #         else:
        #self.train_step = self.optimizer.minimize(self.loss)
        
        gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.train_step = self.optimizer.apply_gradients(zip(gradients, variables))

    def _single_epoch_train(self, train_data, configuration, only_fw=False, separate_loss = False):
        n_examples = train_data.num_examples
        epoch_loss = 0.
        epoch_recon_loss = 0.
        epoch_kl_loss = 0.
        batch_size = configuration.batch_size
        n_batches = int(n_examples / batch_size)
        start_time = time.time()

        if only_fw:
            fit = self.reconstruct
        else:
            fit = self.partial_fit

        # Loop over all batches
        for _ in xrange(n_batches):

            if self.is_denoising:
                original_data, _, batch_i = train_data.next_batch(batch_size)
                if batch_i is None:  # In this case the denoising concern only the augmentation.
                    batch_i = original_data
            else:
                batch_i, _, _ = train_data.next_batch(batch_size)

            batch_i = apply_augmentations(batch_i, configuration)   # This is a new copy of the batch.

            if separate_loss:
                if self.is_denoising:
                    _, loss, recon_loss, kl_loss = fit(batch_i, GT=original_data, beta=configuration.beta, separate_loss = separate_loss)
                else:
                    _, loss, recon_loss, kl_loss = fit(batch_i,beta=configuration.beta, separate_loss = separate_loss)
            else:
                if self.is_denoising:
                    _, loss = fit(batch_i, GT=original_data, beta=configuration.beta)
                else:
                    _, loss = fit(batch_i,beta=configuration.beta)

            # Compute average loss
            epoch_loss += loss
            epoch_recon_loss += recon_loss
            epoch_kl_loss += kl_loss
        epoch_loss /= n_batches
        if separate_loss:
            epoch_recon_loss /= n_batches
            epoch_kl_loss /= n_batches
        duration = time.time() - start_time
        
        if configuration.loss == 'emd':
            epoch_loss /= len(train_data.point_clouds[0])
        
        if separate_loss:
            return epoch_loss, epoch_recon_loss, epoch_kl_loss, duration
        else:
            return epoch_loss, duration

    def gradient_of_input_wrt_loss(self, in_points, gt_points=None):
        if gt_points is None:
            gt_points = in_points
        return self.sess.run(tf.gradients(self.loss, self.x), feed_dict={self.x: in_points, self.gt: gt_points})
    
    def get_beta_current(self, conf):
        return 0
        

class PointNetAutoEncoder(AutoEncoder):
    '''
    An Auto-Encoder for point-clouds.
    '''

    def __init__(self, name, configuration, graph=None):
        c = configuration
        self.configuration = c

        AutoEncoder.__init__(self, name, graph, configuration)

        with tf.variable_scope(name):
            self.z = c.encoder(self.x, **c.encoder_args)
            self.bottleneck_size = int(self.z.get_shape()[1])
            layer = c.decoder(self.z, **c.decoder_args)
            
            if c.exists_and_is_not_none('close_with_tanh'):
                layer = tf.nn.tanh(layer)

            self.x_reconstr = tf.reshape(layer, [-1, self.n_output[0], self.n_output[1]])
            
            
            #self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=c.saver_max_to_keep)
            
            self.saver = tf.train.Saver(max_to_keep=c.saver_max_to_keep)

            self._create_loss()
            self._setup_optimizer()

            # GPU configuration
            if hasattr(c, 'allow_gpu_growth'):
                growth = c.allow_gpu_growth
            else:
                growth = True

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = growth

            # Summaries
            self.merged_summaries = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(osp.join(configuration.train_dir, 'summaries'), self.graph)

            # Initializing the tensor flow variables
            self.init = tf.global_variables_initializer()

            # Launch the session
            self.sess = tf.Session(config=config)
            self.sess.run(self.init)

    def _create_loss(self):
        c = self.configuration

        if c.loss == 'chamfer':
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(self.x_reconstr, self.gt)
            self.loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
        elif c.loss == 'emd':
            match = approx_match(self.x_reconstr, self.gt)
            self.loss = tf.reduce_mean(match_cost(self.x_reconstr, self.gt, match))

        reg_losses = self.graph.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if c.exists_and_is_not_none('w_reg_alpha'):
            w_reg_alpha = c.w_reg_alpha
        else:
            w_reg_alpha = 1.0

        for rl in reg_losses:
            self.loss += (w_reg_alpha * rl)

    def _setup_optimizer(self):
        c = self.configuration
        self.lr = c.learning_rate
        if hasattr(c, 'exponential_decay'):
            self.lr = tf.train.exponential_decay(c.learning_rate, self.epoch, c.decay_steps, decay_rate=0.5, staircase=True, name="learning_rate_decay")
            self.lr = tf.maximum(self.lr, 1e-5)
            tf.summary.scalar('learning_rate', self.lr)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        if c.loss == 'pot_tf':
            self.train_step = self.optimizer.minimize(self.loss,grad_loss=self.gradient_tensor)
        self.train_step = self.optimizer.minimize(self.loss)

    def _single_epoch_train(self, train_data, configuration, only_fw=False):
        n_examples = train_data.num_examples
        epoch_loss = 0.
        batch_size = configuration.batch_size
        n_batches = int(n_examples / batch_size)
        start_time = time.time()

        if only_fw:
            fit = self.reconstruct
        else:
            fit = self.partial_fit

        # Loop over all batches
        for _ in xrange(n_batches):

            if self.is_denoising:
                original_data, _, batch_i = train_data.next_batch(batch_size)
                if batch_i is None:  # In this case the denoising concern only the augmentation.
                    batch_i = original_data
            else:
                batch_i, _, _ = train_data.next_batch(batch_size)

            batch_i = apply_augmentations(batch_i, configuration)   # This is a new copy of the batch.

            if self.is_denoising:
                _, loss = fit(batch_i, GT=original_data)
            else:
                _, loss = fit(batch_i)

            # Compute average loss
            epoch_loss += loss
        epoch_loss /= n_batches
        duration = time.time() - start_time
        
        if configuration.loss == 'emd':
            epoch_loss /= len(train_data.point_clouds[0])
        
        return epoch_loss, duration

    def gradient_of_input_wrt_loss(self, in_points, gt_points=None):
        if gt_points is None:
            gt_points = in_points
        return self.sess.run(tf.gradients(self.loss, self.x), feed_dict={self.x: in_points, self.gt: gt_points})