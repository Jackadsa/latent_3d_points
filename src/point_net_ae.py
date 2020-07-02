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
from . tf_sinkhorn import ground_distance_tf, ground_distance_tf_nograd, sinkhorn_knopp_tf, sinkhorn_knopp_tf_64

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
        if not c.exists_and_is_not_none('output_pt'):
            c.output_pt = False
        self.configuration = c

        VarAutoEncoder.__init__(self, name, graph, configuration)
        self.encoder = c.encoder(self.x, **c.encoder_args)

        with tf.variable_scope(name):
            self.z_mu, self.z_log_var, self.z = self.encoder
            self.bottleneck_size = int(self.z.get_shape()[1])
            if c.output_pt:
                self.pt_out, layer = c.decoder(self.z, **c.decoder_args)
            else:
                layer = c.decoder(self.z, **c.decoder_args)
            
            
            if c.exists_and_is_not_none('close_with_tanh'):
                layer = tf.nn.tanh(layer)*tf.stack([tf.fill(tf.shape(layer)[:-1],np.float32(100)), tf.fill(tf.shape(layer)[:-1],np.float32(100))],axis=-1)

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

#     def return_loss_external(self,epsilon=1e-5):
#         c = self.configuration
#         in_locations = self.gt[:,:,1:]

#         @tf.custom_gradient
#         def return_loss(pt_out, x_out):
# #                 pt_out, x_out = xs
#             ground_distance, ground_dist_gradient = ground_distance_tf(in_locations,x_out)
#             self.out_weights = pt_out
#             match = sinkhorn_knopp_tf(self.gt[:,:,0], self.out_weights, tf.stop_gradient(ground_distance), c.sinkhorn_reg, numItermax=c.numItermax, stopThr=c.stopThr, adaptive_min=c.adaptive_min)

#             self.ground_distance = ground_distance
#             self.ground_grad = ground_dist_gradient
#             self.match = match

#             recon_loss = tf.trace(tf.matmul(tf.stop_gradient(match),ground_distance,transpose_b=True))

#             def grad(dL):
#                 aones = tf.fill(tf.shape(self.gt[:,:,0]),np.float32(1.))
#                 bones = tf.fill(tf.shape(pt_out),np.float32(1.))

#                 Mnew = tf.transpose(ground_distance,perm=[0,2,1])

#                 T = tf.transpose(match,perm=[0,2,1])
#                 Ttilde = T[:,:,:-1]

#                 L = T * Mnew
#                 Ltilde = L[:,:,:-1]

#                 D1 = tf.matrix_diag(tf.reduce_sum(T,axis=-1))
#                 D2 = tf.matrix_diag(1/(tf.reduce_sum(Ttilde,axis=-2) + epsilon)) # Add epsilon to insure invertibility

#                 H = D1 - tf.matmul(tf.matmul(Ttilde,D2),Ttilde,transpose_b=True) + epsilon* tf.eye(num_rows = tf.shape(bones)[-1],batch_shape = [tf.shape(bones)[0]]) # Add small diagonal piece to make sure H is invertible in edge cases.

#                 f = - tf.reduce_sum(L,axis=-1) + tf.squeeze(tf.matmul(tf.matmul(Ttilde,D2),tf.expand_dims(tf.reduce_sum(Ltilde,axis=-2),-1)),axis=-1)
#                 g = tf.squeeze(tf.matmul(tf.linalg.inv(H),tf.expand_dims(f,-1)),axis=-1)

#                 grad_pT = g - bones*tf.expand_dims(tf.reduce_sum(g,axis=-1),-1)/tf.cast(tf.shape(bones)[1],tf.float32)

#                 grad_x_out = tf.gradients(recon_loss,x_out)[0]

#                 return [-tf.expand_dims(dL,-1) * grad_pT, tf.expand_dims(tf.expand_dims(dL,-1),-1)*grad_x_out]

#             return recon_loss, grad
#         #self.recon_loss_temp = return_loss(self.x_reconstr)
#         return tf.reduce_mean(return_loss(self.pt_out,self.x_reconstr),axis=0)
            
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
            
#             @tf.custom_gradient
            def return_loss(x_out):     
                ground_distance, ground_dist_gradient = ground_distance_tf(in_locations,x_out)
                if c.output_pt:
                    self.out_weights = self.pt_out
                    match = sinkhorn_knopp_tf(self.gt[:,:,0], self.out_weights, tf.stop_gradient(ground_distance), c.sinkhorn_reg, numItermax=c.numItermax, stopThr=c.stopThr, adaptive_min=c.adaptive_min)
                else:
                    self.out_weights = tf.fill(tf.shape(x_out)[:-1],1./self.n_output[0])
                    match = tf.stop_gradient(sinkhorn_knopp_tf(self.gt[:,:,0], self.out_weights, ground_distance, c.sinkhorn_reg, numItermax=c.numItermax, stopThr=c.stopThr, adaptive_min=c.adaptive_min))
                self.ground_distance = ground_distance
                self.ground_grad = ground_dist_gradient
                self.match = match
                

                return tf.trace(tf.matmul(match,ground_distance,transpose_b=True))
            #self.recon_loss_temp = return_loss(self.x_reconstr)
            self.recon_loss = tf.reduce_mean(return_loss(self.x_reconstr),axis=0)

        elif c.loss == 'pot_tf_pt':
            in_locations = self.gt[:,:,1:]
            
            @tf.custom_gradient
            def return_loss(pt_out, x_out):
                epsilon = np.float64(1e-10)
#                 pt_out, x_out = xs
                #ground_distance, ground_dist_gradient = ground_distance_tf(in_locations,x_out)
                ground_distance = ground_distance_tf_nograd(in_locations,x_out)
                self.out_weights = pt_out
                match = sinkhorn_knopp_tf_64(self.gt[:,:,0], self.out_weights, tf.stop_gradient(ground_distance), c.sinkhorn_reg, numItermax=c.numItermax, stopThr=c.stopThr, adaptive_min=c.adaptive_min)

#                 self.ground_distance = ground_distance
#                 self.ground_grad = ground_dist_gradient
#                 self.match = match
                
                recon_loss = tf.trace(tf.matmul(tf.stop_gradient(tf.cast(match,tf.float32)),ground_distance,transpose_b=True))
                
                def grad(dL):
                    aones = tf.fill(tf.shape(self.gt[:,:,0]),np.float64(1.))
                    bones = tf.fill(tf.shape(pt_out),np.float64(1.))

                    Mnew = tf.cast(tf.transpose(ground_distance,perm=[0,2,1]),tf.float64)

                    T = tf.cast(tf.transpose(match,perm=[0,2,1]),tf.float64)
                    Ttilde = T[:,:,:-1]

                    L = T * Mnew
                    Ltilde = L[:,:,:-1]

                    D1 = tf.matrix_diag(tf.reduce_sum(T,axis=-1))
                    D2 = tf.matrix_diag(1/(tf.reduce_sum(Ttilde,axis=-2) + np.float64(1e-100))) # Add epsilon to ensure invertibility

                    H = D1 - tf.matmul(tf.matmul(Ttilde,D2),Ttilde,transpose_b=True) + epsilon* tf.eye(num_rows = tf.shape(bones)[-1],batch_shape = [tf.shape(bones)[0]],dtype=tf.float64) # Add small diagonal piece to make sure H is invertible in edge cases.

                    f = - tf.reduce_sum(L,axis=-1) + tf.squeeze(tf.matmul(tf.matmul(Ttilde,D2),tf.expand_dims(tf.reduce_sum(Ltilde,axis=-2),-1)),axis=-1)
                    g = tf.squeeze(tf.matmul(tf.linalg.inv(H),tf.expand_dims(f,-1)),axis=-1)

                    grad_pT = g - bones*tf.expand_dims(tf.reduce_sum(g,axis=-1),-1)/tf.cast(tf.shape(bones)[1],tf.float64)
                    
                    grad_x_out = tf.gradients(recon_loss,x_out)[0]
                    
                    return [-tf.expand_dims(dL,-1) * tf.cast(grad_pT,tf.float32),
                            tf.expand_dims(tf.expand_dims(dL,-1),-1)*tf.cast(grad_x_out,tf.float32)]

                return recon_loss, grad
            
            self.recon_loss = tf.reduce_mean(return_loss(self.pt_out,self.x_reconstr),axis=0)

            
        elif c.loss == 'pot_tf_pt_sqr':
            in_locations = self.gt[:,:,1:]
            
            @tf.custom_gradient
            def return_loss(pt_out, x_out):
                epsilon = np.float64(1e-10)
#                 pt_out, x_out = xs
                #ground_distance, ground_dist_gradient = ground_distance_tf(in_locations,x_out)
                ground_distance = ground_distance_tf_nograd(in_locations,x_out)
                self.out_weights = pt_out
                match = sinkhorn_knopp_tf_64(self.gt[:,:,0], self.out_weights, tf.stop_gradient(ground_distance), c.sinkhorn_reg, numItermax=c.numItermax, stopThr=c.stopThr, adaptive_min=c.adaptive_min)

#                 self.ground_distance = ground_distance
#                 self.ground_grad = ground_dist_gradient
#                 self.match = match
                
                recon_loss = tf.trace(tf.matmul(tf.stop_gradient(tf.cast(match,tf.float32)),ground_distance,transpose_b=True))
                
                def grad(dL):
                    aones = tf.fill(tf.shape(self.gt[:,:,0]),np.float64(1.))
                    bones = tf.fill(tf.shape(pt_out),np.float64(1.))

                    Mnew = tf.cast(tf.transpose(ground_distance,perm=[0,2,1]),tf.float64)

                    T = tf.cast(tf.transpose(match,perm=[0,2,1]),tf.float64)
                    Ttilde = T[:,:,:-1]

                    L = T * Mnew
                    Ltilde = L[:,:,:-1]

                    D1 = tf.matrix_diag(tf.reduce_sum(T,axis=-1))
                    D2 = tf.matrix_diag(1/(tf.reduce_sum(Ttilde,axis=-2) + np.float64(1e-100))) # Add epsilon to ensure invertibility

                    H = D1 - tf.matmul(tf.matmul(Ttilde,D2),Ttilde,transpose_b=True) + epsilon* tf.eye(num_rows = tf.shape(bones)[-1],batch_shape = [tf.shape(bones)[0]],dtype=tf.float64) # Add small diagonal piece to make sure H is invertible in edge cases.

                    f = - tf.reduce_sum(L,axis=-1) + tf.squeeze(tf.matmul(tf.matmul(Ttilde,D2),tf.expand_dims(tf.reduce_sum(Ltilde,axis=-2),-1)),axis=-1)
                    g = tf.squeeze(tf.matmul(tf.linalg.inv(H),tf.expand_dims(f,-1)),axis=-1)

                    grad_pT = g - bones*tf.expand_dims(tf.reduce_sum(g,axis=-1),-1)/tf.cast(tf.shape(bones)[1],tf.float64)
                    
                    grad_x_out = tf.gradients(recon_loss,x_out)[0]
                    
                    return [-tf.expand_dims(dL,-1) * tf.cast(grad_pT,tf.float32),
                            tf.expand_dims(tf.expand_dims(dL,-1),-1)*tf.cast(grad_x_out,tf.float32)]

                return recon_loss, grad
            
            self.recon_loss = tf.reduce_mean(tf.square(return_loss(self.pt_out,self.x_reconstr)),axis=0)
            
            
        reg_losses = self.graph.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if c.exists_and_is_not_none('w_reg_alpha'):
            w_reg_alpha = c.w_reg_alpha
        else:
            w_reg_alpha = 1.0

        for rl in reg_losses:
            self.loss +=  (w_reg_alpha * rl)
        
        if c.exists_and_is_not_none('rescale_logvar'):
            if c.rescale_logvar == True:
                logvar_cond = tf.greater(self.z_log_var,0.)
                var_terms_rescaled = tf.where(logvar_cond,
                                              #tf.log(self.z_log_var + 1.) - self.z_log_var - 1., get stupid NaNs on gradient when z_log_var = -1
                                              - 0.5*tf.square(self.z_log_var) - 1.,
                                              self.z_log_var - tf.exp(self.z_log_var))
                self.kl_loss = tf.reduce_mean(- 0.5 * tf.reduce_sum(1 + var_terms_rescaled - tf.square(self.z_mu), axis=-1))
            else:
                self.kl_loss = tf.reduce_mean(- 0.5 * tf.reduce_sum(1 + self.z_log_var - tf.square(self.z_mu) - tf.exp(self.z_log_var), axis=-1))
        else:
            self.kl_loss = tf.reduce_mean(- 0.5 * tf.reduce_sum(1 + self.z_log_var - tf.square(self.z_mu) - tf.exp(self.z_log_var), axis=-1))
                
        if c.loss == 'pot_tf_pt_sqr':
            self.loss = self.kl_loss  + self.recon_loss / self.beta
        else:
            self.loss = self.beta * self.kl_loss +  self.recon_loss

    def _setup_optimizer(self):
        c = self.configuration
#         self.lr = c.learning_rate
#         if hasattr(c, 'exponential_decay'):
#             self.lr = tf.train.exponential_decay(c.learning_rate, self.epoch, c.decay_steps, decay_rate=0.5, staircase=True, name="learning_rate_decay")
#             self.lr = tf.maximum(self.lr, 1e-5)
#             tf.summary.scalar('learning_rate', self.lr)
            
        if c.exists_and_is_not_none('reduceLRonplateau'):
            if c.reduceLRonplateau:
                self.lr = tf.placeholder(tf.float32)
                self.current_lr = c.learning_rate
            else:
                self.lr = c.learning_rate
        else:
            self.lr = c.learning_rate

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        
        
#         if c.loss is "pot_tf":
#             self.train_step = self.optimizer.minimize(self.loss, grad_loss=self.loss_gradient)
    #         else:
        #self.train_step = self.optimizer.minimize(self.loss)
        
        self.gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
        self.gradients, _ = tf.clip_by_global_norm(self.gradients, 1.0)
        self.train_step = self.optimizer.apply_gradients(zip(self.gradients, variables))

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