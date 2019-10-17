'''
Created on February 4, 2017

@author: optas

'''

'''
Edited August/Sept 2019

@author: Jack Collins
'''

import tensorflow as tf
import numpy as np
import warnings

from tflearn.layers.core import fully_connected, dropout
from tflearn.layers.conv import conv_1d, avg_pool_1d
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.core import reshape

#from tf.keras.layers import BatchNormalization

from . tf_utils import expand_scope_by_name, replicate_parameter_for_all_layers

def variational_encoder_with_convs_and_symmetry(in_signal, n_filters=[64, 128, 256, 1024], filter_sizes=[1], strides=[1],
                                        b_norm=True, non_linearity=tf.nn.relu, regularizer=None, weight_decay=0.001,
                                        symmetry=tf.reduce_max, dropout_prob=None, pool=avg_pool_1d, pool_sizes=None, scope=None,
                                        reuse=False, padding='same', verbose=False, latent_size=1, conv_op=conv_1d,
                                        fully_connected_layers = None,
                                        weights_init = 'xavier',
                                        bias_init = 'xavier'):
    '''An Encoder (recognition network), which maps inputs onto a latent space.
    '''
    
    if verbose:
        print 'Building Encoder'

    n_layers = len(n_filters)
    filter_sizes = replicate_parameter_for_all_layers(filter_sizes, n_layers)
    strides = replicate_parameter_for_all_layers(strides, n_layers)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    if n_layers < 2:
        raise ValueError('More than 1 layers are expected.')

    for i in xrange(n_layers):
        if i == 0:
            layer = in_signal

        name = 'encoder_conv_layer_' + str(i)
        scope_i = expand_scope_by_name(scope, name)
        layer = conv_op(layer, nb_filter=n_filters[i], filter_size=filter_sizes[i], strides=strides[i], regularizer=regularizer,
                        weight_decay=weight_decay, name=name, reuse=reuse, scope=scope_i, padding=padding,
                        weights_init = weights_init, bias_init = bias_init)

        if verbose:
            print name, 'conv params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

        if b_norm:
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            #layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            layer = tf.keras.layers.BatchNormalization(axis=-1,renorm=True,name=name)(layer)
#             if verbose:
#                 print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())

        if non_linearity is not None:
            layer = non_linearity(layer)

        if pool is not None and pool_sizes is not None:
            if pool_sizes[i] is not None:
                layer = pool(layer, kernel_size=pool_sizes[i])

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        if verbose:
            print layer
            print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

    if symmetry is not None:
        layer = symmetry(layer, axis=1)
        if verbose:
            print 'Symmetry Layer:'
            print layer, '\n'

    if fully_connected_layers is not None:
        for i, layer_size in enumerate(fully_connected_layers):
            name = 'encoder_dense_layer_' + str(i)
            scope_i = expand_scope_by_name(scope, name)
            
            layer = fully_connected(layer, layer_size, activation='linear', weights_init='xavier', name=name, regularizer=regularizer, 
                                    weight_decay=weight_decay, reuse=reuse, scope=scope_i)
            
            
            if verbose:
                print name, 'FC params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

            if b_norm:
                name += '_bnorm'
                scope_i = expand_scope_by_name(scope, name)
                #layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
                layer = tf.keras.layers.BatchNormalization(axis=-1,renorm=True,name=name)(layer)
#                 if verbose:
#                     print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())
 
            if non_linearity is not None and i<len(fully_connected_layers)-1:
                layer = non_linearity(layer)

            if dropout_prob is not None and dropout_prob[i] > 0:
                layer = dropout(layer, 1.0 - dropout_prob[i])

            if verbose:
                print layer
                print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

                
    #Latent space  
    scope_i = expand_scope_by_name(scope, "latent encoding") 
    name = 'z_mean'
    z_mean = fully_connected(layer, latent_size, activation='linear', weights_init='xavier', name=name, reuse=reuse, scope=scope_i)
    name = 'z_log_var'
    z_log_var = fully_connected(layer, latent_size, activation='linear', weights_init='xavier', name=name, reuse=reuse, scope=scope_i)
                
    K = tf.keras.backend
    
    def sampling(args):
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.

        # Arguments
            args (tensor): mean and log of variance of Q(z|X)

        # Returns
            z (tensor): sampled latent vector
        """

        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    z = tf.keras.layers.Lambda(sampling, name='z')([z_mean, z_log_var])
    
    print [z_mean, z_log_var, z]

    return [z_mean, z_log_var, z]


def variational_encoder_with_convs_and_symmetry_bnormafter(in_signal, n_filters=[64, 128, 256, 1024], filter_sizes=[1], strides=[1],
                                        b_norm=True, non_linearity=tf.nn.relu, regularizer=None, weight_decay=0.001,
                                        symmetry=tf.reduce_max, dropout_prob=None, pool=avg_pool_1d, pool_sizes=None, scope=None,
                                        reuse=False, padding='same', verbose=False, latent_size=1, conv_op=conv_1d,
                                        fully_connected_layers = None,
                                        weights_init = 'xavier',
                                        bias_init = 'xavier'):
    '''An Encoder (recognition network), which maps inputs onto a latent space.
    '''
    
    if verbose:
        print 'Building Encoder'

    n_layers = len(n_filters)
    filter_sizes = replicate_parameter_for_all_layers(filter_sizes, n_layers)
    strides = replicate_parameter_for_all_layers(strides, n_layers)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    if n_layers < 2:
        raise ValueError('More than 1 layers are expected.')

    for i in xrange(n_layers):
        if i == 0:
            layer = in_signal

        name = 'encoder_conv_layer_' + str(i)
        scope_i = expand_scope_by_name(scope, name)
        layer = conv_op(layer, nb_filter=n_filters[i], filter_size=filter_sizes[i], strides=strides[i], regularizer=regularizer,
                        weight_decay=weight_decay, name=name, reuse=reuse, scope=scope_i, padding=padding,
                        weights_init = weights_init, bias_init = bias_init)

        if verbose:
            print name, 'conv params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

        if non_linearity is not None:
            layer = non_linearity(layer)
            
        if b_norm:
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            #layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            layer = tf.keras.layers.BatchNormalization(axis=-1,renorm=True,name=name)(layer)
#             if verbose:
#                 print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())


        if pool is not None and pool_sizes is not None:
            if pool_sizes[i] is not None:
                layer = pool(layer, kernel_size=pool_sizes[i])

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        if verbose:
            print layer
            print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

    if symmetry is not None:
        layer = symmetry(layer, axis=1)
        if verbose:
            print 'Symmetry Layer:'
            print layer, '\n'

    if fully_connected_layers is not None:
        for i, layer_size in enumerate(fully_connected_layers):
            name = 'encoder_dense_layer_' + str(i)
            scope_i = expand_scope_by_name(scope, name)
            
            layer = fully_connected(layer, layer_size, activation='linear', weights_init='xavier', name=name, regularizer=regularizer, 
                                    weight_decay=weight_decay, reuse=reuse, scope=scope_i)
            
            
            if verbose:
                print name, 'FC params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),
                
            if non_linearity is not None and i<len(fully_connected_layers)-1:
                layer = non_linearity(layer)
                
            if b_norm:
                name += '_bnorm'
                scope_i = expand_scope_by_name(scope, name)
                #layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
                layer = tf.keras.layers.BatchNormalization(axis=-1,renorm=True,name=name)(layer)
#                 if verbose:
#                     print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())
 


            if dropout_prob is not None and dropout_prob[i] > 0:
                layer = dropout(layer, 1.0 - dropout_prob[i])

            if verbose:
                print layer
                print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

                
    #Latent space  
    scope_i = expand_scope_by_name(scope, "latent encoding") 
    name = 'z_mean'
    z_mean = fully_connected(layer, latent_size, activation='linear', weights_init='xavier', name=name, reuse=reuse, scope=scope_i)
    name = 'z_log_var'
    z_log_var = fully_connected(layer, latent_size, activation='linear', weights_init='xavier', name=name, reuse=reuse, scope=scope_i)
                
    K = tf.keras.backend
    
    def sampling(args):
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.

        # Arguments
            args (tensor): mean and log of variance of Q(z|X)

        # Returns
            z (tensor): sampled latent vector
        """

        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    z = tf.keras.layers.Lambda(sampling, name='z')([z_mean, z_log_var])
    
    print [z_mean, z_log_var, z]

    return [z_mean, z_log_var, z]



def encoder_with_convs_and_symmetry(in_signal, n_filters=[64, 128, 256, 1024], filter_sizes=[1], strides=[1],
                                        b_norm=True, non_linearity=tf.nn.relu, regularizer=None, weight_decay=0.001,
                                        symmetry=tf.reduce_max, dropout_prob=None, pool=avg_pool_1d, pool_sizes=None, scope=None,
                                        reuse=False, padding='same', verbose=False, closing=None, conv_op=conv_1d,
                                        fully_connected_layers = None):
    '''An Encoder (recognition network), which maps inputs onto a latent space.
    '''

    if verbose:
        print 'Building Encoder'

    n_layers = len(n_filters)
    filter_sizes = replicate_parameter_for_all_layers(filter_sizes, n_layers)
    strides = replicate_parameter_for_all_layers(strides, n_layers)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    if n_layers < 2:
        raise ValueError('More than 1 layers are expected.')

    for i in xrange(n_layers):
        if i == 0:
            layer = in_signal

        name = 'encoder_conv_layer_' + str(i)
        scope_i = expand_scope_by_name(scope, name)
        layer = conv_op(layer, nb_filter=n_filters[i], filter_size=filter_sizes[i], strides=strides[i], regularizer=regularizer,
                        weight_decay=weight_decay, name=name, reuse=reuse, scope=scope_i, padding=padding)

        if verbose:
            print name, 'conv params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

        if b_norm:
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            if verbose:
                print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())

        if non_linearity is not None:
            layer = non_linearity(layer)

        if pool is not None and pool_sizes is not None:
            if pool_sizes[i] is not None:
                layer = pool(layer, kernel_size=pool_sizes[i])

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        if verbose:
            print layer
            print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

    if symmetry is not None:
        layer = symmetry(layer, axis=1)
        if verbose:
            print 'Symmetry Layer:'
            print layer, '\n'

    if fully_connected_layers is not None:
        for i, layer_size in enumerate(fully_connected_layers):
            name = 'encoder_dense_layer_' + str(i)
            scope_i = expand_scope_by_name(scope, name)
            
            layer = fully_connected(layer, layer_size, activation='linear', weights_init='xavier', name=name, regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)
            
            
            if verbose:
                print name, 'FC params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

            if b_norm:
                name += '_bnorm'
                scope_i = expand_scope_by_name(scope, name)
                layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
                if verbose:
                    print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())
 
            if non_linearity is not None and i<len(fully_connected_layers)-1:
                layer = non_linearity(layer)

            if dropout_prob is not None and dropout_prob[i] > 0:
                layer = dropout(layer, 1.0 - dropout_prob[i])

            if verbose:
                print layer
                print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

       
    if closing is not None:
        name = 'encoder_closing_layer'
        scope_i = expand_scope_by_name(scope, name)
        layer = closing(layer)
        print layer

    return layer




def decoder_with_fc_only(latent_signal, layer_sizes=[], b_norm=True, non_linearity=tf.nn.relu,
                         regularizer=None, weight_decay=0.001, reuse=False, scope=None, dropout_prob=None,
                         b_norm_finish=False, verbose=False):
    '''A decoding network which maps points from the latent space back onto the data space.
    '''
    if verbose:
        print 'Building Decoder'

    n_layers = len(layer_sizes)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    if n_layers < 2:
        raise ValueError('For an FC decoder with single a layer use simpler code.')

    for i in xrange(0, n_layers - 1):
        name = 'decoder_fc_' + str(i)
        scope_i = expand_scope_by_name(scope, name)

        if i == 0:
            layer = latent_signal

        layer = fully_connected(layer, layer_sizes[i], activation='linear', weights_init='xavier', name=name, regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)

        if verbose:
            print name, 'FC params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

        if b_norm:
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            #layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            layer = tf.keras.layers.BatchNormalization(axis=-1,renorm=True,name=name)(layer)
            if verbose:
                print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())

        if non_linearity is not None:
            layer = non_linearity(layer)

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        if verbose:
            print layer
            print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

    # Last decoding layer never has a non-linearity.
    name = 'decoder_fc_' + str(n_layers - 1)
    scope_i = expand_scope_by_name(scope, name)
    layer = fully_connected(layer, layer_sizes[n_layers - 1], activation='linear', weights_init='xavier', name=name, regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)
    if verbose:
        print name, 'FC params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

    if b_norm_finish:
        name += '_bnorm'
        scope_i = expand_scope_by_name(scope, name)
        #layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
        layer = tf.keras.layers.BatchNormalization(axis=-1,renorm=True,name=name)(layer)
        if verbose:
            print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())

    if verbose:
        print layer
        print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

    return layer


def decoder_with_convs_only(in_signal, n_filters, filter_sizes, strides, padding='same', b_norm=True, non_linearity=tf.nn.relu,
                            conv_op=conv_1d, regularizer=None, weight_decay=0.001, dropout_prob=None, upsample_sizes=None,
                            b_norm_finish=False, scope=None, reuse=False, verbose=False,
                            fully_connected_layers = None, reshape_last_connected = None):

    if verbose:
        print 'Building Decoder'

    n_layers = len(n_filters)
    filter_sizes = replicate_parameter_for_all_layers(filter_sizes, n_layers)
    strides = replicate_parameter_for_all_layers(strides, n_layers)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    layer = in_signal
    
    if (fully_connected_layers is not None) and (reshape_last_connected is not None):
        for i, layer_size in enumerate(fully_connected_layers):
            name = 'decoder_dense_layer_' + str(i)
            scope_i = expand_scope_by_name(scope, name)
            
            layer = fully_connected(layer, layer_size, activation='linear', weights_init='xavier', name=name, regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)
            
            
            if verbose:
                print name, 'FC params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

            if b_norm:
                name += '_bnorm'
                scope_i = expand_scope_by_name(scope, name)
                #layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
                layer = tf.keras.layers.BatchNormalization(axis=-1,renorm=True,name=name, momentum=0.9,epsilon=1e-5)(layer)
#                 if verbose:
#                     print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())

            if non_linearity is not None:
                layer = non_linearity(layer)

            if dropout_prob is not None and dropout_prob[i] > 0:
                layer = dropout(layer, 1.0 - dropout_prob[i])

            if verbose:
                print layer
                print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'
                
        layer = reshape(layer,[-1,reshape_last_connected[0],reshape_last_connected[1]])
        if verbose:
            print layer
            print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

    
    for i in xrange(n_layers):

        name = 'decoder_conv_layer_' + str(i)
        scope_i = expand_scope_by_name(scope, name)

        layer = conv_op(layer, nb_filter=n_filters[i], filter_size=filter_sizes[i],
                        strides=strides[i], padding=padding, regularizer=regularizer, weight_decay=weight_decay,
                        name=name, reuse=reuse, scope=scope_i, weights_init='xavier')

        if verbose:
            print name, 'conv params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

        if (b_norm and i < n_layers - 1) or (i == n_layers - 1 and b_norm_finish):
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            #layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            layer = tf.keras.layers.BatchNormalization(axis=-1,renorm=True,name=name, momentum=0.9,epsilon=1e-5)(layer)
#             if verbose:
#                 print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())

        if non_linearity is not None and i < n_layers - 1:  # Last layer doesn't have a non-linearity.
            layer = non_linearity(layer)

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        if upsample_sizes is not None and upsample_sizes[i] is not None:
            layer = tf.tile(layer, multiples=[1, upsample_sizes[i], 1])

        if verbose:
            print layer
            print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

    return layer


def decoder_with_convs_and_fc(in_signal, n_filters, filter_sizes, strides, padding='same', b_norm=True, non_linearity=tf.nn.relu,
                            conv_op=conv_1d, regularizer=None, weight_decay=0.001, dropout_prob=None, upsample_sizes=None,
                            b_norm_finish=False, scope=None, reuse=False, verbose=False,
                            fully_connected_layers = None, reshape_last_connected = None):

    if verbose:
        print 'Building Decoder'

    n_layers = len(n_filters)
    filter_sizes = replicate_parameter_for_all_layers(filter_sizes, n_layers)
    strides = replicate_parameter_for_all_layers(strides, n_layers)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    layer = in_signal
    
    if (fully_connected_layers is not None) and (reshape_last_connected is not None):
        for i, layer_size in enumerate(fully_connected_layers):
            name = 'decoder_dense_layer_' + str(i)
            scope_i = expand_scope_by_name(scope, name)
            
            layer = fully_connected(layer, layer_size, activation='linear', weights_init='xavier', name=name, regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)
            
            
            if verbose:
                print name, 'FC params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

            if non_linearity is not None:
                layer = non_linearity(layer)
                
            if b_norm:
                name += '_bnorm'
                scope_i = expand_scope_by_name(scope, name)
                #layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
                layer = tf.keras.layers.BatchNormalization(axis=-1,renorm=True,name=name, momentum=0.9,epsilon=1e-5)(layer)
#                 if verbose:
#                     print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())


            if dropout_prob is not None and dropout_prob[i] > 0:
                layer = dropout(layer, 1.0 - dropout_prob[i])

            if verbose:
                print layer
                print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'
                
        layer = reshape(layer,[-1,reshape_last_connected[0],reshape_last_connected[1]])
        if verbose:
            print layer
            print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

    
    for i in xrange(n_layers):

        name = 'decoder_conv_layer_' + str(i)
        scope_i = expand_scope_by_name(scope, name)

        layer = conv_op(layer, nb_filter=n_filters[i], filter_size=filter_sizes[i],
                        strides=strides[i], padding=padding, regularizer=regularizer, weight_decay=weight_decay,
                        name=name, reuse=reuse, scope=scope_i, weights_init='xavier')

        if verbose:
            print name, 'conv params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

        if non_linearity is not None and i < n_layers - 1:  # Last layer doesn't have a non-linearity.
            layer = non_linearity(layer) 
           
        if (b_norm and i < n_layers - 1) or (i == n_layers - 1 and b_norm_finish):
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            #layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            layer = tf.keras.layers.BatchNormalization(axis=-1,renorm=True,name=name, momentum=0.9,epsilon=1e-5)(layer)
#             if verbose:
#                 print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())


        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        if upsample_sizes is not None and upsample_sizes[i] is not None:
            layer = tf.tile(layer, multiples=[1, upsample_sizes[i], 1])

        if verbose:
            print layer
            print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

    return layer