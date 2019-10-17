'''
Created on September 2, 2017

@author: optas
'''
import numpy as np
import tensorflow as tf

from . encoders_decoders_vae import encoder_with_convs_and_symmetry, decoder_with_fc_only, decoder_with_convs_only, decoder_with_convs_and_fc
from . encoders_decoders_vae import variational_encoder_with_convs_and_symmetry, variational_encoder_with_convs_and_symmetry_bnormafter

def jack_vae_template_3(n_pc_points, bneck_size, bneck_post_mlp=False):
    ''' Single class experiments.
    '''
#     if n_pc_points != 2048:
#         raise ValueError()

    encoder = variational_encoder_with_convs_and_symmetry_bnormafter
    decoder = decoder_with_convs_and_fc

    n_input = [n_pc_points, 2]

    encoder_args = {'n_filters': [128, 128],
                    'filter_sizes': [1],
                    'strides': [1],
                    'b_norm': True,
                    'verbose': True,
                    'symmetry': tf.reduce_mean,
                    'fully_connected_layers': [128,128],
                    'latent_size':bneck_size
                    }

    decoder_args = {'n_filters': [128,128,128,2],
                    'filter_sizes': [4,4,4,1],
                    'strides': [1],
                    'upsample_sizes': [2,2,2,None],
                    'b_norm': True,
                    'b_norm_finish': True,
                    'verbose': True,
                    'fully_connected_layers': [128,125*3],
                    'reshape_last_connected': [125,3]
                    }

    if bneck_post_mlp:
        encoder_args['n_filters'].pop()
        decoder_args['layer_sizes'][0] = bneck_size

    return encoder, decoder, encoder_args, decoder_args


def jack_vae_template_1_bnormtest(n_pc_points, bneck_size, bneck_post_mlp=False):
    ''' Single class experiments.
    '''
#     if n_pc_points != 2048:
#         raise ValueError()

    encoder = variational_encoder_with_convs_and_symmetry_bnormafter
    decoder = decoder_with_convs_and_fc

    n_input = [n_pc_points, 2]

    encoder_args = {'n_filters': [128, 128],
                    'filter_sizes': [1],
                    'strides': [1],
                    'b_norm': True,
                    'verbose': True,
                    'symmetry': tf.reduce_mean,
                    'fully_connected_layers': [128,128],
                    'latent_size':bneck_size
                    }

    decoder_args = {'n_filters': [128,128,2],
                    'filter_sizes': [1],
                    'strides': [1],
                    'upsample_sizes': [4,2,None],
                    'b_norm': True,
                    'b_norm_finish': True,
                    'verbose': True,
                    'fully_connected_layers': [128,125*3],
                    'reshape_last_connected': [125,3]
                    }

    if bneck_post_mlp:
        encoder_args['n_filters'].pop()
        decoder_args['layer_sizes'][0] = bneck_size
        
    return encoder, decoder, encoder_args, decoder_args


def jack_vae_template_1(n_pc_points, bneck_size, bneck_post_mlp=False):
    ''' Single class experiments.
    '''
#     if n_pc_points != 2048:
#         raise ValueError()

    encoder = variational_encoder_with_convs_and_symmetry
    decoder = decoder_with_convs_only

    n_input = [n_pc_points, 2]

    encoder_args = {'n_filters': [128, 128],
                    'filter_sizes': [1],
                    'strides': [1],
                    'b_norm': True,
                    'verbose': True,
                    'symmetry': tf.reduce_mean,
                    'fully_connected_layers': [128,128],
                    'latent_size':bneck_size
                    }

    decoder_args = {'n_filters': [128,128,2],
                    'filter_sizes': [1],
                    'strides': [1],
                    'upsample_sizes': [4,2,None],
                    'b_norm': True,
                    'b_norm_finish': True,
                    'verbose': True,
                    'fully_connected_layers': [128,125*3],
                    'reshape_last_connected': [125,3]
                    }

    if bneck_post_mlp:
        encoder_args['n_filters'].pop()
        decoder_args['layer_sizes'][0] = bneck_size

    return encoder, decoder, encoder_args, decoder_args

def jack_vae_template_2(n_pc_points, bneck_size, bneck_post_mlp=False):
    ''' Single class experiments.
    '''
#     if n_pc_points != 2048:
#         raise ValueError()

    encoder = variational_encoder_with_convs_and_symmetry
    decoder = decoder_with_convs_only

    n_input = [n_pc_points, 2]

    encoder_args = {'n_filters': [256, 256,256],
                    'filter_sizes': [1],
                    'strides': [1],
                    'b_norm': True,
                    'verbose': True,
                    'symmetry': tf.reduce_mean,
                    'fully_connected_layers': [512,512,512],
                    'latent_size':bneck_size
                    }

    decoder_args = {'fully_connected_layers': [512,512,250*3],
                    'reshape_last_connected': [250,3],
                    'n_filters': [256,256,256,256,2],
                    'filter_sizes': [1],
                    'strides': [1],
                    'upsample_sizes': [None,2,None,2,None],
                    'b_norm': True,
                    'b_norm_finish': True,
                    'verbose': True
                    }

    if bneck_post_mlp:
        encoder_args['n_filters'].pop()
        decoder_args['layer_sizes'][0] = bneck_size

    return encoder, decoder, encoder_args, decoder_args

def jack_2_template(n_pc_points, bneck_size, bneck_post_mlp=False):
    ''' Single class experiments.
    '''
#     if n_pc_points != 2048:
#         raise ValueError()

    encoder = encoder_with_convs_and_symmetry
    decoder = decoder_with_convs_only

    n_input = [n_pc_points, 2]

    encoder_args = {'n_filters': [128, 128],
                    'filter_sizes': [1],
                    'strides': [1],
                    'b_norm': True,
                    'verbose': True,
                    'symmetry': tf.reduce_sum,
                    'fully_connected_layers': [128,128,bneck_size]
                    }

    decoder_args = {'n_filters': [128, 128,2],
                    'filter_sizes': [1],
                    'strides': [1],
                    'upsample_sizes': [4,2,None],
                    'b_norm': False,
                    'b_norm_finish': False,
                    'verbose': True,
                    'fully_connected_layers': [128,125*3],
                    'reshape_last_connected': [125,3]
                    }

    if bneck_post_mlp:
        encoder_args['n_filters'].pop()
        decoder_args['layer_sizes'][0] = bneck_size

    return encoder, decoder, encoder_args, decoder_args

def jack_1_template(n_pc_points, bneck_size, bneck_post_mlp=False):
    ''' Single class experiments.
    '''
#     if n_pc_points != 2048:
#         raise ValueError()

    encoder = encoder_with_convs_and_symmetry
    decoder = decoder_with_fc_only

    n_input = [n_pc_points, 2]

    encoder_args = {'n_filters': [128, 128],
                    'filter_sizes': [1],
                    'strides': [1],
                    'b_norm': True,
                    'verbose': True,
                    'symmetry': tf.reduce_sum,
                    'fully_connected_layers': [128,128,bneck_size]
                    }

    decoder_args = {'layer_sizes': [256, 256, np.prod(n_input)],
                    'b_norm': False,
                    'b_norm_finish': False,
                    'verbose': True
                    }

    if bneck_post_mlp:
        encoder_args['n_filters'].pop()
        decoder_args['layer_sizes'][0] = bneck_size

    return encoder, decoder, encoder_args, decoder_args

def mlp_architecture_ala_iclr_18(n_pc_points, bneck_size, bneck_post_mlp=False):
    ''' Single class experiments.
    '''
#     if n_pc_points != 2048:
#         raise ValueError()

    encoder = encoder_with_convs_and_symmetry
    decoder = decoder_with_fc_only

    n_input = [n_pc_points, 2]

    encoder_args = {'n_filters': [64, 128, 128, 256, bneck_size],
                    'filter_sizes': [1],
                    'strides': [1],
                    'b_norm': True,
                    'verbose': True
                    }

    decoder_args = {'layer_sizes': [256, 256, np.prod(n_input)],
                    'b_norm': False,
                    'b_norm_finish': False,
                    'verbose': True
                    }

    if bneck_post_mlp:
        encoder_args['n_filters'].pop()
        decoder_args['layer_sizes'][0] = bneck_size

    return encoder, decoder, encoder_args, decoder_args


def default_train_params(single_class=True):
    params = {'batch_size': 50,
              'training_epochs': 500,
              'denoising': False,
              'learning_rate': 0.0005,
              'z_rotate': False,
              'saver_step': 10,
              'loss_display_step': 1
              }

    if not single_class:
        params['z_rotate'] = True
        params['training_epochs'] = 1000

    return params
