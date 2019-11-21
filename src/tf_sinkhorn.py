from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys
import ot as pot
import math


def sinkhorn_knopp_tf(a, b, M, reg, adaptive_min=None, numItermax=1000, stopThr=1e-9, verbose=False, **kwargs):
    M = tf.cast(M, tf.float64)
    a = tf.cast(a, tf.float64)
    b = tf.cast(b, tf.float64)
    if reg == 'adaptive':
        maxd = tf.reduce_max(M,axis=[-1,-2])
        if adaptive_min is None:
            reg = maxd/np.float64(708.)
        else:
            reg = tf.maximum(maxd/np.float64(708.), np.float64(adaptive_min))
        K = tf.exp(-M/tf.expand_dims(tf.expand_dims(reg,-1),-1))
    else:
        K = tf.exp(-M/reg)
    #u = tf.ones(a.shape)/tf.reduce_sum(a,axis=-1,keepdims=True)
    u = tf.fill(tf.shape(a),np.float64(1.))/tf.cast(tf.shape(a)[1],tf.float64)
    v = tf.fill(tf.shape(b),np.float64(1.))/tf.cast(tf.shape(b)[1],tf.float64)
    uprev = tf.fill(tf.shape(a),np.float64(1.))/tf.reduce_sum(a,axis=-1,keepdims=True)
    vprev = tf.fill(tf.shape(b),np.float64(1.))/tf.reduce_sum(b,axis=-1,keepdims=True)
    Kp = tf.expand_dims(1/a,-1)*K
    
    err = tf.Variable(1.,dtype=tf.float64)
    cpt = tf.Variable(0)
    
    flag = tf.Variable(1)
    
    mycond = lambda flag, err, cpt, Kp, u, v, uprev, vprev : tf.logical_and(tf.less(cpt, numItermax),tf.greater(err,stopThr))


    def loopfn(flag, err, cpt, Kp, u, v, uprev, vprev):
        uprev = u
        vprev = v
        
        KtransposeU = tf.squeeze(tf.matmul(K,tf.expand_dims(u,-1),transpose_a=True),axis=-1)
        v = b / KtransposeU
        u = 1/tf.squeeze(tf.matmul(Kp, tf.expand_dims(v,-1)),axis=-1)
        
#         error_cond = tf.reduce_any(tf.equal(KtransposeU,0))

        error_cond = tf.reduce_any(tf.is_nan(u))
        error_cond = tf.logical_or(error_cond, tf.reduce_any(tf.is_nan(v)))
        error_cond = tf.logical_or(error_cond, tf.reduce_any(tf.is_inf(u)))
        error_cond = tf.logical_or(error_cond, tf.reduce_any(tf.is_inf(v)))
        error_cond = tf.logical_or(error_cond, tf.reduce_any(tf.equal(KtransposeU,0)))
        
        def error_function_true():
            return tf.Variable(numItermax), uprev, vprev
        def error_function_false():
            return cpt+1, u, v
        cpt, u, v = tf.cond(error_cond,error_function_true,error_function_false)

        def cptmod10_true():
            
            tmp2 = tf.squeeze(tf.matmul(tf.expand_dims(u,-2),K),axis=1)*v
            #tmp2 = tf.einsum('ai,aij,aj->aj', u, K, v)
            newerr = tf.norm(tmp2-b,axis=-1)
            stopthr_cond = tf.reduce_all(tf.less(newerr,stopThr))
            
            def stopthr_false():
                return tf.reduce_max(newerr), flag + 1, cpt
            def stopthr_true():
                return tf.reduce_max(newerr), flag + 1, tf.Variable(numItermax)
            
            return tf.cond(stopthr_cond,stopthr_true,stopthr_false)
        
        def cptmod10_false():
            return err, flag, cpt
        
        cptmod10_cond = tf.equal(tf.floormod(cpt,10),0)
        err, flag, cpt = tf.cond(cptmod10_cond,cptmod10_true,cptmod10_false)
        



        return flag, err, cpt, Kp, u, v, uprev, vprev
    
    this = tf.while_loop(mycond, loopfn,[flag, err, cpt,  Kp, u, v, uprev, vprev])

    u = this[4]
    v = this[5]
    
    return tf.cast(tf.expand_dims(u,-1)*K*tf.expand_dims(v,-2), tf.float32)

# def ground_distance_tf(pointsa,pointsb,gradients=False):
    
#     a_dim = tf.shape(pointsa)[-2]
#     b_dim = tf.shape(pointsb)[-2]
    
#     amat = tf.tile(tf.expand_dims(pointsa,2),[1,1,b_dim,1])
#     bmat = tf.tile(tf.expand_dims(pointsb,1),[1,a_dim,1,1])
    
#     if gradients:
#         return (bmat - amat) / tf.norm(bmat - amat,axis=3)
#     else:
#         return tf.norm(bmat - amat,axis=3)


def ground_distance_tf_old(pointsa,pointsb,return_gradients=False,clip=True,epsilon=1e-2):
    
    a_dim = pointsa.shape[-2]
    b_dim = pointsb.shape[-2]
    
    amat = tf.tile(tf.expand_dims(pointsa,2),[1,1,b_dim,1])
    bmat = tf.tile(tf.expand_dims(pointsb,1),[1,a_dim,1,1])
    
    diffmat = bmat - amat
#     return tf.norm(diffmat,axis=3)

    zerogradients = tf.fill(amat.shape,0.)
    if clip:
        clipentries = tf.tile(tf.greater(epsilon, tf.expand_dims(dist,-1)),[1,1,1,2])
        diffmat = tf.where(clipentries,zerogradients,diffmat)
        gradients = tf.where(clipentries, zerogradients, diffmat/tf.expand_dims(dist,-1))
    elif return_gradients:
        gradients = diffmat / tf.expand_dims(dist,-1)

    if return_gradients:
        return tf.norm(diffmat,axis=3), gradients
    else:
        return tf.norm(bmat - amat,axis=3)

def ground_distance_tf(pointsa,pointsb,epsilon=1e-8, mod2pi=True):
    
    a_dim = pointsa.shape[-2]
    b_dim = pointsb.shape[-2]
    
    amat = tf.tile(tf.expand_dims(pointsa,2),[1,1,b_dim,1])
    bmat = tf.tile(tf.expand_dims(pointsb,1),[1,a_dim,1,1])
    
    diffmat = bmat - amat
    
    if mod2pi:
        dphi, deta = tf.unstack(diffmat,axis=-1)
        dphimod2pi = tf.floormod(dphi + math.pi,2*math.pi) - math.pi
        diffmat = tf.stack([dphimod2pi, deta],axis=-1)

    dist = tf.norm(diffmat,axis=3)
    
    
#     return tf.norm(diffmat,axis=3)

    epstensor = tf.constant(epsilon,dtype=tf.float32)

    zerogradients = tf.fill(tf.shape(amat),np.float32(0.))
    
    clipentries = tf.tile(tf.greater(epstensor, tf.expand_dims(dist,-1)),[1,1,1,2])
    diffmat = tf.where(clipentries,zerogradients,diffmat)
    gradients = tf.where(clipentries, zerogradients, diffmat/tf.expand_dims(dist,-1))



    return tf.norm(diffmat,axis=3), gradients

def ground_distance_tf_64(pointsa,pointsb,epsilon=1e-2):
    
    a_dim = pointsa.shape[-2]
    b_dim = pointsb.shape[-2]
    
    amat = tf.tile(tf.expand_dims(pointsa,2),[1,1,b_dim,1])
    bmat = tf.tile(tf.expand_dims(pointsb,1),[1,a_dim,1,1])
    
    diffmat = bmat - amat
    dist = tf.norm(diffmat,axis=3)
#     return tf.norm(diffmat,axis=3)

    epstensor = tf.constant(epsilon,dtype=tf.float64)

    zerogradients = tf.fill(amat.shape,np.float64(0.))
    
    clipentries = tf.tile(tf.greater(epstensor, tf.expand_dims(dist,-1)),[1,1,1,2])
    diffmat = tf.where(clipentries,zerogradients,diffmat)
    gradients = tf.where(clipentries, zerogradients, diffmat/tf.expand_dims(dist,-1))



    return tf.norm(diffmat,axis=3), gradients
    

def sinkhorn_loss_tf(in_locations, out_locations, c, out_weights = None, in_weights = None):     
    ground_distance, ground_dist_gradient = ground_distance_tf(in_locations,out_locations)
#self.out_weights = tf.placeholder(tf.int32,shape=([None] + self.n_output)[:-1])
    if out_weights is None:
        out_weights = tf.constant(1./500.,shape=out_locations.shape[:-1])
       
    if c.exists_and_is_not_none('adaptive_min'):
        adaptive_min = c.adaptive_min
    else:
        adaptive_min = None
    match = sinkhorn_knopp_tf(in_weights, out_weights, ground_distance, c.sinkhorn_reg, numItermax=c.numItermax, stopThr=c.stopThr, adaptive_min =adaptive_min)


#     def grad(dL):
#         ground_dist_gradient_perm = tf.transpose(ground_dist_gradient,[0,3,1,2])
#         loss_grad_temp = tf.matrix_diag_part(tf.matmul(tf.tile(tf.expand_dims(match,1),[1,2,1,1]),ground_dist_gradient_perm,transpose_a = True))
#         return tf.transpose(loss_grad_temp,[0,2,1])  

    return tf.trace(tf.matmul(match,ground_distance,transpose_b=True))