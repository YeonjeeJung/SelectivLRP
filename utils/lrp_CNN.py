import tensorflow as tf
import tensorflow.keras.layers as tkl
from tensorflow.python.ops import nn_ops, gen_nn_ops
import matplotlib.pyplot as plt

import numpy as np

import copy

class LRP:
    def __init__(self, inputmodel, name):
        self.model = inputmodel
        self.name = name
        self.layers = self.model.layers

    def __call__(self, X, label):
        self.activations, _ = self.model(X)
        
        Rs = {}
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            layertype = type(layer).__name__
            #print(layer.name, layertype)
            
            if 'predictions' in layer.name:
                weight, bias = layer.variables
                rs = self.cal_last_relevance(self.activations[layer.output.name], label, self.activations[layer.input.name], weight, bias)
                
                Rs[layer.input.name] = rs
            
            else:
                if 'Conv' in layertype:
                    weight, bias = layer.variables
                    strides = layer.strides
                    padding = layer.padding.upper()
                    
                    rs = self.backprop_conv(self.activations[layer.input.name], weight, bias, strides, padding, Rs[layer.output.name])
                    
                    if layer.input.name in Rs.keys():
                        rs = self.backprop_clone(self.activations[layer.input.name], [Rs[layer.input.name], rs])
                        
                    Rs[layer.input.name] = rs
                    
                elif 'Dense' in layertype:
                    weight, bias = layer.variables
                    rs = self.backprop_dense(self.activations[layer.input.name], weight, bias, Rs[layer.output.name])
                    Rs[layer.input.name] = rs
                
                elif 'MaxPooling' in layertype:
                    rs = self.backprop_grad(self.activations[layer.input.name], layer, Rs[layer.output.name])
                    Rs[layer.input.name] = rs
                    
                elif 'Flatten' in layertype:
                    shape = self.activations[layer.input.name].shape.as_list()
                    Rs[layer.input.name] = Rs[layer.output.name].numpy().reshape(shape)
                    
                elif 'Add' in layertype:
                    rs = self.backprop_add([self.activations[layer.input[0].name], self.activations[layer.input[1].name]], Rs[layer.output.name])
                    Rs[layer.input[0].name] = rs[0]
                    Rs[layer.input[1].name] = rs[1]
                    
                elif 'InputLayer' in layertype:
                    continue
                    
                elif 'GlobalAveragePooling' in layertype:
                    rs = self.backprop_grad(self.activations[layer.input.name], layer, Rs[layer.output.name])
                    Rs[layer.input.name] = rs
                    
                elif 'BatchNormalization' in layertype:
                    Rs[layer.input.name] = Rs[layer.output.name]
                    
                elif 'Activation' in layertype:
                    rs = self.backprop_grad(self.activations[layer.input.name], layer, Rs[layer.output.name])
                    Rs[layer.input.name] = rs
                    #Rs[layer.input.name] = Rs[layer.output.name]
                    
                elif 'ZeroPadding' in layertype:
                    rs = self.backprop_grad(self.activations[layer.input.name], layer, Rs[layer.output.name])
                    Rs[layer.input.name] = rs
                
                else:
                    raise Error('Unknown operation')
                    
        return Rs[self.layers[0].input.name]
    
    def backprop_conv(self, activation, weight, bias, strides, padding, relevance):
        in_sum = tf.reduce_sum(relevance)
        
        stride = [1, *strides, 1]
        
        a_p = tf.maximum(0., activation)
        a_n = tf.minimum(0., activation)
        
        w_p = tf.maximum(0., weight)
        b_p = tf.maximum(0., bias)
        
        z_p = nn_ops.conv2d(a_p, w_p, stride, padding) + b_p
        
        w_n = tf.minimum(0., weight)
        b_n = tf.minimum(0., bias)
        
        z_n = nn_ops.conv2d(a_n, w_n, stride, padding) + b_n
        
        z = z_p + z_n
        
        relevance = tf.cast(relevance, z.dtype)
        s = tf.math.divide_no_nan(relevance, z)
        
        tmp_p = nn_ops.conv2d_backprop_input(tf.shape(a_p), w_p, s, stride, padding)
        tmp_n = nn_ops.conv2d_backprop_input(tf.shape(a_n), w_n, s, stride, padding)
        
        tmp_p = tf.multiply(a_p, tmp_p)
        tmp_n = tf.multiply(a_n, tmp_n)
        
        out_sum = tf.reduce_sum(tmp_p + tmp_n)
        
        return (tmp_p + tmp_n) * in_sum / out_sum
    
    def backprop_dense(self, activation, weight, bias, relevance):
        in_sum = tf.reduce_sum(relevance)
        
        a_p = tf.maximum(0., activation)
        a_n = tf.minimum(0., activation)
        
        w_p = tf.maximum(0., weight)
        b_p = tf.maximum(0., bias)
        
        z_p = tf.matmul(a_p, w_p) + b_p
        
        w_n = tf.minimum(0., weight)
        b_n = tf.minimum(0., bias)
        
        z_n = tf.matmul(a_n, w_n) + b_n
        
        z = z_p + z_n
        
        relevance = tf.cast(relevance, z.dtype)
        s = tf.math.divide_no_nan(relevance, z)
        
        tmp_p = tf.matmul(s, tf.transpose(w_p))
        tmp_n = tf.matmul(s, tf.transpose(w_n))
        
        tmp_p = tf.multiply(a_p, tmp_p)
        tmp_n = tf.multiply(a_n, tmp_n)
        
        out_sum = tf.reduce_sum(tmp_p + tmp_n)
        
        return (tmp_p + tmp_n) * in_sum / out_sum
    
    def backprop_grad(self, activation, layer, relevance):
        with tf.GradientTape() as tape:
            tape.watch(activation)
            z = layer(activation)
            
            s = tf.math.divide_no_nan(relevance, z)
        
        c = tape.gradient(z, activation, output_gradients=s)
        tmp = activation * c
    
        return tmp
    
    def backprop_add(self, activations, relevance):
        z = activations[0] + activations[1]
        s = tf.math.divide_no_nan(relevance, z)
        
        return [activations[0] * s, activations[1] * s]
    
    def backprop_clone(self, activation, relevance):
        with tf.GradientTape() as tape:
            tape.watch(activation)
            
            z = [activation, activation]
            s = [tf.math.divide_no_nan(r, zval) for r, zval in zip(relevance, z)]
            
        c = tape.gradient(z, activation, output_gradients=s)
        tmp = activation * c
        return tmp
    
    def cal_last_relevance(self, last_activation, label, activation, weight, bias):
        mask = np.zeros(last_activation.shape, dtype=np.float32)
        mask[0, label] = 1
        
        rs = tf.multiply(last_activation, mask)
        return self.backprop_dense(activation, weight, bias, rs)
    
class SelectiveLRP(LRP):
    def __init__(self, inputmodel, name):
        self.model = inputmodel
        self.name = name
        self.layers = self.model.layers
        self.original_model = tf.keras.applications.ResNet50()

    def __call__(self, X, label):
        self.gradients = {}
        with tf.GradientTape() as tape:
            activation, logit = self.model(X)
            
            logit = tf.nn.softmax(logit)
            
            act_arr = []
            for layer in self.layers:
                if 'Conv' in type(layer).__name__:
                    act_arr.append(activation[layer.input.name])
                    #self.gradients[layer.input.name] = tape.gradient(logit[0,int(label)], activation[layer.input.name])
            grad_arr = tape.gradient(logit[0, int(label)], act_arr)
            
        i = 0
        for layer in self.layers:
            if 'Conv' in type(layer).__name__:
                self.gradients[layer.input.name] = grad_arr[i]
                i += 1
            
        self.activations = activation
        self.layers = self.model.layers

        Rs = {}
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            layertype = type(layer).__name__
            #print(layer.name, layertype)

            if 'predictions' in layer.name:
                weight, bias = layer.variables
                rs = self.cal_last_relevance(self.activations[layer.output.name], label, self.activations[layer.input.name], weight, bias)

                Rs[layer.input.name] = rs

            else:
                if 'Conv' in layertype:
                    weight, bias = layer.variables
                    strides = layer.strides
                    padding = layer.padding.upper()
                    gradient = self.gradients[layer.input.name]

                    rs = self.backprop_conv(self.activations[layer.input.name], weight, bias, strides, padding, gradient, Rs[layer.output.name])

                    if layer.input.name in Rs.keys():
                        rs = self.backprop_clone(self.activations[layer.input.name], [Rs[layer.input.name], rs])

                    Rs[layer.input.name] = rs

                elif 'Dense' in layertype:
                    weight, bias = layer.variables
                    rs = self.backprop_dense(self.activations[layer.input.name], weight, bias, Rs[layer.output.name])
                    Rs[layer.input.name] = rs

                elif 'MaxPooling' in layertype:
                    rs = self.backprop_grad(self.activations[layer.input.name], layer, Rs[layer.output.name])
                    Rs[layer.input.name] = rs

                elif 'Flatten' in layertype:
                    shape = self.activations[layer.input.name].shape.as_list()
                    Rs[layer.input.name] = Rs[layer.output.name].numpy().reshape(shape)

                elif 'Add' in layertype:
                    rs = self.backprop_add([self.activations[layer.input[0].name], self.activations[layer.input[1].name]], Rs[layer.output.name])
                    Rs[layer.input[0].name] = rs[0]
                    Rs[layer.input[1].name] = rs[1]

                elif 'InputLayer' in layertype:
                    continue

                elif 'GlobalAveragePooling' in layertype:
                    rs = self.backprop_grad(self.activations[layer.input.name], layer, Rs[layer.output.name])
                    Rs[layer.input.name] = rs

                elif 'BatchNormalization' in layertype:
                    Rs[layer.input.name] = Rs[layer.output.name]

                elif 'Activation' in layertype:
                    rs = self.backprop_grad(self.activations[layer.input.name], layer, Rs[layer.output.name])
                    Rs[layer.input.name] = rs
                    #Rs[layer.input.name] = Rs[layer.output.name]

                elif 'ZeroPadding' in layertype:
                    rs = self.backprop_grad(self.activations[layer.input.name], layer, Rs[layer.output.name])
                    Rs[layer.input.name] = rs

                else:
                    raise Error('Unknown operation')
                    
        return Rs[self.layers[0].input.name]
    
    def backprop_conv(self, activation, weight, bias, strides, padding, gradient, relevance):
        in_sum = tf.reduce_sum(relevance)
        
        stride = [1, *strides, 1]
        
        a_p = tf.maximum(0., activation)
        a_n = tf.minimum(0., activation)
        
        w_p = tf.maximum(0., weight)
        b_p = tf.maximum(0., bias)
        
        z_p = nn_ops.conv2d(a_p, w_p, stride, padding) + b_p
        
        w_n = tf.minimum(0., weight)
        b_n = tf.minimum(0., bias)
        
        z_n = nn_ops.conv2d(a_n, w_n, stride, padding) + b_n
        
        z = z_p + z_n
        
        relevance = tf.cast(relevance, z.dtype)
        s = tf.math.divide_no_nan(relevance, z)
        
        tmp_p = nn_ops.conv2d_backprop_input(tf.shape(a_p), w_p, s, stride, padding)
        tmp_n = nn_ops.conv2d_backprop_input(tf.shape(a_n), w_n, s, stride, padding)
        
        if gradient is not None:
            if activation.shape != gradient.shape:
                raise Error("shape not match")
                
            partial_lin = tf.nn.avg_pool(gradient, [1,gradient.shape[1],gradient.shape[2],1], [1,1,1,1], 'VALID')
            partial_lin = tf.where(partial_lin > 0, partial_lin, 0)
            denominator = tf.reduce_sum(partial_lin)
            a_p = tf.multiply(a_p, partial_lin / denominator)
            a_n = tf.multiply(a_n, partial_lin / denominator)
        
        tmp_p = tf.multiply(a_p, tmp_p)
        tmp_n = tf.multiply(a_n, tmp_n)
        
        out_sum = tf.reduce_sum(tmp_p + tmp_n)
        
        return (tmp_p + tmp_n) * in_sum / out_sum
    
class _SGLRPTarget(LRP):
    def cal_last_relevance(self, last_activation, label, activation, weight, bias):
        logit = tf.nn.softmax(last_activation)
        mask = np.zeros(logit.shape, dtype=np.float32)
        
        logit = logit * (1 - logit[0, label])
        
        # Target is 1, else 0
        mask[0, label] = 1
        rs = logit * mask
    
        return self.backprop_dense(activation, weight, bias, rs)
    
class _SGLRPDual(LRP):
    def cal_last_relevance(self, last_activation, label, activation, weight, bias):
        logit = tf.nn.softmax(last_activation)
        mask = np.ones(logit.shape, dtype=np.float32)
        
        logit = logit * logit[0, label]
        
        # Target is 0, else 1
        mask[0, label] = 0
        rs = logit * mask
    
        return self.backprop_dense(activation, weight, bias, rs)
    
class SGLRP():
    def __init__(self, inputmodel, name):
        self.sglrpT = _SGLRPTarget(inputmodel, name)
        self.sglrpD = _SGLRPDual(inputmodel, name)
        
    def __call__(self, X, label):
        return self.sglrpT(X, label) - self.sglrpD(X, label)

class _CLRPTarget(LRP):
    def cal_last_relevance(self, last_activation, label, activation, weight, bias):
        
        # Target is 1, else 0
        mask = np.zeros(last_activation.shape)
        mask[0, label] = 1
        rs = tf.multiply(last_activation, mask)
    
        return self.backprop_dense(activation, weight, bias, rs)
    
class _CLRPDual(LRP):
    def cal_last_relevance(self, last_activation, label, activation, weight, bias):
        
        targetz = last_activation[0, label]
        clrplogit = np.ones(last_activation.shape) * targetz 
        clrplogit /= (last_activation.shape[-1] - 1)
        
        # Target is 0, else 1
        mask = np.ones(last_activation.shape)
        mask[0, label] = 0
        rs = tf.multiply(clrplogit, mask)
    
        return self.backprop_dense(activation, weight, bias, rs)
    
class CLRP():
    def __init__(self, inputmodel, name):
        self.clrpT = _CLRPTarget(inputmodel, name)
        self.clrpD = _CLRPDual(inputmodel, name)
        
    def __call__(self, X, label):
        return self.clrpT(X, label) - self.clrpD(X, label)