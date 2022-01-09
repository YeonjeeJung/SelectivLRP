import tensorflow as tf
import tensorflow.keras.layers as tkl

class VGG16(tf.keras.Model):
    """
    input size must me 224x224
    """
    def __init__(self):
        super(VGG16, self).__init__()
        original_model = tf.keras.applications.VGG16()
        layers = []
        for layer in original_model.layers:
            #print(layer.name)
            if layer.name == 'predictions':
                layer.activation = None
            layers.append(layer)
        
        self.layer_arr = layers
        
    def call(self, inputs):
        activations = {}
        
        for layer in self.layers:
            # for first layer
            if not activations:
                activations[layer.input.name] = inputs
                
            activations[layer.output.name] = layer(activations[layer.input.name])
            
        outputs = activations[self.layers[-1].output.name]
        return activations, outputs
    
class VGG19(tf.keras.Model):
    """
    input size must me 224x224
    """
    def __init__(self):
        super(VGG19, self).__init__()
        original_model = tf.keras.applications.VGG19()
        layers = []
        for layer in original_model.layers:
            #print(layer.name)
            if layer.name == 'predictions':
                layer.activation = None
            layers.append(layer)
        
        self.layer_arr = layers
        
    def call(self, inputs):
        activations = [inputs]
        outputs = inputs
        
        for layer in self.layers:
            outputs = layer(outputs)
            activations.append(outputs)
        
        return activations, outputs
    
class ResNet50(tf.keras.Model):
    """
    input size must me 224x224
    """
    def __init__(self):
        super(ResNet50, self).__init__()
        layers = []
        original_model = tf.keras.applications.ResNet50()
        for layer in original_model.layers:
            #print(layer.name)
            if layer.name == 'predictions':
                layer.activation = None
            layers.append(layer)
        
        self.layer_arr = layers
        
    def call(self, inputs):
        activations = {}
        
        for layer in self.layers:
            # for first layer
            if not activations:
                activations[layer.input.name] = inputs
                
            # for add layer
            if type(layer.input) is list:
                activations[layer.output.name] = layer([activations[layer.input[0].name], activations[layer.input[1].name]])
            else:
                activations[layer.output.name] = layer(activations[layer.input.name])
            
        outputs = activations[self.layers[-1].output.name]
        return activations, outputs