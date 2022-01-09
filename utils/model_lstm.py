import pickle
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tkl

class bidirLSTM(tf.keras.Model):
    def __init__(self):
        super(bidirLSTM, self).__init__()
        
        self.voc, embedding, weights = self.get_data()
        
        self.embedding = tkl.Embedding(embedding.shape[0], embedding.shape[1])
        self.lstm_left = tkl.LSTM(60, return_sequences=True, return_state=True)
        self.lstm_right = tkl.LSTM(60, return_sequences=True, return_state=True)
        self.dense_left = tkl.Dense(5, use_bias=False)
        self.dense_right = tkl.Dense(5, use_bias=False)
        
        initialinput = ['.']
        initialinputindex = tf.convert_to_tensor([self.voc.index(w) for w in initialinput])
        self.embedding(initialinputindex)
        self.set_weights(embedding, weights)
    
    def call(self, inputs):
        input_index = tf.convert_to_tensor([self.voc.index(w) for w in inputs])
        
        x = self.embedding(input_index)
        outputs_lstmleft, acts_lstmleft = self.forward_lstm(x, self.Wxh_Left, self.Whh_Left, self.bxh_Left, self.bhh_Left)
        outputs_lstmright, acts_lstmright = self.forward_lstm(x[::-1,:], self.Wxh_Right, self.Whh_Right, self.bxh_Right, self.bhh_Right)
        outputs_denseleft = self.forward_dense(outputs_lstmleft, self.Why_Left)
        outputs_denseright = self.forward_dense(outputs_lstmright, self.Why_Right)
        outputs = outputs_denseleft + outputs_denseright
        
        return outputs
    
    def predict_with_cover(self, inputs, cover_idx):
        input_index = np.array([self.voc.index(w) for w in inputs])
        
        x = self.embedding(input_index)
        x = x.numpy()
        
        for i in cover_idx:
            x[i] = np.zeros((60))
            
        outputs_lstmleft, acts_lstmleft = self.forward_lstm(x, self.Wxh_Left, self.Whh_Left, self.bxh_Left, self.bhh_Left)
        outputs_lstmright, acts_lstmright = self.forward_lstm(x[::-1,:], self.Wxh_Right, self.Whh_Right, self.bxh_Right, self.bhh_Right)
        outputs_denseleft = self.forward_dense(outputs_lstmleft, self.Why_Left)
        outputs_denseright = self.forward_dense(outputs_lstmright, self.Why_Right)
        outputs = outputs_denseleft + outputs_denseright
        
        return outputs
    
    def call_and_get_activations(self, inputs):
        input_index = tf.convert_to_tensor([self.voc.index(w) for w in inputs])
        
        x = self.embedding(input_index)
        outputs_lstmleft, acts_lstmleft = self.forward_lstm(x, self.Wxh_Left, self.Whh_Left, self.bxh_Left, self.bhh_Left)
        x_rev = x[::-1,:]
        outputs_lstmright, acts_lstmright = self.forward_lstm(x_rev, self.Wxh_Right, self.Whh_Right, self.bxh_Right, self.bhh_Right)
        outputs_denseleft = self.forward_dense(outputs_lstmleft, self.Why_Left)
        outputs_denseright = self.forward_dense(outputs_lstmright, self.Why_Right)
        outputs = outputs_denseleft + outputs_denseright
        
        return outputs, x, x_rev, acts_lstmleft, acts_lstmright, outputs_denseleft, outputs_denseright
        
    def forward_lstm(self, x, Wxh, Whh, bxh, bhh):
        T = x.shape[0]
        d = int(self.Wxh_Left.shape[0]/4)
        
        h = [tf.zeros((60,))]
        c = [tf.zeros((60,))]
        gates_xh = []
        gates_hh = []
        gates_pre = []
        gates = []
        
        for t in range(T):
            gates_xh.append(tf.tensordot(Wxh, x[t], axes=1))
            gates_hh.append(tf.tensordot(Whh, h[-1], axes=1))
            gates_pre.append(gates_xh[-1] + gates_hh[-1] + bxh + bhh)
            gates_pre_i, gates_pre_f, gates_pre_c, gates_pre_o = tf.split(gates_pre[-1], 4)
            gates_i, gates_f, gates_o = tf.math.sigmoid([gates_pre_i, gates_pre_f, gates_pre_o])
            gates_c = tf.math.tanh(gates_pre_c)
            gates.append(tf.concat([gates_i, gates_f, gates_c, gates_o], axis=0))
            c.append(gates_f * c[-1] + gates_i * gates_c)
            h.append(gates_o * tf.math.tanh(c[-1]))
        
        acts = {}
        acts['h'] = h[1:]
        acts['h'].append(h[0])
        acts['c'] = c[1:]
        acts['c'].append(c[0])
        acts['gates_xh'] = gates_xh
        acts['gates_hh'] = gates_hh
        acts['gates_pre'] = gates_pre
        acts['gates'] = gates
        
        return h[-1], acts
    
    def forward_dense(self, x, Why):
        return tf.tensordot(Why, x, axes=1)
        
    def get_data(self, model_path='./data/'):
    
        f_voc = open(model_path + 'vocab', 'rb')
        voc = pickle.load(f_voc)
        f_voc.close()

        embedding = np.load(model_path + 'embeddings.npy', mmap_mode='r')

        f_weights = open(model_path + 'model', 'rb')
        weights = pickle.load(f_weights)
        f_weights.close()

        return voc, embedding, weights
    
    def set_weights(self, embedding, weights):
        self.emb_weights = embedding.copy()
        self.embedding.set_weights([self.emb_weights])
        
        Wxh_Left_i, Wxh_Left_c, Wxh_Left_f, Wxh_Left_o = np.split(weights['Wxh_Left'].copy(), 4)
        Whh_Left_i, Whh_Left_c, Whh_Left_f, Whh_Left_o = np.split(weights['Whh_Left'].copy(), 4)
        bxh_Left_i, bxh_Left_c, bxh_Left_f, bxh_Left_o = np.split(weights['bxh_Left'].copy(), 4)
        bhh_Left_i, bhh_Left_c, bhh_Left_f, bhh_Left_o = np.split(weights['bhh_Left'].copy(), 4)
        
        self.Wxh_Left = np.concatenate([Wxh_Left_i, Wxh_Left_f, Wxh_Left_c, Wxh_Left_o]).astype(np.float32)
        self.Whh_Left = np.concatenate([Whh_Left_i, Whh_Left_f, Whh_Left_c, Whh_Left_o]).astype(np.float32)
        self.bxh_Left = np.concatenate([bxh_Left_i, bxh_Left_f, bxh_Left_c, bxh_Left_o]).astype(np.float32)
        self.bhh_Left = np.concatenate([bhh_Left_i, bhh_Left_f, bhh_Left_c, bhh_Left_o]).astype(np.float32)
        lstm_left_weight = [self.Wxh_Left.T, self.Whh_Left.T, self.bxh_Left+self.bhh_Left]
        
        Wxh_Right_i, Wxh_Right_c, Wxh_Right_f, Wxh_Right_o = np.split(weights['Wxh_Right'].copy(), 4)
        Whh_Right_i, Whh_Right_c, Whh_Right_f, Whh_Right_o = np.split(weights['Whh_Right'].copy(), 4)
        bxh_Right_i, bxh_Right_c, bxh_Right_f, bxh_Right_o = np.split(weights['bxh_Right'].copy(), 4)
        bhh_Right_i, bhh_Right_c, bhh_Right_f, bhh_Right_o = np.split(weights['bhh_Right'].copy(), 4)
        
        self.Wxh_Right = np.concatenate([Wxh_Right_i, Wxh_Right_f, Wxh_Right_c, Wxh_Right_o]).astype(np.float32)
        self.Whh_Right = np.concatenate([Whh_Right_i, Whh_Right_f, Whh_Right_c, Whh_Right_o]).astype(np.float32)
        self.bxh_Right = np.concatenate([bxh_Right_i, bxh_Right_f, bxh_Right_c, bxh_Right_o]).astype(np.float32)
        self.bhh_Right = np.concatenate([bhh_Right_i, bhh_Right_f, bhh_Right_c, bhh_Right_o]).astype(np.float32)
        lstm_right_weight = [self.Wxh_Right.T, self.Whh_Right.T, self.bxh_Right+self.bhh_Right]
        
        self.Why_Left = weights['Why_Left'].copy().astype(np.float32)
        self.Why_Right = weights['Why_Right'].copy().astype(np.float32)
        
    def sensitivity_analysis(self, inputs, target_class):
        with tf.GradientTape() as tape:
            outputs, x, x_rev, acts_lstmleft, acts_lstmright, outputs_denseleft, outputs_denseright = self.call_and_get_activations(inputs)
            gradient_x, gradient_x_rev = tape.gradient(outputs[target_class], [x, x_rev])
            
        return gradient_x, gradient_x_rev

    
    def lrp(self, inputs, target_class):
        outputs, x, x_rev, acts_lstmleft, acts_lstmright, outputs_denseleft, outputs_denseright = self.call_and_get_activations(inputs)
        outputs = tf.nn.softmax(outputs)
        outputs = outputs.numpy()
        
        C = self.Why_Left.shape[0]
        
        r_mask = np.zeros((C))
        r_mask[target_class] = 1.0
        
        R_Left, R_Right = self.lrp_add(outputs_denseleft, outputs_denseright, outputs*r_mask)
        Rx_Left, Rh_Left, Rc_Left = self.lrp_lstm(x, acts_lstmleft, self.Wxh_Left, self.bxh_Left, self.Whh_Left, self.bhh_Left, self.Why_Left, R_Left)
        Rx_Right, Rh_Right, Rc_Right = self.lrp_lstm(x_rev, acts_lstmright, self.Wxh_Right, self.bxh_Left, self.Whh_Right, self.bhh_Right, self.Why_Right, R_Right)
        
        return Rx_Left, Rx_Right[::-1,:], tf.reduce_sum(Rh_Left[-1]+Rc_Left[-1]+Rh_Right[-1]+Rc_Right[-1])
        
    def lrp_lstm(self, x, acts_lstm, Wxh, Bxh, Whh, Bhh, Why, relevance):
        T = len(x)
        d = int(Wxh.shape[0]/4)
        e = self.emb_weights.shape[1]
        C = Why.shape[0]
        
        Rx = np.zeros(x.shape)
        Rh = np.zeros((T+1, d))
        Rc = np.zeros((T+1, d))
        Rg = np.zeros((T, d))
        
        Rh[T-1] = self.lrp_dense(acts_lstm['h'][T-1], Why.T, np.zeros(C), relevance)
        
        for t in reversed(range(T)):
            Rc[t] += Rh[t]
            gates_i, gates_f, gates_c, gates_o = np.split(acts_lstm['gates'][t], 4)
            
            Rc[t-1], Rg[t] = self.lrp_add(gates_f*acts_lstm['c'][t-1], gates_i*gates_c, Rc[t])

            a_x = tf.tensordot(Wxh[2*d:3*d], x[t], axes=1) + Bxh[2*d:3*d]
            a_h = tf.tensordot(Whh[2*d:3*d], acts_lstm['h'][t-1], axes=1) + Bhh[2*d:3*d]
        
            R_x, R_h = self.lrp_add(a_x, a_h, Rg[t])
            Rx[t] = self.lrp_dense(x[t], Wxh[2*d:3*d].T, Bxh[2*d:3*d], R_x)
            Rh[t-1] = self.lrp_dense(acts_lstm['h'][t-1], Whh[2*d:3*d].T, Bhh[2*d:3*d], R_h)
            
        return Rx, Rh, Rc
        
    def lrp_dense(self, hin, w, b, relevance, eps=0.001):
        
        a_p = tf.maximum(0., hin)
        a_n = tf.minimum(0., hin)
        
        w_p = tf.maximum(0., w)
        b_p = tf.maximum(0., b)
        
        z_p = tf.tensordot(np.transpose(w_p), a_p, axes=1) #+ b_p
        
        w_n = tf.minimum(0., w)
        b_n = tf.minimum(0., b)
        
        z_n = tf.tensordot(np.transpose(w_n), a_n, axes=1) #- b_n
        
        z = z_p + z_n
        
        relevance = tf.cast(relevance, z.dtype)
        s = tf.math.divide_no_nan(relevance, z)
        
        tmp_p = tf.tensordot(w_p, s, axes=1)
        tmp_n = tf.tensordot(w_n, s, axes=1)
        
        tmp_p = tf.multiply(a_p, tmp_p)
        tmp_n = tf.multiply(a_n, tmp_n)
        
        return tmp_p + tmp_n
        
    
    def lrp_add(self, hin1, hin2, relevance):
        z = hin1 + hin2
        relevance = tf.cast(relevance, z.dtype)
        s = tf.math.divide_no_nan(relevance, z)
        
        r_1, r_2 = hin1 * s, hin2 * s
        
        return r_1, r_2
    
    def selective_lrp(self, inputs, target_class):
        with tf.GradientTape() as tape:
            outputs, x, x_rev, acts_lstmleft, acts_lstmright, outputs_denseleft, outputs_denseright = self.call_and_get_activations(inputs)
            gradient_x, gradient_x_rev, gradient_left, gradient_right = tape.gradient(outputs[target_class], [x, x_rev, acts_lstmleft, acts_lstmright])
            
        gradient_left['x'] = gradient_x
        gradient_right['x'] = gradient_x_rev
        
        outputs = tf.nn.softmax(outputs)
        outputs = outputs.numpy()
        
        C = self.Why_Left.shape[0]
        
        r_mask = np.zeros((C))
        r_mask[target_class] = 1.0
        
        R_Left, R_Right = self.lrp_add(outputs_denseleft, outputs_denseright, outputs*r_mask)
        Rx_Left, Rh_Left, Rc_Left = self.selective_lrp_lstm(x, acts_lstmleft, self.Wxh_Left, self.bxh_Left, self.Whh_Left, self.bhh_Left, self.Why_Left, R_Left, gradient_left)
        Rx_Right, Rh_Right, Rc_Right = self.selective_lrp_lstm(x_rev, acts_lstmright, self.Wxh_Right, self.bxh_Left, self.Whh_Right, self.bhh_Right, self.Why_Right, R_Right, gradient_right)
        
        return Rx_Left, Rx_Right[::-1,:], tf.reduce_sum(Rh_Left[-1]+Rc_Left[-1]+Rh_Right[-1]+Rc_Right[-1])
        
    def selective_lrp_lstm(self, x, acts_lstm, Wxh, bxh, Whh, bhh, Why, relevance, gradient):
        T = len(x)
        d = int(Wxh.shape[0]/4)
        e = self.emb_weights.shape[1]
        C = Why.shape[0]
        
        Rx = np.zeros(x.shape)
        Rh = np.zeros((T+1, d))
        Rc = np.zeros((T+1, d))
        Rg = np.zeros((T, d))
        
        Rh[T-1] = self.lrp_dense(acts_lstm['h'][T-1], Why.T, np.zeros(C), relevance)
        
        for t in reversed(range(T)):
            Rc[t] += Rh[t]
            gates_i, gates_f, gates_c, gates_o = np.split(acts_lstm['gates'][t], 4)
            
            Rc[t-1], Rg[t] = self.lrp_add(gates_f*acts_lstm['c'][t-1], gates_i*gates_c, Rc[t])

            Rc[t-1] = self.selective_lrp_dense(gates_f*acts_lstm['c'][t-1], np.identity(d), np.zeros(d), Rc[t-1], gradient['c'][t-1])

            a_x = tf.tensordot(Wxh[2*d:3*d], x[t], axes=1) + bxh[2*d:3*d]
            a_h = tf.tensordot(Whh[2*d:3*d], acts_lstm['h'][t-1], axes=1) + bhh[2*d:3*d]
        
            R_x, R_h = self.lrp_add(a_x, a_h, Rg[t])
            Rx[t] = self.selective_lrp_dense(x[t], Wxh[2*d:3*d].T, bxh[2*d:3*d], R_x, gradient['x'][t])
            Rh[t-1] = self.selective_lrp_dense(acts_lstm['h'][t-1], Whh[2*d:3*d].T, bhh[2*d:3*d], R_h, gradient['h'][t-1])
            
        return Rx, Rh, Rc
        
    def selective_lrp_dense(self, hin, w, b, relevance, gradient, eps=0.001):
        if gradient is not None:
            gradient_mask = np.where(gradient > 0, 1., 0.)
            gradient_sum = gradient_mask.sum()
            hin = hin*gradient_mask
        
        a_p = tf.maximum(0., hin)
        a_n = tf.minimum(0., hin)
        
        w_p = tf.maximum(0., w)
        b_p = tf.maximum(0., b)
        
        z_p = tf.tensordot(np.transpose(w_p), a_p, axes=1) #+ b_p
        
        w_n = tf.minimum(0., w)
        b_n = tf.minimum(0., b)
        
        z_n = tf.tensordot(np.transpose(w_n), a_n, axes=1) #- b_n
        
        z = z_p + z_n
        
        relevance = tf.cast(relevance, z.dtype)
        s = tf.math.divide_no_nan(relevance, z)
        
        tmp_p = tf.tensordot(w_p, s, axes=1)
        tmp_n = tf.tensordot(w_n, s, axes=1)
        
        tmp_p = tf.multiply(a_p, tmp_p)
        tmp_n = tf.multiply(a_n, tmp_n)
        
        return tmp_p + tmp_n