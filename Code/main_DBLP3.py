# encoding: utf-8
#import matplotlib.pyplot as plt

import os
os.environ["CUDA_DEVICE_ORDER" ] ="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES" ] = "0"
import tensorflow as tf

#from tensorflow.python.ops import gen_manip_ops as _gen_manip_ops


import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn import preprocessing
from sklearn.metrics import f1_score, roc_auc_score

def get_dimension(file):
    with open(file, 'r') as f:
        first_line = f.readline()
        words = first_line.split(",")
        return len(words)

def merge_list(list1, list2):
    return list1 + list2

def temporal_attention(h_list_tensor):
    # h_list_tensor: 6400*T*M
    # V: L*M
    # W: r*L

    #A = tf.transpose(tf.nn.softmax(tf.matmul(W, tf.multiply(tf.tanh(tf.matmul(V, tf.transpose(h_list_tensor))), tf.sigmoid(tf.matmul(U, tf.transpose(h_list_tensor)))))))

    temp = tf.tanh(tf.einsum('ij,jkl->ikl', V, tf.transpose(h_list_tensor, perm=[2,1,0]))) # L*T*6400
    A = tf.nn.softmax(tf.einsum('ij,jkl->ikl', W, temp), dim=1) #r*T*6400

    return tf.transpose(A, perm=[2,0,1]) #6400*r*T

def normalize(v):
    return tf.nn.l2_normalize(v, axis=1, epsilon=1e-12)

def mi_gru(Fea_batch, idx_batch, samples_idx_batch, support_sizes_batch, n_hidden_units, num_stacked_layers, weights, biases, t_dim):
    # weights['out']: (r*n_hidden_units)*n_classes
    # biases['out']:  n_classes

    h_list_tensor, Beta = GRU_Neighbor(Fea_batch, idx_batch, samples_idx_batch, support_sizes_batch, n_hidden_units, num_stacked_layers) #6400*T*M; T*6400*[0]*(1+[1]/[0])
    Alpha = temporal_attention(h_list_tensor) #6400*r*T
    temp  = tf.matmul(Alpha, h_list_tensor) #6400*r*M
    temp1 = tf.reshape(temp, [-1, tf.cast(temp.get_shape()[1].value*temp.get_shape()[2].value, tf.int32)]) #(6400)*(r*M)

    return tf.matmul(temp1, weights['out']) + biases['out'], h_list_tensor, Alpha, temp, Beta #6400*num_classes

def GRU_Neighbor(Fea, idx, saps_idx, sup_sizes, n_h_units, num_stacked_layers):  #X_Nebr, n_h_units, num_stacked_layers

    #d = tf.cast(n_hidden_units/n_dim, tf.int32) # d = 4
    T     = Fea.get_shape()[1].value # T = 50
    n_dim = Fea.get_shape()[2].value # n_dim = 42
    idx = tf.cast(idx, tf.int32)
    X     = tf.gather(Fea, idx) # shape: 6400*50*42
    #XN    = X_Nebr[:,:,n_dim:n_dim*2] # 6400*50*42

    #print(len(sup_sizes))
    for i in range(T):
        #print("index i in LSTM:", i)
        xt_temp  = X[:,i,:] # 6400*42
        xt       = tf.transpose(xt_temp) # 42*6400
        
        # shapes: 10000*42, 6400*(1+[0]+[1]), [[0],[1]], 6400
        xnt, Beta_step = aggregate_att_mean(Fea[:,i,:], saps_idx[i,:,:], sup_sizes[i], idx) # shape: 42*6400; 6400*[0]*(1+[1]/[0])

        if i == 0:
            rt2_temp = tf.matmul(Wr2_2, tf.concat([xt, xnt], 0)) # (42*84)*((42+42)*6400)=42*6400
            rt2      = tf.transpose(tf.sigmoid(tf.transpose(rt2_temp) + br2)) # 6400*42 + 42 -> 42*6400

            ht_temp  = tf.matmul(Wh_2, tf.concat([xt, tf.multiply(rt2, xnt)], 0)) # (20*84)*(84*6400)=20*6400
            ht_tilde = tf.transpose(tf.tanh(tf.transpose(ht_temp) + bh)) # 6400*20 + 20 -> 20*6400
            ht       = ht_tilde

            hp       = ht #20*6400
            h_list_tensor = tf.expand_dims(hp, 1) # 20*1*6400

            Beta = tf.expand_dims(Beta_step, 0) # 1*6400*[0]*(1+[1]/[0])

        else:
            Wr2   = tf.concat([Wr2_1,Wr2_2], 1) # [42*20,42*84] -> 42*104
            Wh    = tf.concat([Wh_1,Wh_2], 1)   # [20*20,20*84] -> 20*104
            Wzr   = tf.concat([Wz,Wr1,Wr2], 0)  # (20+20+42)*104
            bzr   = tf.concat([bz,br1,br2], 0)  # (20+20+42)

            zr_temp = tf.matmul(Wzr, tf.concat([hp, xt, xnt], 0)) # (82*104)*((20+42+42)*6400)=82*6400
            zr      = tf.transpose(tf.sigmoid(tf.transpose(zr_temp) + bzr)) # (20+20+42)*6400: zt, rt1, rt2

            rt1      = zr[n_h_units:n_h_units*2,:] # 20*6400
            rt2      = zr[n_h_units*2:n_h_units*2+n_dim, :] # 42*6400
            ht_temp  = tf.matmul(Wh, tf.concat([tf.multiply(rt1, hp), xt, tf.multiply(rt2, xnt)], 0)) # 20*6400
            ht_tilde = tf.transpose(tf.tanh(tf.transpose(ht_temp) + bh)) # 6400*20 + 20 -> 20*6400

            zt       = zr[0:n_h_units,:] # 20*6400
            zt_neg   = 1.0 - zt
            ht       = tf.multiply(zt_neg, hp) + tf.multiply(zt, ht_tilde)

            hp       = ht # 20*6400
            h_list_tensor = tf.concat([h_list_tensor, tf.expand_dims(hp, 1)], 1) # 20*T*6400

            Beta = tf.concat([Beta, tf.expand_dims(Beta_step, 0)], 0)  # (T)*6400*[0]*(1+[1]/[0])

    h_list_tensor = tf.transpose(h_list_tensor, perm=[2, 1, 0]) # 6400*T*20 (M=20)

    return h_list_tensor, Beta

def aggregate_mean(fea_mat,  samples,  sup_sizes, targets):
    # shapes: 10000*42, 6400*(1+[0]+[1]), [[0],[1]], 6400
	# return xnt: 42*6400

    sup_sizes = [1] + sup_sizes # [1, [0], [1]]
    #for i in range(len(sup_sizes)-1,0,-1): # '-1 0 -1'
    i = 2
    dim_temp = np.int32(sup_sizes[i]/sup_sizes[i-1])
    samples_temp = tf.reshape(samples[:,-sup_sizes[i]:], [-1, sup_sizes[i-1], dim_temp]) # shape: 6400*[0]*([1]/[0])
    fea_mat_samples_temp = tf.nn.embedding_lookup(fea_mat, samples_temp) # shape: 6400*[0]*([1]/[0])*42
    fea_hop_mean = tf.reduce_mean(fea_mat_samples_temp, 2) # shape: 6400*[0]*42

    #fea_mat_samples_temp_1 = tf.sigmoid(tf.matmul(weights_hops[str(i-1)], tf.transpose(fea_hop_mean, perm=[2, 1, 0]))) # shape: (42,42)*(42*[0]*6400)=42*[0]*6400
    fea_mat_samples_temp_1 = tf.sigmoid(tf.einsum('ij,jkl->ikl', weights_hops[str(i-1)], tf.transpose(fea_hop_mean, perm=[2, 1, 0]))) # shape: (42,42)*(42*[0]*6400)=42*[0]*6400
    fea_hop_mean_1 = tf.reduce_mean(tf.transpose(fea_mat_samples_temp_1, perm=[2, 1, 0]), 1) # shape: 6400*42
    res = tf.sigmoid(tf.matmul(weights_hops[str(i-2)], tf.transpose(fea_hop_mean_1, perm=[1, 0]))) # shape: (42*42)*(42*6400)=42*6400

    return res

def spatial_attention_1(f_tar_samp):
    # f_tar_samp: 6400*[0]*([1]/[0])*84; 6400*1*[0]*84
    # V1_h1: L11*n_dim
    # w1_h1: (2*L11)
    temp_sp1 = tf.einsum('ij,jklm->iklm', V1_h1, tf.transpose(f_tar_samp[:,:,:,0:n_dim], perm=[3,2,1,0])) # L11*([1]/[0])*[0]*6400
    temp_sp2 = tf.einsum('ij,jklm->iklm', V1_h1, tf.transpose(f_tar_samp[:,:,:,n_dim:2*n_dim], perm=[3,2,1,0])) # L11*([1]/[0])*[0]*6400
    B = tf.nn.softmax(tf.nn.leaky_relu(tf.einsum('j,jklm->klm', w1_h1, tf.concat([temp_sp1, temp_sp2], 0))), dim=0) #([1]/[0])*[0]*6400

    return tf.transpose(B, perm=[2,1,0]) #6400*[0]*([1]/[0]); 6400*1*[0]

def spatial_attention_0(f_tar_samp):
    # f_tar_samp: (2*dim_hop0)*[0]*6400
    # V1_h0: L10*L11
    # w1_h0: (2*L10)
    # dim_hop0 = L11
    
    temp_sp1 = tf.einsum('ij,jkl->ikl', V1_h0, f_tar_samp[0:dim_hop0,:,:]) # L10*[0]*6400
    temp_sp2 = tf.einsum('ij,jkl->ikl', V1_h0, f_tar_samp[dim_hop0:2*dim_hop0,:,:]) # L10*[0]*6400
    B = tf.nn.softmax(tf.nn.leaky_relu(tf.einsum('j,jkl->kl', w1_h0, tf.concat([temp_sp1, temp_sp2], 0))), dim=0) # [0]*6400

    return tf.transpose(B, perm=[1,0]) # 6400*[0]

def aggregate_att_mean(fea_mat,  samples,  sup_sizes, targets):
    # shapes: 10000*42, 6400*(1+[0]+[1]), [[0],[1]], 6400
    # return xnt: 42*6400

    sup_sizes = [1] + sup_sizes # [1, [0], [1]]
    ## k=1: update node representations of Hop-0
    dim_temp = np.int32(sup_sizes[2]/sup_sizes[1])
    # gather neighbors at hop-1
    samples_temp = tf.reshape(samples[:,-sup_sizes[2]:], [-1, sup_sizes[1], dim_temp]) # shape: 6400*[0]*([1]/[0])
    fea_mat_samples_temp = tf.nn.embedding_lookup(fea_mat, samples_temp) # shape: 6400*[0]*([1]/[0])*42

    # gather target nodes for hop-1 and replicate attributes
    samples_temp_t = samples[:,-(sup_sizes[1]+sup_sizes[2]):-sup_sizes[2]] # shape: 6400*[0]
    target_temp = tf.expand_dims(samples_temp_t, 2) # 6400*[0]*1
    target_temp = tf.tile(target_temp, [1,1,dim_temp]) # 6400*[0]*([1]/[0])
    fea_mat_samples_temp_t = tf.nn.embedding_lookup(fea_mat, target_temp) # shape: 6400*[0]*([1]/[0])*42

    # learn attention for hop-1
    fea_tar_samp = tf.concat([fea_mat_samples_temp_t, fea_mat_samples_temp], 3) # shape: 6400*[0]*([1]/[0])*84
    Beta_hop1 = spatial_attention_1(fea_tar_samp) # shape: 6400*[0]*([1]/[0])

    # calculate neighbor representation
    #   V1_h1: L11*n_dim
    #   fea_mat_samples_temp: 6400*[0]*([1]/[0])*42
    temp_aam_0 = tf.einsum('ij,jklm->iklm', V1_h1, tf.transpose(fea_mat_samples_temp, perm=[3,2,1,0])) # shape: L11*([1]/[0])*[0]*6400
    temp_aam_1 = tf.matmul(tf.expand_dims(Beta_hop1, 2), tf.transpose(temp_aam_0, perm=[3,2,1,0])) # shape: 6400*[0]*1*L11 ???
    fea_hop_mean_att = tf.squeeze(temp_aam_1, squeeze_dims=2) # shape: 6400*[0]*L11

    # concatenate and transform
    fea_mat_samples_temp_t_emb = tf.einsum('ij,jkl->ikl', V1_h1, tf.transpose(fea_mat_samples_temp_t[:,:,0,:], perm=[2,1,0])) # L11*[0]*6400
    con_hop1 = tf.concat([fea_mat_samples_temp_t_emb, tf.transpose(fea_hop_mean_att, perm=[2,1,0])], 0) # shape:(L11*2)*[0]*6400, dim_hop1=L11*2
    hop1 = tf.sigmoid(tf.einsum('ij,jkl->ikl', weights_hops['1'], con_hop1)) # shape: (dim_hop0,dim_hop1)*((L11*2),[0]*6400)=dim_hop0*[0]*6400

    ## k=1: update node representations of nodes in batch
    dim_temp = np.int32(sup_sizes[1])
    # gather neighbors at hop-0
    samples_temp = samples[:,-(sup_sizes[1]+sup_sizes[2]):-sup_sizes[2]] # shape: 6400*[0]
    fea_mat_samples_temp = tf.nn.embedding_lookup(fea_mat, samples_temp) # shape: 6400*[0]*42
    # gather target nodes for hop-0 and replicate attributes
    samples_temp_t = samples[:,0] # shape: 6400*1
    target_temp = tf.tile(tf.expand_dims(samples_temp_t, 1), [1,dim_temp]) # 6400*[0]
    fea_mat_samples_temp_t = tf.nn.embedding_lookup(fea_mat, target_temp) # shape: 6400*[0]*42
    # learn attention for hop-0
    fea_tar_samp = tf.concat([fea_mat_samples_temp_t, fea_mat_samples_temp], 2) # shape: 6400*[0]*84
    Beta_hop0 = spatial_attention_1(tf.expand_dims(fea_tar_samp, 1)) # shape: 6400*1*[0]
    # calculate neighbor representation
    #   V1_h1: L11*n_dim
    #   fea_mat_samples_temp: 6400*[0]*42
    temp_aam_0 = tf.einsum('ij,jkl->ikl', V1_h1, tf.transpose(fea_mat_samples_temp, perm=[2,1,0])) # shape: L11*[0]*6400
    temp_aam_1 = tf.matmul(Beta_hop0, tf.transpose(temp_aam_0, perm=[2,1,0])) # shape: 6400*1*L11 ???
    fea_hop_mean_att = tf.squeeze(temp_aam_1, squeeze_dims=1) # shape: 6400*L11
    
    # concatenate and transform
    fea_mat_samples_temp_t_emb = tf.einsum('ij,jk->ik', V1_h1, tf.transpose(fea_mat_samples_temp_t[:,0,:], perm=[1,0])) # L11*6400
    con_hop1 = tf.concat([fea_mat_samples_temp_t_emb, tf.transpose(fea_hop_mean_att, perm=[1,0])], 0) # shape: (L11*2)*6400, dim_hop1=L11*2
    hop0 = tf.sigmoid(tf.einsum('ij,jk->ik', weights_hops['1'], con_hop1)) # shape: (dim_hop0,dim_hop1)*((L11*2)*6400)=dim_hop0*6400


    ## k=2: generate neighbor representation
    dim_temp = np.int32(sup_sizes[1])
    # gather target nodes for hop-0 and replicate attributes
    target_temp = tf.expand_dims(hop0, 1) # dim_hop0*1*6400
    target_temp = tf.tile(target_temp, [1,dim_temp,1]) # dim_hop0*[0]*6400
    # learn attention
    fea_tar_samp = tf.concat([target_temp, hop1], 0) # shape: (2*dim_hop0)*[0]*6400
    Beta_hop = spatial_attention_0(fea_tar_samp) # shape: 6400*[0]

    # calculate neighbor representation
    #   V1_h0: L10*dim_hop0
    #   hop1: dim_hop0*[0]*6400
    temp_aam_0 = tf.einsum('ij,jkl->ikl', V1_h0, hop1) # shape: L10*[0]*6400
    temp_aam_1 = tf.matmul(tf.expand_dims(Beta_hop, 1), tf.transpose(temp_aam_0, perm=[2,1,0])) # shape: 6400*1*L10 ???
    fea_hop_mean_att = tf.squeeze(temp_aam_1, squeeze_dims=1) # shape: 6400*L10

    ## concatenate spatial attention
    # Beta_hop1: 6400*[0]*([1]/[0])
    # Beta_hop: 6400*[0]
    Beta_step = tf.concat([tf.expand_dims(Beta_hop, 2), Beta_hop1], 2) # 6400*[0]*(1+[1]/[0])

    return tf.transpose(fea_hop_mean_att), Beta_step  #L10*6400 = 42*6400; 6400*[0]*(1+[1]/[0])

def minmax_normalization_tensor(tensor):
    # tensor: n_node, n_steps, n_dim
    n_node, n_steps, n_dim = tensor.shape
    tensor_norm = np.ones([n_node, n_steps, n_dim])
    for i in range(tensor.shape[1]):
        mat = tensor[:,i,:]
        max_val = np.max(mat, 0) # shape: n_dim
        min_val = np.min(mat, 0)
        mat_norm = (mat - min_val) / (max_val - min_val + 1e-12)

        tensor_norm[:,i,:] = mat_norm

    # print norm_x
    return tensor_norm

def meanstd_normalization_tensor(tensor):
    # tensor: n_node, n_steps, n_dim
    n_node, n_steps, n_dim = tensor.shape
    #tensor_norm = np.ones([n_node, n_steps, n_dim])
    tensor_reshape = preprocessing.scale(np.reshape(tensor, [n_node, n_steps*n_dim]), axis=1)
    tensor_norm = np.reshape(tensor_reshape, [n_node, n_steps, n_dim])
        
    # print norm_x
    return tensor_norm

def get_Batch(data, label, batch_size, n_epochs):
    #print(data.shape, label.shape)
    input_queue = tf.train.slice_input_producer([data, label], num_epochs=n_epochs, shuffle=True, capacity=1000) 
    x_batch, y_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=1000, allow_smaller_final_batch=True)
    return x_batch, y_batch

def Neightbor_aggre(node_attr, k): #node_attr: N_tr*50*42
    N, T, fea = np.shape(node_attr)
    for i in range(T):
        frame = node_attr[:,i,:]
        nbrs  = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(frame)
        _, indices = nbrs.kneighbors() #N_tr*k; '()' will not include the node itself as its neighbor
        frame_nbrs = frame[indices] #N_tr*k*42
        aggr_frame = np.mean(frame_nbrs, axis=1) #N_tr*42
        if i == 0:
            aggr = aggr_frame
        else:
            aggr = np.stack((aggr, aggr_frame), axis=1) 

    return aggr #N_tr*50*42

def generate_graph(fea_mat, n_nbor):
    # euclidean equals to pearson corrcoef after normalization
    graph = kneighbors_graph(fea_mat, n_nbor, mode='connectivity', metric='euclidean', include_self=False).toarray()

    return graph

def construct_adj(graph, n_sample):
    # graph: n_node*n_node
    # '1' indicates connection, '0' for no-connection
    # return adj: n_node*n_sample
    n_node = graph.shape[0]
    adj = (-1)*np.ones((n_node, n_sample))
    for i in range(n_node):
        idx_nbor = np.nonzero(graph[i])[0]
        if len(idx_nbor) == 0:
            continue
        if len(idx_nbor) > n_sample:
            idx_nbor = np.random.choice(idx_nbor, n_sample, replace=False)
        elif len(idx_nbor) < n_sample:
            idx_nbor = np.random.choice(idx_nbor, n_sample, replace=True)
        adj[i,:] = idx_nbor

    return adj

def sample_tf(inputs, n_layers, sample_sizes, adj_tensor, n_steps):
    # inputs: batch of nodes; adj_tensor: adj_mats for all time steps
    # return: 
    #   samples_tensor    : (n_steps)*inputs*(1+[0]+[1])
    #   support_sizes_list: a list of lists

    support_sizes_list = []
    for t in range(n_steps):
        samples = inputs
        input_temp = inputs
        support_sizes = []
        adj_mat = adj_tensor[t]

        #print(adj_mat)
        #print(inputs)

        support_size = 1

        samples_frame = tf.reshape(input_temp, [-1, 1])
        samples_frame = tf.cast(samples_frame, tf.int32)
        for i in range(n_layers):
            support_size *= sample_sizes[i]
            support_sizes.append(support_size)
            samples = tf.cast(samples, tf.int32)
            neighs = tf.gather(adj_mat, samples) # shape: samples*n_sample
            neighs = tf.random_shuffle(tf.transpose(neighs)) #shuffle columns

            samples = tf.reshape(neighs, [-1])
            samples_frame = tf.concat([samples_frame, tf.reshape(neighs, [-1, support_sizes[i]])], 1) # shape: inputs*support_sizes
            
        if t == 0:
            samples_tensor = tf.expand_dims(samples_frame, 0) # shape: 1*inputs*(1+support_sizes[0]+support_sizes[1])
        samples_tensor = tf.concat([samples_tensor, tf.expand_dims(samples_frame, 0)], 0) # shape: (1+1)*inputs*(1+[0]+[1])

        support_sizes_list.append(support_sizes)

    return samples_tensor, support_sizes_list


def normalize(v):
    return tf.nn.l2_normalize(v, axis=1, epsilon=1e-12)

def eval(y_true, y_pred):
    # both size: 6400*num_classes
    
    # ACC
    correct_pred = np.equal(np.argmax(y_pred, 1), np.argmax(y_true, 1))
    accuracy = np.mean(correct_pred.astype(float))

    # AUC
    auc_mac = roc_auc_score(y_true, y_pred, average='macro')
    auc_mic = roc_auc_score(y_true, y_pred, average='micro')
    
    # F1
    f1_mac = f1_score(np.argmax(y_true, 1), np.argmax(y_pred, 1), average='macro')
    f1_mic = f1_score(np.argmax(y_true, 1), np.argmax(y_pred, 1), average='micro')


    return accuracy, auc_mac, auc_mic, f1_mac, f1_mic

if __name__ == '__main__':

    filedpath = '.../Data'
    filename  = '.../DBLP3.npz'
    file      = np.load(filedpath+filename)

    Features  = file['attmats'] #(n_node, n_time, att_dim)
    Labels    = file['labels']  #(n_node, num_classes)
    Graphs    = file['adjs']    #(n_time, n_node, n_node)

    Features = meanstd_normalization_tensor(Features)

    n_hidden_units = 10
    n_hop = 2
    n_sample = 4    
    sample_sizes = [n_sample, n_sample]
    n_nbor = 40

    n_node, n_steps, n_dim = np.shape(Features)
    num_classes = 3 
    tmp_dim = num_classes

    batch_size = 2000
    lr = 0.0025
    training_iters = 500
    num_stacked_layers = 1
    display_step = 1
    lambda_l2_reg = 1e-3
    lambda_reg_att = 1e-5

    M  = n_hidden_units
    # temporal attention
    r  = 3
    L  = 10
    # spatial attention
    L10 = n_dim
    L11 = 10
    # dims of hops
    dim_hop0 = L11
    dim_hop1 = L11*2

    # add self-loop
    for i in range(n_steps):
        Graphs[i,:,:] += np.eye(n_node, dtype=np.int32)

    Data_idx = np.arange(n_node)
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(Data_idx, Labels, test_size=0.1) #N_tr, N_te
    X_train_idx, X_val_idx, y_train, y_val = train_test_split(X_train_idx, y_train, test_size=0.1) #N_tr, N_te

    tf.reset_default_graph()
    
    weights_att = {'W': tf.get_variable('Weights_W', shape=[r,L], dtype=tf.float64, \
                        initializer=tf.truncated_normal_initializer(mean=-0.1, stddev=0.1)), \
                    'V': tf.get_variable('Weights_V', shape=[L,M], dtype=tf.float64, \
                         initializer=tf.truncated_normal_initializer(mean=-0.1, stddev=0.1)), \
                        'w1_h1': tf.get_variable('Weights_w1_h1', shape=[2*L11], dtype=tf.float64, \
                        initializer=tf.truncated_normal_initializer(mean=-0.1, stddev=0.1)), \
                        'V1_h1': tf.get_variable('Weights_V1_h1', shape=[L11,n_dim], dtype=tf.float64, \
                         initializer=tf.truncated_normal_initializer(mean=-0.1, stddev=0.1)), \
                        'w1_h0': tf.get_variable('Weights_w1_h0', shape=[2*L10], dtype=tf.float64, \
                        initializer=tf.truncated_normal_initializer(mean=-0.1, stddev=0.1)), \
                        'V1_h0': tf.get_variable('Weights_V1_h0', shape=[L10,L11], dtype=tf.float64, \
                         initializer=tf.truncated_normal_initializer(mean=-0.1, stddev=0.1))}

    W, V = weights_att['W'], weights_att['V']
    w1_h0, V1_h0 = weights_att['w1_h0'], weights_att['V1_h0']
    w1_h1, V1_h1 = weights_att['w1_h1'], weights_att['V1_h1']


    Wz    = tf.Variable(tf.truncated_normal(shape=[M, M+n_dim*2], mean=-0.1, stddev=0.1, dtype=tf.float64)) # 20*(20+42*2)
    Wr1   = tf.Variable(tf.truncated_normal(shape=[M, M+n_dim*2], mean=-0.1, stddev=0.1, dtype=tf.float64)) # 20*(20+42*2)
    Wr2_1 = tf.Variable(tf.truncated_normal(shape=[n_dim, M], mean=-0.1, stddev=0.1, dtype=tf.float64)) # 42*20
    Wr2_2 = tf.Variable(tf.truncated_normal(shape=[n_dim, n_dim*2], mean=-0.1, stddev=0.1, dtype=tf.float64)) # 42*(42*2)
    Wh_1  = tf.Variable(tf.truncated_normal(shape=[M, M], mean=-0.1, stddev=0.1, dtype=tf.float64)) # 20*20
    Wh_2  = tf.Variable(tf.truncated_normal(shape=[M, n_dim*2], mean=-0.1, stddev=0.1, dtype=tf.float64)) # 20*(42*2)

    bz  = tf.Variable(tf.constant(0.0, shape=[M], dtype=tf.float64)) # 20
    br1 = tf.Variable(tf.constant(0.0, shape=[M], dtype=tf.float64)) # 20
    br2 = tf.Variable(tf.constant(0.0, shape=[n_dim], dtype=tf.float64)) # 42
    bh  = tf.Variable(tf.constant(0.0, shape=[M], dtype=tf.float64)) # 20

    weights = {'out': tf.get_variable('Weights_out', shape=[r*M, num_classes], \
                                        dtype=tf.float64, initializer=tf.truncated_normal_initializer(mean=-0.1, stddev=0.1))}
    biases  = {'out': tf.get_variable('Biases_out', shape=[num_classes], dtype=tf.float64, \
                                     initializer=tf.constant_initializer(0.0))}

    weights_hops = {'0': tf.get_variable('Weights_hop_0', shape=[n_dim, dim_hop0], dtype=tf.float64, \
                                     initializer=tf.truncated_normal_initializer(mean=-0.1, stddev=0.1)), \
    				'1': tf.get_variable('Weights_hop_1', shape=[dim_hop0, dim_hop1], dtype=tf.float64, \
					               	initializer=tf.truncated_normal_initializer(mean=-0.1, stddev=0.1))}

    # placeholder
    x_idx = tf.placeholder(tf.int64, [None,])
    y     = tf.placeholder(tf.int64, [None, num_classes])

    # construct adjacent matrix
    adj_tensor = np.ones((n_steps, n_node, n_sample), dtype=np.int32)
    for i in range(n_steps):
        adj_tensor[i,:,:] = construct_adj(Graphs[i], n_sample)
    adj_tensor = tf.convert_to_tensor(adj_tensor, dtype=tf.int32)

    samples_idx, support_sizes = sample_tf(x_idx, n_hop, sample_sizes, adj_tensor, n_steps) # samples_idx: (n_steps)*inputs*(1+[0]+[1])
    Features_tf = tf.convert_to_tensor(Features, dtype=tf.float64) # n_node*n_steps*n_dim

    #6400*num_classes, 6400*T*M, 6400*r*T, 6400*r*M, T*6400*[0]*(1+[1]/[0])
    logits_batch, h_list_batch, Alpha, embedding_att, Beta = mi_gru(Features_tf, x_idx, samples_idx, support_sizes, n_hidden_units, num_stacked_layers, weights, biases, tmp_dim) 
    prediction = tf.nn.softmax(logits_batch) #softmax row by row, 6400*num_classes

    # L2 regularization for weights and biases
    reg_loss = 0
    for tf_var in tf.trainable_variables():
        reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

    # regularization for different hops of attentions
    # Alpha: 6400*r*20
    reg_att_temp = tf.matmul(Alpha, tf.transpose(Alpha, perm=[0, 2, 1])) #6400*r*r
    I_mat = tf.convert_to_tensor(np.eye(r), dtype=tf.float64)
    reg_att = tf.reduce_mean(tf.nn.l2_loss(reg_att_temp - I_mat))

    # Define loss and optimization
    loss_op  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_batch, labels=y))
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_op + lambda_l2_reg*reg_loss + lambda_reg_att*reg_att)

    saver = tf.train.Saver()

    x_batch_idx, y_batch = get_Batch(X_train_idx, y_train, batch_size, training_iters)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        try:
            ite_tag = 0
            train_loss = 0
            List_tr_B_loss = []

            while not coord.should_stop():
                x_batch_idx_feed, y_batch_feed= sess.run([x_batch_idx, y_batch])

                # training
                train_op.run(feed_dict={x_idx:x_batch_idx_feed, y:y_batch_feed})

                fetch = {'loss_op':loss_op, 'x_idx': x_idx, 'y':y, 'prediction':prediction, 'logits_batch':logits_batch, 'h_list_batch':h_list_batch, 'reg_loss':reg_loss, 'reg_att':reg_att, 'Alpha':Alpha, 'Beta':Beta, 'samples_idx':samples_idx} # Alpha: 6400*r*T; Beta: T*6400*[0]*(1+[1]/[0])
                
                # training result
                Res = sess.run(fetch, feed_dict={x_idx:x_batch_idx_feed, y:y_batch_feed})
                tr_acc, tr_auc1, tr_auc2, tr_f11, tr_f12 = eval(Res['y'], Res['prediction'])

                # validation resilt
                Res_val = sess.run(fetch, feed_dict={x_idx:X_val_idx, y:y_val})
                val_acc, val_auc1, val_auc2, val_f11, val_f12 = eval(Res_val['y'], Res_val['prediction'])

                if ite_tag % display_step == 0 or ite_tag == 0:
                    print("Epoch %d, Ite %d, tr_loss=%g, L2=%g, L2_att=%g, val_loss=%g" % (ite_tag//(len(X_train_idx)//batch_size + 1), ite_tag, Res['loss_op'], Res['reg_loss'], Res['reg_att'], Res_val['loss_op']))
                    print("Epoch %d, Ite %d, tr_acc=%g, tr_auc1=%g, tr_auc2=%g, tr_f11=%g, tr_f12=%g, val_acc=%g, val_auc1=%g, val_auc2=%g, val_f11=%g, val_f12=%g" % (ite_tag//(len(X_train_idx)//batch_size + 1), ite_tag, tr_acc, tr_auc1, tr_auc2, tr_f11, tr_f12, val_acc, val_auc1, val_auc2, val_f11, val_f12))

                ite_tag += 1

                List_tr_B_loss = np.append(List_tr_B_loss, Res['loss_op'])

        except tf.errors.OutOfRangeError:
            print("---Train end---")
        finally:
            coord.request_stop()
            print('---Programm end---')
        coord.join(threads)


        # testing
        Res_test = sess.run(fetch, feed_dict={x_idx:X_test_idx, y:y_test})
        te_acc, te_auc1, te_auc2, te_f11, te_f12 = eval(Res_test['y'], Res_test['prediction'])
        print("test_acc=%g, test_auc1=%g, test_auc2=%g, test_f11=%g, test_f12=%g" % (te_acc, te_auc1, te_auc2, te_f11, te_f12))

        # save results
        np.savez(".../results/DBLP4000.npz", List_tr_B_loss=List_tr_B_loss, Alpha=Res_test['Alpha'], Beta=Res_test['Beta'], X_test_idx=X_test_idx, y_test=y_test, samples_idx=Res_test['samples_idx'])


