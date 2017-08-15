# This is the network structure for the tf implementation of dietnet. 

""" notes on the the structure:
    - if there is batch norm and dropout on a layer there is no need to regularize.
    - auxnet: perhaps add another layer if we get a single layer to learn.
    - discnet:  add batch_norm and/or dropout to the layers in the disc net. 
"""
import os
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
layers = tf.contrib.layers


def hist_embedding(path):
    """
    loads the embedding in order to save as a Variable for the graph. It makes more sense
    to do this than to pass as a placeholder. 
    Args:
        path: path to the directory that histo3x26.npy exists
    returns:
        np.load(): loaded np array
    """
    return np.load(os.path.join(path, 'histo3x26.npy'))


def auxnet(embedding, size, dropout_rate=.5, std=.2, is_training=True, scope='auxnet'):
    """
    Defines the fully connected layers for the auxnet: 
        -- so far, one layer to batch norm to relu to dropout
    Args:
        embedding: the histogram embedding matrix
        size: int size of each hidden layer
        dropout_rate: rate to dropout (usually .5)
        std: standard deviation used for initilizer 
        is_training: bool--used to turn off dropout for inference
        scope: name the op/tensor
    Returns:
        fc: the fully connected network as a tensor of size (pxsize)
    """
    # make lower/upper for uniform init
    a,b = 0 - np.sqrt(3)*std, np.sqrt(3)*std

    with tf.variable_scope(scope,'Aux'):
	# notes: if you use dropout and batchnorm no need for regularizer
        with slim.arg_scope([slim.fully_connected],
                weights_initializer = tf.random_uniform_initializer(minval=a,maxval=b),
                #weights_initializer = tf.truncated_normal_initializer(std),
	        weights_regularizer = slim.l2_regularizer(.005),
		activation_fn=tf.nn.relu):
            
            """
            net = slim.fully_connected(embedding, size, scope='hidden')
            net = slim.dropout(net, dropout_rate,
                    is_training=is_training, scope='dropout')
            net= slim.fully_connected(net, size, scope='output',
                    activation_fn=None)
	    """
	    fc  = slim.fully_connected(embedding, size,
					 biases_initializer=tf.zeros_initializer(), 
                                         activation_fn=None, #tf.nn.relu,
                                         scope='hidden')
	    #tf.summary.histogram('beforebn/%s' % scope, fc, collections=['train'])
            

	    fc = slim.batch_norm(fc, center=True, 
                                    scale=True, 
			      	    zero_debias_moving_mean=True,
				    is_training=is_training,
                                    scope='bn')
            
            # mod option: add another layer here:
            fc = tf.nn.relu(fc, 'relu')
    	    
            # now apply the dropout:
            fc = slim.dropout(fc, dropout_rate,
			      is_training=is_training, 
			      scope='dropout')
	
	    # add another layer:	
	    fc = slim.fully_connected(fc, size, biases_initializer=tf.zeros_initializer(),
					activation_fn=tf.nn.tanh, scope="hidden2")

    #tf.summary.histogram('activations/auxnet/%s' % scope, fc, collections=['train'])

    return fc 


def dietnet(path=None,
            input_size=None,
            batch_rate=0,
	    output_size=26,
            embed_size=None,
            hidden_size=100,
            dropout_rate=0.5,
            is_training=True,
            std=.2,
            gamma=10,
            scope=None):
    """ 
    Defines the discriminative network. Depends on the auxnet and hist_embedding. Will 
    produce the predictions and make reconstructed input.
    Args:
        path: path to where the npy files live
        input_size: dim of the input (num of SNPs)
        output_size: dim of the one-hot class vectors (num of classes)
        embed_size: same as num SNPs
        hidden_size: ...
        dropout_rate: ...
        is_training: bool--for training or inference
        std: the standard deviation for the weight init
        gamma: weight the reconstruction loss in the total loss
        scope: ...
    Returns:
        loss: the total loss to be optimized
    """
    a,b = 0 - np.sqrt(3)*std, np.sqrt(3)*std

    with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
        
        # asser that the row of embed==input_size
        assert embed_size==input_size, "row of embeddings neq dim SNPs"

        embedding = hist_embedding(path)
	embedding = embedding.astype('float32')
        # define the placeholders:
        inputs  = tf.placeholder(tf.float32, shape=[None,  input_size], name='inputs')
        outputs = tf.placeholder(tf.float32, shape=[None, output_size], name='outputs')
        is_training = tf.placeholder(tf.bool, name='is_training')
	embed = tf.constant(embedding,tf.float32, name='embed_mat')
        
	# make embedding weight matrix net        
        We = auxnet(embed, hidden_size, dropout_rate=dropout_rate,
                    std=std, is_training=is_training, scope='aux_We')
        
        # clip by norm: paper suggest to restrict the norm of weights to 1
        We = tf.clip_by_norm(We, clip_norm=3)

        #tf.summary.histogram('weights/dietnet/We/', We, collections=['train'])

        # "first" MLP layer in the discriminative network
        out_mlp1 = tf.matmul(inputs, We)
        out_mlp1 = slim.bias_add(out_mlp1, activation_fn=tf.nn.tanh, scope="mlp1")
        out_mlp1 = layers.batch_norm(out_mlp1, center=True, scale=True,
				     zero_debias_moving_mean=True,
				     is_training=is_training, 
				     scope='bn')
	out_mlp1 = tf.nn.relu(out_mlp1,'relu')
	#tf.summary.histogram('activations/dietnet/mlp1', out_mlp1, collections=['train'])
        out_mlp1 = slim.dropout(out_mlp1, 
				dropout_rate,
				is_training=is_training)
        
	# Final MLP in the discriminative network:
        """get the logits: the idea is to use activation_fn as None => linear
           then pass to softmax_cross_enctropy for speed purposes.""" 
        with tf.variable_scope(scope, 'Dietnet'):
            #pred_logits = slim.fully_connected(out_mlp1, output_size,
            #                              activation_fn=None,
            #                              scope='output')
	
            fc  = layers.fully_connected(out_mlp1, hidden_size,
                           biases_initializer=tf.zeros_initializer(), 
                           weights_initializer = tf.random_uniform_initializer(minval=a,maxval=b),
			   activation_fn=None,
                           scope='hidden')

            fc = layers.batch_norm(fc, center=True, 
                                    scale=True, 
                                    is_training=is_training,
                                    scope='bn')
            
            # mod option: add another layer here:
            fc = tf.nn.relu(fc, 'relu')
            
            # now apply the dropout:
            fc = slim.dropout(fc, dropout_rate, is_training=is_training, scope='dropout')
	    # add another layer:	
            pred_logits = layers.fully_connected(fc, output_size, 
                        weights_initializer = tf.random_uniform_initializer(minval=a,maxval=b),   
		        weights_regularizer = slim.l2_regularizer(.1),
			biases_initializer=tf.zeros_initializer(),
                        activation_fn=None, scope="output")
        
	#tf.summary.histogram('finalmlp',pred_logits, collections = ['train'])
        ent_loss = tf.losses.softmax_cross_entropy(outputs, pred_logits)
        tf.summary.scalar('loss/cross_entropy', ent_loss, collections=['train'])
	# get the accuracy. argmax,1 gives the index of the largest value
        accuracy = slim.metrics.accuracy(tf.argmax(pred_logits,1),
                                         tf.argmax(outputs,1))
        
	# build the autoencoder (not sharing params, just embedding)
        Wd = auxnet(embed, hidden_size, dropout_rate=dropout_rate, 
                    std=std, is_training=is_training, scope='aux_Wd')

        # again, paper suggests restricting the norm of weights to 1
        Wd = tf.clip_by_norm(Wd, clip_norm=3)
        #tf.summary.histogram('weights/dietnet/Wd', Wd, collections=['train'])

        # mlp for the reconstruction:
        rec_x = tf.matmul(fc,Wd, transpose_b=True)
        rec_x = slim.bias_add(rec_x, activation_fn=tf.nn.relu)
	#tf.summary.histogram('activations/rec_x', rec_x)
        mse = slim.losses.mean_squared_error(rec_x,
                                             inputs,
                                             weights=gamma)
        tf.summary.scalar('loss/autoencoder_mse_loss', mse, collections=['train'])
	#"""
    total_loss = slim.losses.get_total_loss()

    return total_loss, accuracy






        
        
