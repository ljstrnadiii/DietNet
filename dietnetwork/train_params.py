# This script trains the dietnet

""" The idea is to use the basic feed_dict procedure with numpy arrays to 
    perform batch learning. 
    Possible Mods:
        - none yet
"""

import time
import os
import tensorflow as tf
slim = tf.contrib.slim
import numpy as np
from network import dietnet
from load import load_data
import random
import argparse
import itertools

def grid():
    """
    This function create the hyper parameter grid in order to train many models
    stds: .01, .1
    activations: relu, tanh, None (linear)
    optimizers: RMS prop and Adam
    learning rates: .01, .001
    gives 48 different models
    Args:
        -None
    Returns:
        -list of dictionarys defining each model.abs
    """
    
    param_grid = {'w_init_dist':[
                      tf.random_uniform_initializer(minval=-.017,maxval=0.17),
                      tf.random_uniform_initializer(minval=-.17,maxval=.17),
                      tf.truncated_normal_initializer(.01),
                      tf.truncated_normal_initializer(.1)],
                  'act_funs':[
                      tf.nn.relu,
                      tf.nn.tanh,
                      None],
                  'optims':[
                      tf.train.AdamOptimizer(.01),
                      tf.train.AdamOptimizer(.001),
                      tf.train.RMSPropOptimizer(.01),
                      tf.train.RMSPropOptimizer(.001)] 
                }

    def dict_product(param):
        return (dict(itertools.izip(param, x)) for x in itertools.product(*param.itervalues()))           
    return list(dict_product(param_grid))



def train(args):
    """
    This function trains the dietnetwork using the histogram embedding. The idea is to
    use batch learning from numpy files using basic dict_feed (not much improvement in time)
    Args:
        - args.path: path to the data dir which contains train/val/test
        - args.learning_rate: learning rate for the optimizer
        - args.sum_dir: summry
        - args.num_epoch: ...
        - args.batchsize: ...
    """
 


    # load the data: (note:already preshuffled)
    trainX, trainY, validX, validY, testX, testY = load_data(args.path)
    trainX = np.array(trainX).astype(np.float32)
    trainY = np.array(trainY).astype(np.float32)
    validX = np.array(validX).astype(np.float32)
    validY = np.array(validY).astype(np.float32)
    testX = np.array(testX).astype(np.float32)
    testY = np.array(testY).astype(np.float32)
 
    val_len = np.shape(validX)[0]
    test_len = np.shape(testX)[0]

    # get dietnet input values:
    input_dim=np.shape(trainX)[1]
    output_dim=np.shape(trainY)[1]
    embed_size=input_dim

    # build the graph:
    loss, accuracy = dietnet(path=args.path,
                             input_size=input_dim, 
                             output_size=output_dim,
                             dropout_rate=args.dropout_rate,
			     embed_size=embed_size,
                             hidden_size=100,
                             std=args.std,
                             gamma=args.gamma)

    #final ops: accuracy, loss, optimizer:
    #optimizer = tf.train.RMSPropOptimizer(args.learning_rate)
    optimizer = tf.train.AdamOptimizer(args.learning_rate)
    training_op = slim.learning.create_train_op(loss, optimizer,
                                                #summarize_gradients=True,
                                                clip_gradient_norm=10)
    
    # Summary stuff: get the train/valid/test loss and accuracy
    test_acc_summary = tf.summary.scalar('test_accuracy', accuracy, collections=['test'])
    valid_acc_summary = tf.summary.scalar('valid_accuracy', accuracy, collections=['valid'])
    train_acc_summary = tf.summary.scalar('train_accuracy', accuracy, collections=['train'])

    test_loss_summary = tf.summary.scalar('test_loss', loss, collections=['test'])
    valid_loss_summary = tf.summary.scalar('valid_loss', loss, collections=['valid'])
    train_loss_summary = tf.summary.scalar('train_loss', loss, collections=['train'])

    # separates the summaries according to the collection
    train_ops = tf.summary.merge_all('train')
    valid_ops = tf.summary.merge_all('valid')
    test_ops = tf.summary.merge_all('test')

    with tf.Session() as sess:
        # init variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

	# print out all trainable variables
	print([i for i in  tf.trainable_variables()])

        # saver for summary
        swriter = tf.summary.FileWriter(args.sum_dir, sess.graph)

        step = 0

        try:
            for i in range(args.num_epoch):
                for idx in range(int(np.shape(trainX)[0] / args.batchsize)):
                    # prep data for train:
                    a,b = idx*args.batchsize, (idx+1)*args.batchsize
                    batch_x = trainX[a:b,:]
                    batch_y = trainY[a:b,:]

                    #get time
                    start_time=time.time()

                    # run train op and get train loss
                    trainloss, accur, summaries = sess.run([training_op, accuracy, train_ops],
                                                    feed_dict={
                                                        'inputs:0': batch_x,
                                                        'outputs:0': batch_y,
                                                        'is_training:0': True})

                    # add sumamries every other step for memory
                    if not idx % 2: swriter.add_summary(summaries,step)
                    
                    duration=time.time() - start_time

                    # every 5 steps get train and test loss/accur
                    if not idx % 5: 
                        # sample random 25% from test/valid for error
                        val_ind = [i for i in random.sample(xrange(val_len), args.batchsize)]
                        test_ind = [i for i in random.sample(xrange(test_len), args.batchsize)]
                        val_x = validX[val_ind,:]
                        val_y = validY[val_ind,:]
                        
                        test_x = testX[test_ind,:]
                        test_y = testY[test_ind,:]
                        
                        # get val loss/accur:
                        val_loss, accur_valid, summaries = sess.run([loss, accuracy, valid_ops],
                                                       feed_dict={
                                                           'inputs:0': val_x,
                                                           'outputs:0': val_y,
                                                           'is_training:0': False})
                        swriter.add_summary(summaries,step)

                        # get test loss/accur
                        test_loss,accur_test, summaries = sess.run([loss, accuracy,test_ops],
                                                        feed_dict={
                                                            'inputs:0': test_x,
                                                            'outputs:0': test_y,
                                                            'is_training:0': False})
                        swriter.add_summary(summaries, step)
                        
                        # print to console in order to watch:
                        print('step {:d}-train/v/test acc:={:.3f},{:.3f},{:.3f}'.format(step, 
										accur,
										accur_valid,
										accur_test))

                    step += 1

                    # add checkpoint here:...
            
            # if num_epochs is complete close swriter
            swriter.close()

        finally:
            swriter.close()


def parse_arguments():
    """ 
    parser for the inputs of the network i.e. all the args inputs in dietnet
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--path',type=str, help="path to the data dir",
                        default="/usr/local/diet_code/genomic")
    parser.add_argument('--learning_rate', type=float, help="learning rate for optimizer",
                        default=.0001)
    parser.add_argument('--sum_dir',type=str, help="dir to the summary path",
                        default="/usr/local/diet_code/log")
    parser.add_argument('--num_epoch', type=int, help="number of epochs",
                        default=800)
    parser.add_argument('--batchsize', type=int, help="batch size for training",
                        default=128)
    parser.add_argument('--std', type=float, help="standard deviation for the weight init",
                        default=.05)
    parser.add_argument('--gamma', type=float, help="gamma for the loss",
                        default=1)
    parser.add_argument('--dropout_rate', type=float, help="prob for batchnorm",
			default=.5)
    return parser.parse_args()


if __name__=='__main__':
    train(parse_arguments())


