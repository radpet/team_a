# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

import os
import sys
import pandas as pd
import numpy as np

# <codecell>

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

# <codecell>

text_processing_scripts_path = os.path.abspath('TextProcessingScripts')
sys.path.append(text_processing_scripts_path)

scripts_path = os.path.abspath('Scripts')
sys.path.append(scripts_path)

# <codecell>

from CustomNN import create_RNN, create_attention
from CustomRNN import CustomRNN
from LengthEstimation import estimate_sentences_and_document_lengths

# <codecell>

from CommonUtilities.FileUtilities import return_file_content, save_pickle_file, load_pickle_file
from CreateTrainingBatches import CreateTrainingBatches

# <codecell>

data_path = os.path.abspath('data')

# <codecell>

training_data = load_pickle_file(os.path.join(data_path, 'training_data.p'))
validation_data = load_pickle_file(os.path.join(data_path, 'validation_data.p'))
training_params = load_pickle_file(os.path.join(data_path, 'training_params.p'))

# <codecell>

embedding_matrix = training_params['embedding_matrix']
rev_vocab_dict = training_params['rev_vocab_dict']

estimated_doc_len = training_params['estimated_doc_len']
estimated_sent_len = training_params['estimated_sent_len']

vocab_dict = training_params['vocab_dict']

# <codecell>

create_training_batches = CreateTrainingBatches(training_data['X_train'], training_data['y_train'],
                                                validation_data['X_valid'], validation_data['y_valid'])

# <codecell>

n_neurons_GRU_1 = 50
n_neurons_GRU_2 = 100
attention_n_neurons_1 = 100
attention_n_neurons_2 = 100
learning_rate = 0.01

tf.reset_default_graph()
with tf.device('/cpu:0'):
    with tf.name_scope('Inputs'):
        X = tf.placeholder(tf.int32, [None, estimated_doc_len, estimated_sent_len], name='X')
        y = tf.placeholder(tf.float32, [None, 1], name='y')
        tf_sentences_length = tf.placeholder(tf.int32, [None], name = 'sentences_length')
        tf_documents_length = tf.placeholder(tf.int32, [None], name = 'documents_length')
        tf_keep_prob = tf.placeholder(tf.float32, name='tf_keep_prob')

        tf_embedding_matrix = tf.Variable(initial_value=embedding_matrix,
                                          trainable=False, dtype=tf.float32, name='tf_embedding_matrix')
        
        X_embeddings = tf.nn.embedding_lookup(tf_embedding_matrix, X, name='X_embeddings')
        X_embeddings_reshaped = tf.reshape(X_embeddings, shape=(-1, estimated_sent_len, X_embeddings.get_shape().as_list()[-1]))

    with tf.variable_scope('Bi-RNN-1', initializer=tf.contrib.layers.xavier_initializer()):
        conc_outputs_1 = create_RNN(tf.contrib.rnn.GRUCell, n_neurons = n_neurons_GRU_1,
                                  rnn_input = X_embeddings_reshaped, seq_length = tf_sentences_length)
    with tf.variable_scope('Attention-1'):
        sentence_vectors, _ = create_attention(conc_outputs_1, attention_n_neurons_1)
        sentence_vectors_dropped = tf.nn.dropout(sentence_vectors, keep_prob=tf_keep_prob)
        sentence_vectors_reshaped = tf.reshape(sentence_vectors_dropped, shape=(-1, estimated_doc_len, sentence_vectors_dropped.get_shape().as_list()[-1]))
    
        
    with tf.variable_scope('Attention-2'):
        doc_vectors, normalized_sentence_attentions  = create_attention(sentence_vectors_reshaped, attention_n_neurons_2)
    
        
    with tf.name_scope('Prediction'):
        logits = fully_connected(doc_vectors, 1, activation_fn=None)
        prob = tf.nn.sigmoid(logits, name='prob')
        x_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits, name='x_entropy')
        loss = tf.reduce_mean(x_entropy, name ='loss')
      
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)
 

# <codecell>

init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.InteractiveSession()
init.run()


# <codecell>

def write_highest_validation_accuracy(validation_accuracy):
    with open(os.path.join(data_path,'highest_validation_accuracy.txt'),'w') as f:
        f.write(str(validation_accuracy[0]))


# <codecell>

highest_validation_accuracy = float(return_file_content(os.path.join(data_path,'highest_validation_accuracy.txt')))

X_valid_samples, y_valid_samples = create_training_batches.create_validation_data(num_pos=65, num_neg=65)
valid_sentences_length, valid_documents_length = estimate_sentences_and_document_lengths(X_valid_samples, vocab_dict['my_dummy'])


for i in range(1000):
    X_train_samples, y_train_samples = create_training_batches.create_training_data(num_pos=25, num_neg=40)
    sentences_length, documents_length = estimate_sentences_and_document_lengths(X_train_samples, vocab_dict['my_dummy'])
    _, np_prob, np_y = sess.run([training_op, prob, y], feed_dict={X:X_train_samples, y:y_train_samples,
                                                                   tf_sentences_length:sentences_length,
                                                                   tf_documents_length:documents_length,
                                                                   tf_keep_prob:0.9})

    if i%50 == 1:
        np_prob, np_y = sess.run([prob, y],feed_dict={X:X_valid_samples, y:y_valid_samples,
                                                      tf_sentences_length:valid_sentences_length,
                                                      tf_documents_length:valid_documents_length,
                                                      tf_keep_prob:1})

        validation_accuracy = sum((np_prob>0.5)==(np_y>0.5))/len(np_y)
        print('Validation Accuracy', i, validation_accuracy)
        
        if validation_accuracy > highest_validation_accuracy:
            write_highest_validation_accuracy(validation_accuracy)
            save_path = saver.save(sess, os.path.join(data_path,"consent.ckpt"))
            print('Saved Highest accurate model')
            highest_validation_accuracy = validation_accuracy

# <codecell>

np_normalized_sentence_attentions, np_prob, np_y = sess.run([normalized_sentence_attentions, prob, y],feed_dict={X:X_valid_samples, y:y_valid_samples,tf_sentences_length:valid_sentences_length, tf_documents_length:valid_documents_length,tf_keep_prob:1})
attention_scores = np.squeeze(np_normalized_sentence_attentions)
accuracy = sum((np_prob>0.5)==(np_y>0.5))/len(np_y)
print('Validation Accuracy', i, accuracy)

# <codecell>

text_num =  20
np_prob[text_num], np_y[text_num]

# <codecell>

for sent in X_valid_samples[text_num]:
    for word_id in sent:
        word = rev_vocab_dict[word_id]
        if word!='my_dummy':
            print(word, end=' ')

# <codecell>

important_sentence_num = np.argmax(attention_scores,1)

text_num =  20
for word_id in X_valid_samples[text_num][important_sentence_num[text_num]]:
    word = rev_vocab_dict[word_id]
    if word!='my_dummy':
        print(word, end=' ')


# <codecell>

now = 'consent'
root_logdir = "tf_logs"
logdir = os.path.join(data_path,"{}/run-{}".format(root_logdir, now))
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
file_writer.close()

# <codecell>


