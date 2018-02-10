# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# <codecell>

text_processing_scripts_path = os.path.abspath('TextProcessingScripts')
sys.path.append(text_processing_scripts_path)

scripts_path = os.path.abspath('Scripts')
sys.path.append(scripts_path)

# <codecell>

from FeatureExtraction.VocabDict import create_vocab_dict
from FeatureExtraction.UnknownWordsProcessing import UnknownWordsProcessing
from FeatureExtraction.Word2VecUtilities import create_word2vector_model, create_embeddings_matrix, save_word2vector_model, load_word2vector_model
from TensorflowInputProcessing.SentenceProcessing import SentenceProcessing
from TensorflowInputProcessing.DocumentProcessing import DocumentProcessing
from TensorflowInputProcessing.MapWordToID  import MapWordToID 

# <codecell>

data_path = os.path.abspath('data')
test_data = pd.read_csv(os.path.join(data_path, 'test.csv'))

# <codecell>

def preprocess_and_group_data(data):
    data = data.drop_duplicates()
    data.index = range(len(data))
    aliased_snippet = []
    for i in range(len(data)):
        aliased_snippet.append(data['snippet'][i].replace(data['company1'][i],'company1').replace(data['company2'][i],'company2'))
    data['snippet'] = aliased_snippet

    data['snippet'] = data['snippet'].str.lower()

    grouped_data = data.groupby(['company1','company2'])['snippet'].apply(list)
    grouped_data = grouped_data.to_frame().reset_index()
    return data, grouped_data

def word_tokenizer(string):
    return string.split()

# <codecell>

test_data, grouped_test_data = preprocess_and_group_data(test_data)

# <codecell>

from CommonUtilities.FileUtilities import return_file_content, save_pickle_file, load_pickle_file

# <codecell>

training_params = load_pickle_file(os.path.join(data_path, 'training_params.p'))

# <codecell>

vocab_dict = training_params['vocab_dict'] 
rev_vocab_dict = training_params['rev_vocab_dict']
estimated_sent_len = training_params['estimated_sent_len']
estimated_doc_len = training_params['estimated_doc_len']
embedding_matrix = training_params['embedding_matrix']

# <codecell>

sentence_processing = SentenceProcessing()
document_processing = DocumentProcessing()
map_word_to_id = MapWordToID(vocab_dict)
unknown_words_processing = UnknownWordsProcessing(vocab_list=vocab_dict.keys(), replace=False)


# <codecell>

def return_X(grouped_snippets):
    tokenized_sentences_tokenized_words = [word_tokenizer(sent) for sent in grouped_snippets]
    tokenized_sentences_tokenized_words = unknown_words_processing.remove_or_replace_unkown_word_from_sentences(tokenized_sentences_tokenized_words)
    preprocessed_sentences_of_document = sentence_processing.pad_truncate_sent(tokenized_sentences_tokenized_words, estimated_sent_len,  dummy_token='my_dummy')
    preprocessed_document = document_processing.pad_truncate_document(preprocessed_sentences_of_document, estimated_doc_len, estimated_sent_len)
    id_array = np.asarray(map_word_to_id.word_lists_to_id_lists(preprocessed_document))
    return id_array

# <codecell>

X_test = []
y_test = []
for index , row in grouped_test_data.iterrows():
    grouped_snippets = row['snippet']
    id_array = return_X(grouped_snippets)
    X_test.append(id_array)
    y_test.append(0)

# <codecell>

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

# <codecell>

tf.reset_default_graph()

# <codecell>

sess = tf.Session()
saver = tf.train.import_meta_graph(os.path.join(data_path,"consent.ckpt.meta"))
saver.restore(sess, os.path.join(data_path,"consent.ckpt"))
graph = tf.get_default_graph()

X = graph.get_operation_by_name('Inputs/X').outputs[0]
y = graph.get_operation_by_name('Inputs/y').outputs[0]
tf_sentences_length = graph.get_operation_by_name('Inputs/sentences_length').outputs[0]
tf_documents_length = graph.get_operation_by_name('Inputs/documents_length').outputs[0]
normalized_sentence_attentions = graph.get_operation_by_name('Attention-2/ExpandDims').outputs[0]
prob = graph.get_operation_by_name('Prediction/prob').outputs[0]


# <codecell>

from CustomNN import create_RNN, create_attention
from CustomRNN import CustomRNN
from LengthEstimation import estimate_sentences_and_document_lengths

# <codecell>

X_valid_samples, y_valid_samples = np.asarray(X_test), np.asarray(y_test).reshape(len(y_test),1)
valid_sentences_length, valid_documents_length = estimate_sentences_and_document_lengths(X_valid_samples, vocab_dict['my_dummy'])

# <codecell>

np_normalized_sentence_attentions, np_prob = sess.run([normalized_sentence_attentions, prob],
                                                            feed_dict={X:X_valid_samples, y:y_valid_samples,
                                                                       tf_sentences_length:valid_sentences_length,
                                                                       tf_documents_length:valid_documents_length})
attention_scores = np.squeeze(np_normalized_sentence_attentions)
#np_y = y_valid_samples
#accuracy = sum((np_prob>0.5)==(np_y>0.5))/len(np_y)

# <codecell>

grouped_test_data['prob'] = np_prob
grouped_test_data = grouped_test_data[np_prob>0.5]

# <codecell>

grouped_test_data.index= range(len(grouped_test_data))

# <codecell>

grouped_test_data
