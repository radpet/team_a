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
from CommonUtilities.FileUtilities import return_file_content, save_pickle_file, load_pickle_file


# <codecell>

from CustomNN import create_RNN, create_attention
from CustomRNN import CustomRNN
from LengthEstimation import estimate_sentences_and_document_lengths

# <codecell>

data_path = os.path.abspath('data_copy')
train_data = pd.read_csv(os.path.join(data_path, 'train.csv'))
test_data = pd.read_csv(os.path.join(data_path, 'test.csv'))

# <codecell>

# test_data_copy = pd.DataFrame([test_data['company1'],test_data['company2'], test_data['is_parent'],test_data['snippet']]).T
# test_data_copy.columns = ['company2','company1','is_parent','snippet']
# test_data = pd.concat([test_data,test_data_copy])

# <codecell>

def preprocess_and_group_data(data):
    data = data.drop_duplicates()
    data.index = range(len(data))
    aliased_snippet = []
    for i in range(len(data)):
        aliased_snippet.append(data['snippet'][i].replace(data['company2'][i],' company2 ').replace(data['company1'][i],' company1 '))
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
tf_keep_prob = graph.get_operation_by_name('Inputs/tf_keep_prob').outputs[0]
tf_sentences_length = graph.get_operation_by_name('Inputs/sentences_length').outputs[0]
tf_documents_length = graph.get_operation_by_name('Inputs/documents_length').outputs[0]
normalized_sentence_attentions = graph.get_operation_by_name('Attention-2/ExpandDims').outputs[0]
prob = graph.get_operation_by_name('Prediction/prob').outputs[0]


# <codecell>

# validation_data = load_pickle_file(os.path.join(data_path, 'validation_data.p'))
# X_valid = validation_data['X_valid']
# y_valid = validation_data['y_valid']

# <codecell>

X_valid_samples, y_valid_samples = np.asarray(X_test), np.asarray(y_test).reshape(len(y_test),1)
valid_sentences_length, valid_documents_length = estimate_sentences_and_document_lengths(X_valid_samples, vocab_dict['my_dummy'])

# <codecell>

np_normalized_sentence_attentions, np_prob, np_y = sess.run([normalized_sentence_attentions, prob, y],
                                                            feed_dict={X:X_valid_samples, y:y_valid_samples,tf_sentences_length:valid_sentences_length, tf_documents_length:valid_documents_length,tf_keep_prob:1})
attention_scores = np.squeeze(np_normalized_sentence_attentions)

# <codecell>

np_y = y_valid_samples
accuracy = sum((np_prob>0.5)==(np_y>0.5))/len(np_y)

# <codecell>

_, grouped_test_data = preprocess_and_group_data(test_data)
grouped_test_data['prob'] = np_prob
grouped_test_data['imp_sent_num'] = np.argmax(attention_scores,1)

grouped_test_data_reduced = grouped_test_data[np_prob>0.98]
grouped_test_data_reduced.index= range(len(grouped_test_data_reduced))

# <codecell>

top_results = grouped_test_data.sort_values('prob', ascending=False)[:80]
top_results.index = range(len(top_results))

# <codecell>

top_results

# <codecell>

top_results.iloc[15]['company1'], top_results.iloc[15]['company2']

# <codecell>

def return_actual_parent (predicted_subsidiary_name):
    try:
        return (list(train_data[(train_data['company2']==predicted_subsidiary_name)&(train_data['is_parent'])==True]['company1'])[0])
    except:
        return 'No_parent'

# <codecell>

count = 0
identified_pairs = []
top_results_fp_removed = []
for index, row in top_results.iterrows():
    actual_parent_name = return_actual_parent(row['company2'])
    if actual_parent_name=='No_parent' or actual_parent_name==row['company1']:
        if [row['company2'],row['company1']] not in identified_pairs:
            identified_pairs.append([row['company1'], row['company2']])
            top_results_fp_removed.append(row)

# <codecell>

top_results_fp_removed = pd.DataFrame(top_results_fp_removed)
top_results_fp_removed.index = range(len(top_results_fp_removed))

# <codecell>

top_results_fp_removed

# <codecell>

print(top_results_fp_removed[['company1','company2']])

# <codecell>

text_num = 35

# <codecell>

print('Predicted parent:',top_results_fp_removed['company1'][text_num])
print('Predicted subsidiary:',top_results_fp_removed['company2'][text_num])

# <codecell>

print('Important Sentence')
print(top_results_fp_removed['snippet'][text_num][top_results_fp_removed['imp_sent_num'][text_num]])

# <codecell>

print('Input snippets:')
for i, snippet in enumerate(top_results_fp_removed['snippet'][text_num]):
    print(i+1, snippet, end='\n\n')

# <codecell>


