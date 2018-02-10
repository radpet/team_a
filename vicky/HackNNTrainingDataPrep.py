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

data_path = os.path.abspath('data')
data = pd.read_csv(os.path.join(data_path, 'train.csv'))

# <codecell>

def preprocess_and_group_data(data):
    data = data.drop_duplicates()
    data.index = range(len(data))
    aliased_snippet = []
    for i in range(len(data)):
        aliased_snippet.append(data['snippet'][i].replace(data['company1'][i],'company1').replace(data['company2'][i],'company2'))
    data['snippet'] = aliased_snippet

    data['snippet'] = data['snippet'].str.lower()

    grouped_data = data.groupby(['company1','company2','is_parent'])['snippet'].apply(list)
    grouped_data = grouped_data.to_frame().reset_index()
    return data, grouped_data

def word_tokenizer(string):
    return string.split()

# <codecell>

data, grouped_data = preprocess_and_group_data(data)

# <codecell>

all_documents_tokenized_words = [list(set(word_tokenizer(snippet))) for snippet in data['snippet']]
all_documents_tokenized_sentences_tokenized_words = [word_tokenizer(snippet) for snippet in data['snippet']]

# <codecell>

sent_lens = [len(sent) for sent in all_documents_tokenized_sentences_tokenized_words]
sent_lens = sorted(sent_lens)
estimated_sent_len = sent_lens[int(len(sent_lens)*0.90)]

doc_lens = [len(snippet) for snippet in grouped_data['snippet']]
doc_lens = sorted(doc_lens)
estimated_doc_len = doc_lens[int(len(doc_lens)*0.90)]

# <codecell>

vocab_dict, rev_vocab_dict = create_vocab_dict(all_documents_tokenized_words, min_doc_count=50)

# <codecell>

# parent_subsidy_df = data[data['is_parent']][['company1','company2']].drop_duplicates()
# subsidy_parent = zip(list(parent_subsidy_df['company2']), list(parent_subsidy_df['company1']))
# subsidy_parent = [list(l) for l in subsidy_parent]   

# all_documents_tokenized_sentences_tokenized_words_2 = [word_tokenizer(snippet) for snippet, company1, company2 in zip(data['snippet'],data['company1'],data['company2']) if [company1, company2] not in subsidy_parent]

# <codecell>

unknown_words_processing = UnknownWordsProcessing(vocab_list=vocab_dict.keys(), replace=False)
w2v_training_sentences = unknown_words_processing.remove_or_replace_unkown_word_from_sentences(all_documents_tokenized_sentences_tokenized_words)
w2v_model = create_word2vector_model(w2v_training_sentences, wv_size=50)

# <codecell>

embedding_matrix = create_embeddings_matrix(w2v_model, rev_vocab_dict)
embedding_matrix = np.vstack((embedding_matrix, np.zeros((1, embedding_matrix.shape[1]))))

vocab_dict['my_dummy']=len(vocab_dict)
rev_vocab_dict[len(rev_vocab_dict)]='my_dummy'

# <codecell>

sentence_processing = SentenceProcessing()
document_processing = DocumentProcessing()
map_word_to_id = MapWordToID(vocab_dict)

# <codecell>

def return_X(grouped_snippets):
    tokenized_sentences_tokenized_words = [word_tokenizer(sent) for sent in grouped_snippets]
    tokenized_sentences_tokenized_words = unknown_words_processing.remove_or_replace_unkown_word_from_sentences(tokenized_sentences_tokenized_words)
    preprocessed_sentences_of_document = sentence_processing.pad_truncate_sent(tokenized_sentences_tokenized_words, estimated_sent_len,  dummy_token='my_dummy')
    preprocessed_document = document_processing.pad_truncate_document(preprocessed_sentences_of_document, estimated_doc_len, estimated_sent_len)
    id_array = np.asarray(map_word_to_id.word_lists_to_id_lists(preprocessed_document))
    return id_array

# <codecell>

grouped_data = grouped_data.sample(frac=1)
grouped_data.index = range(len(grouped_data))

# <codecell>

grouped_train_data = grouped_data[:750]
grouped_train_data.index = range(len(grouped_train_data))
grouped_validation_data = grouped_data[750:]
grouped_validation_data.index = range(len(grouped_validation_data))

# <codecell>

# grouped_train_data['is_parent'].value_counts(), grouped_validation_data['is_parent'].value_counts()

# <codecell>

X_train = []
y_train = []
for index , row in grouped_train_data.iterrows():
    grouped_snippets = row['snippet']
    id_array = return_X(grouped_snippets)
    X_train.append(id_array)
    y_train.append(row['is_parent'])

# <codecell>

X_valid = []
y_valid = []
for index , row in grouped_validation_data.iterrows():
    grouped_snippets = row['snippet']
    id_array = return_X(grouped_snippets)
    X_valid.append(id_array)
    y_valid.append(row['is_parent'])

# <codecell>

training_data = {}
training_params = {}

training_data['X_train'] = X_train
training_data['y_train'] = y_train

training_params['vocab_dict'] = vocab_dict
training_params['rev_vocab_dict'] = rev_vocab_dict
training_params['estimated_sent_len'] = estimated_sent_len
training_params['estimated_doc_len'] = estimated_doc_len
training_params['embedding_matrix'] = embedding_matrix

# <codecell>

validation_data = {}

validation_data['X_valid'] = X_valid
validation_data['y_valid'] = y_valid

# <codecell>



# <codecell>

save_pickle_file(training_data, os.path.join(data_path, 'training_data.p'))
save_pickle_file(training_params, os.path.join(data_path, 'training_params.p'))
save_pickle_file(validation_data, os.path.join(data_path, 'validation_data.p'))


# <codecell>


