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
train_data = pd.read_csv(os.path.join(data_path, 'train.csv'))

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

train_data, grouped_train_data = preprocess_and_group_data(train_data)

# <codecell>

sentence_processing = SentenceProcessing()
document_processing = DocumentProcessing()
map_word_to_id = MapWordToID(vocab_dict)

# <codecell>

i = 30
tokenized_sentences_tokenized_words = [word_tokenizer(sent) for sent in grouped_train_data['snippet'][i]]
tokenized_sentences_tokenized_words = unknown_words_processing.remove_or_replace_unkown_word_from_sentences(tokenized_sentences_tokenized_words)
preprocessed_sentences_of_document = sentence_processing.pad_truncate_sent(tokenized_sentences_tokenized_words, estimated_sent_len,  dummy_token='my_dummy')
preprocessed_document = document_processing.pad_truncate_document(preprocessed_sentences_of_document, estimated_doc_len, estimated_sent_len)
id_array = np.asarray(map_word_to_id.word_lists_to_id_lists(preprocessed_document))

