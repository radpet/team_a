# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

import numpy as np

# <codecell>

import sys
sys.path.insert(0,'..')
from CustomExceptions.CustomExceptions import SentenceLongerError, SentenceShorterError, DocumentLongerError, DocumentShorterError

# <codecell>

class DocumentProcessing(object):
    
    def pad_doc(self, preprocessed_sentences_of_document, chosen_doc_len, chosen_sent_len):
        if len(preprocessed_sentences_of_document) <= chosen_doc_len:
            return np.vstack((preprocessed_sentences_of_document, [["my_dummy"]*chosen_sent_len]*(chosen_doc_len-len(preprocessed_sentences_of_document))))
        else:
            raise DocumentLongerError
            
    def truncate_doc(self, preprocessed_sentences_of_document, chosen_doc_len):
        if len(preprocessed_sentences_of_document) >= chosen_doc_len:
            return preprocessed_sentences_of_document[:chosen_doc_len]
        else:
            raise DocumentShorterError
            
    def pad_truncate_document(self, preprocessed_sentences_of_document, chosen_doc_len, chosen_sent_len):
        if len(preprocessed_sentences_of_document) < chosen_doc_len:
            preprocessed_document = self.pad_doc(preprocessed_sentences_of_document, chosen_doc_len, chosen_sent_len)
        elif len(preprocessed_sentences_of_document) > chosen_doc_len:
            preprocessed_document = self.truncate_doc(preprocessed_sentences_of_document, chosen_doc_len)
        else:
            preprocessed_document = preprocessed_sentences_of_document
        return preprocessed_document
    
    

# <codecell>


