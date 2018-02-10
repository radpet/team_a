# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

from Tokenizer import Tokenizer
from TensorflowInputProcessing.SentenceProcessing import SentenceProcessing
from TensorflowInputProcessing.DocumentProcessing import DocumentProcessing
from FeatureExtraction.UnknownWordsProcessing import UnknownWordsProcessing


# <codecell>

class PreprocessSentencesAndDocument(object):
    def __init__(self, vocab_dict, estimated_sent_len, estimated_doc_len):
        self.vocab_dict = vocab_dict
        self.estimated_sent_len = estimated_sent_len
        self.estimated_doc_len = estimated_doc_len
        self.tokenizer = Tokenizer()

        self.sentence_processing = SentenceProcessing()
        self.document_processing = DocumentProcessing()
        self.unknown_words_processing = UnknownWordsProcessing(vocab_list=vocab_dict.keys(), replace=False)

    def preprocess_sentences_and_document(self, textdata):
        tokenized_sentences = self.tokenizer.sentence_tokenizer(textdata)
        tokenized_sentences_tokenized_words = [self.tokenizer.word_tokenizer(sent) for sent in tokenized_sentences]
        tokenized_sentences_tokenized_words = self.unknown_words_processing.remove_or_replace_unkown_word_from_sentences(tokenized_sentences_tokenized_words)
        small_sentences_merged_tokenized_sentences_tokenized_words = self.sentence_processing.merge_small_sentences(tokenized_sentences_tokenized_words, min_len=4)
        preprocessed_sentences_of_document = self.sentence_processing.pad_truncate_sent(small_sentences_merged_tokenized_sentences_tokenized_words, self.estimated_sent_len,  dummy_token='my_dummy')
        preprocessed_document = self.document_processing.pad_truncate_document(preprocessed_sentences_of_document, self.estimated_doc_len, self.estimated_sent_len)
        return preprocessed_document

