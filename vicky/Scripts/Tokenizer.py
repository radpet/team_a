# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

from segtok.segmenter import split_single, split_multi

# <codecell>

class Tokenizer(object):
    
    def sentence_tokenizer(self, string):
        tokenized_sentences = list(split_multi(string))
        return tokenized_sentences

    def word_tokenizer(self, string):
        tokenized_words = string.split()
        return tokenized_words

    def tokenize_sentences_and_words(textdata):
        tokenized_sentences = self.sentence_tokenizer(textdata)
        tokenized_sentences_tokenized_words = [self.word_tokenizer(sentence) for sentence in tokenized_sentences]
        return tokenized_sentences_tokenized_words
