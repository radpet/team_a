# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

'''
This module contains the class SentenceProcessing which consists of functions useful for pre-processing the sentences to desired lengths.

'''

# <codecell>

import sys
sys.path.insert(0,'..')
from CustomExceptions.CustomExceptions import SentenceLongerError, SentenceShorterError, DocumentLongerError, DocumentShorterError

# <codecell>

class SentenceProcessing(object):
        
    def merge_small_sentences(self, tokenized_sentences_tokenized_words, min_len = 3):
        '''
        This function merges small sentences with previous sentences.

        Parameters
        ----------
        tokenized_sentences_tokenized_words : list of lists
            Each inner list consists of tokenized words corresponding to a sentence.
        min_len : int
            The minimum threshold length of a sentence.

        Returns
        -------
        reformatted_tokenized_sentences_tokenized_words : list of lists
            Each inner list will be of minimum length.

        '''
    
        reformatted_tokenized_sentences_tokenized_words = [[]]
        for sent in tokenized_sentences_tokenized_words:
            if len(sent)>min_len:
                reformatted_tokenized_sentences_tokenized_words.append(sent)
            else:
                reformatted_tokenized_sentences_tokenized_words[-1] = reformatted_tokenized_sentences_tokenized_words[-1] + sent

        if reformatted_tokenized_sentences_tokenized_words[0] == []:
            del reformatted_tokenized_sentences_tokenized_words[0]
        return reformatted_tokenized_sentences_tokenized_words
    
    def truncate_sent(self, sent, chosen_sent_len):
        '''
        This function truncates a sentence longer than chosen_sent_len into lists of length ``chosen_sent_len``.
        
        Parameters
        ----------
        sent : list
            A list of words
        chosen_sent_len : int
        
        Returns
        -------
        reformatted_sent : list of lists
            Each inner list is the truncated input list ``sent``.        
        
        '''
         
        if len(sent) >= chosen_sent_len:
            if len(sent)%chosen_sent_len == 0:
                reformatted_sent = [sent[x*chosen_sent_len:(x*chosen_sent_len)+chosen_sent_len] for x in range(int(len(sent)/chosen_sent_len))]
            else:
                reformatted_sent = [sent[x*chosen_sent_len:(x*chosen_sent_len)+chosen_sent_len] for x in range(int(len(sent)/chosen_sent_len) + 1 )]
            return reformatted_sent
        else:
            raise SentenceShorterError
    
    def pad_sent(self, sent, chosen_sent_len, dummy_token = 'my_dummy'):
        '''
        This function pads a dummy token - my_dummy to sentences shoerter than chosen_sent_len.
        
        Parameters
        ----------
        sent : list
            A list of words
        chosen_sent_len : int
        
        Returns
        -------
        reformatted_sent : list
            The input list is padded with dummy token.
        
        '''
        
        if len(sent) <= chosen_sent_len:
            reformatted_sent = sent + [dummy_token] * (chosen_sent_len - len(sent))
            return reformatted_sent
        else:
            raise SentenceLongerError
    
    def pad_truncate_sent(self, tokenized_sentences_tokenized_words, chosen_sent_len, dummy_token='my_dummy'):
        '''
        This function pads/truncates a list of tokenized sentences.
        First, if the length of the sentence is longer than the chosen_sent_len, then its truncated into sublists.
        Else, if the length of the sentence is shorter than the chosen_sent_len, then its padded with dummy tokens.
        Else, if the length of the sentence is equal to the chosen_sent_len, then nothing is done.
        
        Parameters
        ----------
        tokenized_sentences_tokenized_words : List of list
            Each inner list consists of tokenized words corresponding to a sentence.
        
        chosen_sent_len : int
            
        Returns
        -------
        padded_truncated_tokenized_sentences_tokenized_words : List of lists 
        
        Notes
        -----
        truncate_sent function returns a list of lists as output whereas pad_sent function returns only a list asd output.
        
        '''
        padded_truncated_tokenized_sentences_tokenized_words = []
        for sent in tokenized_sentences_tokenized_words:
            if len(sent) > chosen_sent_len:
                for truncated_sentences in self.truncate_sent(sent, chosen_sent_len):
                    padded_truncated_tokenized_sentences_tokenized_words.append(self.pad_sent(truncated_sentences, chosen_sent_len, dummy_token))
            if len(sent) < chosen_sent_len:
                    padded_truncated_tokenized_sentences_tokenized_words.append(self.pad_sent(sent, chosen_sent_len, dummy_token))
            if len(sent) == chosen_sent_len:
                padded_truncated_tokenized_sentences_tokenized_words.append(sent)
        return padded_truncated_tokenized_sentences_tokenized_words
    

# <codecell>


