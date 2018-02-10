# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

from TensorflowInputProcessing.SequenceProcessing import SequenceProcessing
import numpy as np

# <codecell>

def estimate_sentences_and_document_lengths(X_samples, dummy_word_id):
    sequence_processing = SequenceProcessing()

    sentences_length = []
    for X_sample in X_samples:
        sentences_length += sequence_processing.estimate_actual_sequences_length(X_sample, dummy_word_id)

    documents_length = np.asarray([np.sum(np.asarray([len(set(sent)) for sent in X_sample])>1) for X_sample in X_samples])
    return sentences_length, documents_length
