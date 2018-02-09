# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

class SequenceProcessing(object):
    def remove_item_from_list(self, the_list, item_to_remove):
        return [item for item in the_list if item != item_to_remove]

    def estimate_actual_sequence_length(self, the_list, dummy_word):
        # Estimate actual length of a sequence padded with dummy word at the end
        return len(self.remove_item_from_list(the_list, dummy_word))

    def estimate_actual_sequences_length(self, the_lists, dummy_word):
        sequence_lengths=[]
        for the_list in the_lists:
            sequence_lengths.append(self.estimate_actual_sequence_length(the_list, dummy_word))
        return sequence_lengths
