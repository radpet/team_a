# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

import numpy as np

# <codecell>

from CommonUtilities.NumpyArrayUtilities import sample_from_array

# <codecell>

class CreateTrainingBatches(object):
    def __init__(self, X_train, y_train, X_valid, y_valid):
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        
    def create_data(self, X, y, num_pos, num_neg):
        X = np.asarray(X)
        y = np.asarray(y)
        pos_X = X[y==1]
        neg_X = X[y==0]

        pos_X_samples = sample_from_array(pos_X, num_pos, num_rows=True)
        pos_y_samples = np.asarray([1]*num_pos)

        neg_X_samples = sample_from_array(neg_X, num_neg, num_rows=True)
        neg_y_samples = np.asarray([0]*num_neg)

        X_samples = np.vstack((pos_X_samples, neg_X_samples))
        y_samples = np.hstack((pos_y_samples, neg_y_samples)).T.reshape(-1,1)
        return X_samples, y_samples

    def create_training_data(self, num_pos, num_neg):
        X_train_samples, y_train_samples = self.create_data(self.X_train, self.y_train, num_pos, num_neg)
        return X_train_samples, y_train_samples

    def create_validation_data(self, num_pos, num_neg):
        X_valid_samples, y_valid_samples = self.create_data(self.X_valid, self.y_valid, num_pos, num_neg)
        return X_valid_samples, y_valid_samples
