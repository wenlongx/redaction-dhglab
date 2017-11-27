#!/usr/bin/env python

import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

def load_data(split_size):
    """

    :return: y is a dictionary of {confounder_name: [categories, one_hot_representation]}
    """
    #gene_counts = pd.DataFrame.from_csv('RNASeqQC.all_samples.gene.counts.txt', sep='\t')
    gene_counts = pd.DataFrame.from_csv('log_transformed_gene_counts.csv', sep=',')
    gene_counts = gene_counts.set_index(gene_counts.iloc[:,0]).iloc[:,1:]
    X = gene_counts.transpose().as_matrix()

    confounders = ['sample_type', 'library', 'lane']

    meta = pd.DataFrame.from_csv('RNASeqQC.all_samples.meta.txt', sep='\t')[gene_counts.columns].loc[
        confounders].transpose()

    y = {}
    for confounder in confounders:
        c_unique, c_idx = np.unique(meta[confounder], return_inverse=True)
        y[confounder] = [c_unique, np.eye(len(c_unique))[c_idx]]

    X_train, X_test, y_train, y_test = train_test_split(X, y['sample_type'][1], test_size=split_size)
    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    if len(sys.argv) > 1:
        X_train, y_train, X_test, y_test = load_data(float(sys.argv[1]))

        np.savetxt('X_train.txt', X_train)
        np.savetxt('y_train.txt', y_train)
        np.savetxt('X_test.txt', X_test)
        np.savetxt('y_test.txt', y_test)

    else:
        print "Needs the size of the test split"