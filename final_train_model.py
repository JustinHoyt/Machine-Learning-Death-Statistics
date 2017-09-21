# Import libraries
# import pandas as pd                 # pandas is a dataframe library
# import matplotlib.pyplot as plt      # matplotlib.pyplot plots data
import numpy as np
#from scipy.misc import logsumexp
from sklearn.utils.extmath import safe_sparse_dot, logsumexp

#pickle.dump(write_data, nb_model)
# print(nb_model.class_log_prior_)

# from sklearn.feature_extraction import DictVectorizer

import pickle

from categorical_nb import CategoricalNB

import time

output_file_name = 'ho_avg_ll_k_final_test'

# vectorizer used
s = 'vec'
with open('test_' + s + '.pkl', 'rb') as file:
    vec = pickle.load(file)
file.closed

# Dict representation of all the data
s = 'x_as_list_of_dict'
with open('test_' + s + '.pkl', 'rb') as file:
    x_as_list_of_dict = pickle.load(file)
file.closed

# sparse matrix representation of all the data absed on vec
s = 'x_vec'
with open('test_' + s + '.pkl', 'rb') as file:
    x_vec = pickle.load(file)
file.closed

# Split into train and test
from sklearn.model_selection import train_test_split

split_test_size = 0.30

X_train_vec, _ = train_test_split(x_vec, test_size=split_test_size, random_state=42)

X_train_cv_vec, X_test_cv_vec = train_test_split(X_train_vec, test_size=split_test_size)

ho_avg_ll = []
num_clusters = 30
num_trials = 10
num_iter = 100
print(num_clusters)
# t = number of samples of same type of problem with same clusters
for t in np.arange(num_trials):
    print(t)
    nb_model = CategoricalNB(n_classes=num_clusters, output_space=[str(v) for v in range(num_clusters)], max_EM_iter=1)
    nb_model.set_binarized_input_info(vec)
    prev_cv_avg_ll = float('-inf')
    # num iterations of em (unsupervised learning algorithm)
    for iter in np.arange(1,num_iter):
        print(iter)
        t0 = time.clock()
        print("Fitting model")
        nb_model.fit(X_train_cv_vec)
        t1 = time.clock()
        print("Execution time: %s" % (t1-t0))
        t0 = time.clock()
        print("Cross-validating model")
        cv_jll = nb_model._joint_log_likelihood(X_test_cv_vec)
        cv_avg_ll = np.mean(logsumexp(cv_jll,axis=1))
        t1 = time.clock()
        print("Execution time: %s" % (t1-t0))
        result = [num_clusters, t, iter, np.float64(cv_avg_ll)]
        ho_avg_ll.append(result)
        print(result)
        ho_avg_ll_array = np.array(ho_avg_ll)
        np.save(output_file_name + '.npy',ho_avg_ll_array)
        np.savetxt(output_file_name + '.dat',ho_avg_ll_array)
        if cv_avg_ll > prev_cv_avg_ll:
            prev_cv_avg_ll = cv_avg_ll
        else:
            break

ho_avg_ll_array = np.array(ho_avg_ll)
best_row = ho_avg_ll_array[np.argmax(ho_avg_ll_array[:,2]),:]

print(best_row)

np.save(output_file_name + str(num_clusters) + '.npy',ho_avg_ll_array)
np.savetxt(output_file_name + str(num_clusters) + '.dat',ho_avg_ll_array)


with open('final_nb_model.pkl', 'wb') as file:
    pickle.dump(nb_model, file, pickle.HIGHEST_PROTOCOL)
file.closed



