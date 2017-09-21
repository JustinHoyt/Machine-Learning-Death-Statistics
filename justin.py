# Import libraries
import pandas as pd                 # pandas is a dataframe library
# import matplotlib.pyplot as plt      # matplotlib.pyplot plots data
import numpy as np
#from scipy.misc import logsumexp
from sklearn.utils.extmath import safe_sparse_dot, logsumexp

# %matplotlib inline

# Read in data

df = pd.read_csv("./DeathRecords/new_data.csv",nrows=10000)
#df = pd.read_csv("./DeathRecords/new_data.csv")
del df['Id']

# Split into train and test

from sklearn.model_selection import train_test_split

feature_col_names = df.columns.values

x = df[feature_col_names].values     # predictor feature columns (8 X m)
split_test_size = 0.30

X_train, X_test = train_test_split(x, test_size=split_test_size, random_state=42)

X_train_cv, X_test_cv = train_test_split(X_train, test_size=split_test_size, random_state=42)

# Categorical Bernoulli Naive Bayes

from categorical_nb import CategoricalNB

print(X_train_cv.shape)
print(X_test_cv.shape)

import time

output_file_name = 'ho_avg_ll_k_test'

from sklearn.feature_extraction import DictVectorizer

print("DictVectorizing all data")
t0 = time.clock()
n = x.shape[1]

x_as_list_of_dict = []
for z in x:
     vals = {}
     for i in range(n):
         vals[str(i)] = str(z[i]).strip()
     x_as_list_of_dict.append(vals)

vec = DictVectorizer()
x_vec = vec.fit_transform(x_as_list_of_dict)
t1 = time.clock()
print("Execution time: %s" % (t1-t0))

print("DictVectorizing training cv data")
t0 = time.clock()

X_train_cv_as_list_of_dict = []
for z in X_train_cv:
     vals = {}
     for i in range(n):
         vals[str(i)] = str(z[i]).strip()
     X_train_cv_as_list_of_dict.append(vals)

X_train_cv_vec = vec.transform(X_train_cv_as_list_of_dict)

t1 = time.clock()
print("Execution time: %s" % (t1-t0))

print("DictVectorizing hold-out cv data")
t0 = time.clock()
X_test_cv_as_list_of_dict = []
for z in X_test_cv:
     vals = {}
     for i in range(n):
         vals[str(i)] = str(z[i]).strip()
     X_test_cv_as_list_of_dict.append(vals)

X_test_cv_vec = vec.transform(X_test_cv_as_list_of_dict)
t1 = time.clock()
print("Execution time: %s" % (t1-t0))

ho_avg_ll = []
# k = number of clusters
k = 30
for num_clusters in np.arange(k, k+1):
    print(num_clusters)
    # t = number of samples of same type of problem with same clusters
    for t in np.arange(10):
        print(t)
        nb_model = CategoricalNB(n_classes=num_clusters, output_space=[str(v) for v in range(num_clusters)], max_EM_iter=1)
        nb_model.set_binarized_input_info(vec)
        prev_cv_avg_ll = float('-inf')
        # num iterations of em (unsupervised learning algorithm)
        for iter in np.arange(1,100):
            print(iter)
            print("Fitting model")
            t0 = time.clock()
            nb_model.fit(X_train_cv_vec)
            t1 = time.clock()
            print("Execution time: %s" % (t1-t0))
            print("Cross-validating model")
            t0 = time.clock()
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
#best_row = ho_avg_ll_array[np.argmax(ho_avg_ll_array[:,2]),:]

#print(best_row)

np.save(output_file_name + str(k) + '.npy',ho_avg_ll_array)
np.savetxt(output_file_name + str(k) + '.dat',ho_avg_ll_array)

import pickle

with open('test_nb_model.pkl', 'wb') as file:
    pickle.dump(nb_model, file, pickle.HIGHEST_PROTOCOL)
file.closed

for s, o in zip(['vec', 'x_as_list_of_dict', 'x_vec'],[vec, x_as_list_of_dict, x_vec]):
    with open('test_' + s + '.pkl', 'wb') as file:
        pickle.dump(o, file, pickle.HIGHEST_PROTOCOL)
    file.closed


#pickle.dump(write_data, nb_model)

print(nb_model.class_log_prior_)

with open('test_nb_model.pkl', 'rb') as file:
    nb_model_1 = pickle.load(file)
file.closed

#nb_model_1 = pickle.load('test_nb_model.pkl')

print(nb_model_1.class_log_prior_)

#print(vec)
#print(x_as_list_of_dict)
#print(x_vec)

new_vec = DictVectorizer()
new_x_as_list_of_dict = [{}]
#new_x_vec = DictVectorizer(dtype=<class 'numpy.float64'>, separator='=', sort=True, sparse=True)
new_x_vec = DictVectorizer()

s = 'vec'
with open('test_' + s + '.pkl', 'rb') as file:
    new_vec = pickle.load(file)
file.closed

s = 'x_as_list_of_dict'
with open('test_' + s + '.pkl', 'rb') as file:
    new_x_as_list_of_dict = pickle.load(file)
file.closed


s = 'x_vec'
with open('test_' + s + '.pkl', 'rb') as file:
    new_x_vec = pickle.load(file)
file.closed


# for s, o in zip(['vec', 'x_as_list_of_dict', 'x_vec'],[new_vec, new_x_as_list_of_dict, new_x_vec]):
#     with open('test_' + s + '.pkl', 'rb') as file:
#         pickle.loads(file)
#     file.closed

print(new_vec)
print(new_x_as_list_of_dict)
print(new_x_vec)