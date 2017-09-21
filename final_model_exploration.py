
import numpy as np
from categorical_nb import CategoricalNB

from numpy.matlib import repmat

import pickle

from categorical_nb import CategoricalNB

from sklearn.feature_extraction import DictVectorizer


output_file_name = 'sorted_class_log_prior'

vec = DictVectorizer()

# vectorizer used
s = 'vec'
with open('test_' + s + '.pkl', 'rb') as file:
    vec = pickle.load(file)
file.closed


# # Dict representation of all the data
# s = 'x_as_list_of_dict'
# with open('test_' + s + '.pkl', 'rb') as file:
#     x_as_list_of_dict = pickle.load(file)
# file.closed

# sparse matrix representation of all the data absed on vec
s = 'x_vec'
with open('test_' + s + '.pkl', 'rb') as file:
    x_vec = pickle.load(file)
file.closed

with open('final_nb_model.pkl', 'rb') as file:
    nb_model_final = pickle.load(file)
file.closed

ordered_class_log_prior_values = np.sort(nb_model_final.class_log_prior_)[::-1]
ordered_class_log_prior_indexes = np.argsort(nb_model_final.class_log_prior_)[::-1]

np.savetxt(output_file_name + '_values.dat', ordered_class_log_prior_values)
np.savetxt(output_file_name + '_indexes.dat', ordered_class_log_prior_indexes)

high_class_log_prior_indexes = ordered_class_log_prior_indexes[:3]
low_class_log_prior_indexes = ordered_class_log_prior_indexes[-3:]


#print(vec.vocabulary_)

feature_names = ["Sex",
"ResidentStatus",
#"Education1989Revision",
"Education2003Revision",
"AgeRecode12",
"PlaceOfDeathAndDecedentsStatus",
"MaritalStatus",
"InjuryAtWork",
"MannerOfDeath",
"MethodOfDisposition",
"Autopsy",
"ActivityCode",
"PlaceOfInjury",
"CauseRecode39",
"Race",
"BridgedRaceFlag",
"RaceImputationFlag"] #,
#"RaceRecode3",
#"RaceRecode5"]



def most_likely_feature_for_clusters():
    global idx, i, feat_idx, feat_names, j, feat_log_prob, max_val, most_likely_feat_assign_idx
    for idx in indexes:
        print(" ")
        print(" ")
        print("******* CLuster %d *****" % (idx))
        print(nb_model_final.class_log_prior_[idx])
        max_val = []
        for i in range(nb_model_final.n_features):
            feat_idx = nb_model_final.feature_indexes_[i]
            #print(feature_names[i])
            # feat_names = []
            # for j in feat_idx:
            #     feat_names.append(vec.feature_names_[j])
            feat_log_prob = []
            for j in feat_idx:
                feat_log_prob.append(nb_model_final.feature_log_prob_[idx][j])
            #print(feat_names)
            #print(np.exp(feat_log_prob))
            max_val.append(np.max(feat_log_prob))
            # most_likely_feat_assign_idx = []
            # for j in range(len(feat_log_prob)):
            #     if feat_log_prob[j] == max_val:
            #         most_likely_feat_assign_idx.append(j)
            # for j in most_likely_feat_assign_idx:
            #     print(nb_model_final.feature_space[i][j])

        sorted_feat_idx = np.argsort(max_val)[::-1]
        for i in sorted_feat_idx:
            feat_idx = nb_model_final.feature_indexes_[i]
            print("%d. %s" % (i,feature_names[i]))
            feat_names = []
            for j in feat_idx:
                feat_names.append(vec.feature_names_[j])
            feat_log_prob = []
            for j in feat_idx:
                feat_log_prob.append(nb_model_final.feature_log_prob_[idx][j])
            print(feat_names)
            #print(np.exp(feat_log_prob))
            print("Prob(best feature value | class): %f" % (np.exp(max_val[i])))
            most_likely_feat_assign_idx = []
            for j in range(len(feat_log_prob)):
                if feat_log_prob[j] == max_val[i]:
                    most_likely_feat_assign_idx.append(j)
            for j in most_likely_feat_assign_idx:
                print("Feature value: %s" % (nb_model_final.feature_space[i][j]))

print("Most likely features for clusters with highest prior probability")
indexes = high_class_log_prior_indexes
most_likely_feature_for_clusters()

print("Most likely features for clusters with lowest prior probability")
indexes = low_class_log_prior_indexes
most_likely_feature_for_clusters()
# for idx in indexes:
#     print(idx)
#     for i in range(nb_model_final.n_features):
#         feat_idx = nb_model_final.feature_indexes_[i]
#         print(feature_names[i])
#         feat_names=[]
#         for j in feat_idx:
#             feat_names.append(vec.feature_names_[j])
#         feat_log_prob = []
#         for j in feat_idx:
#             feat_log_prob.append(nb_model_final.feature_log_prob_[idx][j])
#         print(feat_names)
#         print(np.exp(feat_log_prob))
#         max_val = np.max(feat_log_prob)
#         most_likely_feat_assign_idx = np.where(feat_log_prob == repmat(max_val,1,len(feat_log_prob)))
#         for j in feat_idx:
#             print(vec.feature_names_[j])


pred_prob = nb_model_final.predict_proba(x_vec)
best_class_labels = np.argmax(pred_prob,axis=1)

y_label_pred = []

for l in range(x_vec.shape[0]):
    y_label_pred.append(nb_model_final.classes_[best_class_labels[l]])

print("Most likely examples for each cluster in full data set")

for k, c in zip(range(nb_model_final.n_classes),nb_model_final.classes_):
    print(c)
    idx = []
    for l in range(len(y_label_pred)):
        if c == y_label_pred[l]:
            idx.append(l)
    val = np.sort(pred_prob[idx][k])[::-1][0]
    best_idx = []
    for l in idx:
        if pred_prob[l][k] == val:
            best_idx.append(l)
    print(val)
    l = best_idx[0]
    print(l)
    print(vec.inverse_transform(x_vec[l]))
    print(len(best_idx))


