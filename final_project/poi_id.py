#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

import pprint
pp = pprint.PrettyPrinter(depth=6)
import numpy as np

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'bonus', 'expenses', 'from_this_person_to_poi']
#, 'expenses', 'from_messages', 'from_this_person_to_poi', 'to_messages', 'from_poi_to_this_person', 'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
#%%
### Task 2: Remove outliers
data_dict.pop('TOTAL', 0)

#%%
### Task 3: Create new feature(s)
for key in data_dict:
    if data_dict[key]['from_messages'] == 'NaN' or data_dict[key]['from_this_person_to_poi'] == 'NaN':
        data_dict[key]['from_messages_poi_ratio'] = 0
    else:                 
        data_dict[key]['from_messages_poi_ratio'] = float(data_dict[key]['from_this_person_to_poi'])/data_dict[key]['from_messages']

    if data_dict[key]['to_messages'] == 'NaN' or data_dict[key]['from_poi_to_this_person'] == 'NaN':
        data_dict[key]['to_messages_poi_ratio'] = 0
    else:
        data_dict[key]['to_messages_poi_ratio'] = float(data_dict[key]['from_poi_to_this_person'])/data_dict[key]['to_messages']

    if data_dict[key]['to_messages'] == 'NaN' or data_dict[key]['shared_receipt_with_poi'] == 'NaN':
        data_dict[key]['shared_receipt_with_poi_ratio'] = 0
    else:
        data_dict[key]['shared_receipt_with_poi_ratio'] = float(data_dict[key]['shared_receipt_with_poi'])/data_dict[key]['to_messages']
    
    if data_dict[key]['bonus'] == 'NaN' or data_dict[key]['salary'] == 'NaN':
        data_dict[key]['bonus_to_salary'] = 0
    else:
        data_dict[key]['bonus_to_salary'] = float(data_dict[key]['bonus'])/data_dict[key]['salary']
    
# 0.55, 0.55
#features_list = ['poi','salary', 'bonus', 'expenses', 'from_messages_poi_ratio', 'to_messages_poi_ratio', 'from_messages', 'from_this_person_to_poi', 'to_messages', 'from_poi_to_this_person', 'shared_receipt_with_poi', 'shared_receipt_with_poi_ratio', 'bonus_to_salary', 'restricted_stock'] # You will need to use more features
# 0.6, 0.4
#features_list = ['poi', 'salary', 'total_payments', 'bonus', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
# 0.73, 0.38
#features_list = ['poi', 'total_payments', 'bonus', 'deferred_income', 'expenses', 'other', 'restricted_stock', 'from_this_person_to_poi']
# 0.51, 0.38
#features_list = ['poi', 'salary', 'bonus', 'expenses', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
# 0.66, 0.4
#features_list = ['poi', 'salary', 'bonus', 'expenses', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi', 'shared_receipt_with_poi_ratio', 'from_messages_poi_ratio', 'to_messages_poi_ratio']
# 0.76, 0.45
#features_list = ['poi', 'deferred_income', 'expenses', 'exercised_stock_options', 'other']
# 0.78, 0.5
features_list = ['poi', 'deferred_income', 'expenses', 'exercised_stock_options', 'other', 'shared_receipt_with_poi_ratio', 'from_messages_poi_ratio', 'to_messages_poi_ratio']
# 0.8, 0.45
#features_list = ['poi', 'deferred_income', 'expenses', 'exercised_stock_options', 'other', 'shared_receipt_with_poi_ratio', 'from_messages_poi_ratio', 'to_messages_poi_ratio', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
# 0.5, 0.4
#features_list = ['poi', 'exercised_stock_options', 'other', 'bonus_to_salary', 'shared_receipt_with_poi_ratio', 'from_messages_poi_ratio', 'to_messages_poi_ratio']
# 0.54, 0.4
#features_list = ['poi', 'total_payments', 'bonus', 'deferred_income', 'expenses', 'other', 'restricted_stock', 'from_this_person_to_poi', 'shared_receipt_with_poi_ratio', 'from_messages_poi_ratio', 'to_messages_poi_ratio']
# 0.62, 0.38
#features_list = ['poi', 'deferred_income', 'expenses', 'other', 'restricted_stock', 'from_this_person_to_poi']
# 0.6, 0.53
#features_list = ['poi', 'deferred_income', 'expenses', 'other', 'restricted_stock', 'from_this_person_to_poi', 'shared_receipt_with_poi_ratio', 'from_messages_poi_ratio', 'to_messages_poi_ratio']
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

labels = np.array(labels)
features = np.array(features)

#%%
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


#%%
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import GridSearchCV

skf  = StratifiedKFold(labels, n_folds=5, random_state=42)
#scores = cross_val_score(clf, features, labels, cv=skf, scoring='f1')

#parameters_svc = {'C':[0.1, 1, 10, 100, 1000, 10000]}
#parameters_tree = {'max_features':range(2, len(features_list)), 
#                   'min_samples_split':[2, 4, 6, 8], 
#                   'min_samples_leaf':[1, 2, 3, 4]}
#dtc = DecisionTreeClassifier(random_state=42)
#clfGS = GridSearchCV(dtc, parameters_tree, cv=skf, scoring='f1')
#clfGS.fit(features, labels)
#dtc = clfGS.best_estimator_
#print dtc, clfGS.best_score_

#parameters_abc = {"n_estimators" : [5, 10, 20, 40, 80], 
#                  "learning_rate" : [.25, .5, 1, 2]}
#abc = AdaBoostClassifier(random_state=42)
#clfGS = GridSearchCV(abc, parameters_abc, cv=skf, scoring='f1')
#clfGS.fit(features, labels)
#abc = clfGS.best_estimator_
#print abc, clfGS.best_score_

#%%
#clf = abc
clf = AdaBoostClassifier(n_estimators=40, learning_rate=1, random_state=42)

from sklearn.cross_validation import cross_val_score

print 'accuracy: ', cross_val_score(clf, features, labels, cv=skf, scoring='accuracy').mean()
print 'precision: ', cross_val_score(clf, features, labels, cv=skf, scoring='precision').mean()
print 'recall: ', cross_val_score(clf, features, labels, cv=skf, scoring='recall').mean()
print 'f1: ', cross_val_score(clf, features, labels, cv=skf, scoring='f1').mean()

#%%
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)