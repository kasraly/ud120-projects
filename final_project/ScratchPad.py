# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 16:38:48 2017

@author: Kasra
"""

data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
labels = np.array(labels)
features = np.array(features)

xind = 0 #salary
yind = 1 #from_poi_to_this_person
x = features[:,xind]
y = features[:,yind]

plt.scatter( x[labels==1], y[labels==1], )
plt.scatter( x[labels==0], y[labels==0], )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

nan_dic = {}
for key in data_dict:
    for feat in data_dict[key]:
        if data_dict[key][feat] == 'NaN' and data_dict[key]['poi'] == 1:
#        if data_dict[key][feat] == 'NaN':
            if feat in nan_dic:
                nan_dic[feat] += 1
            else:
                nan_dic[feat] = 1
pp.pprint(nan_dic)



#%%
x = features[:,10] #shared_receipt_with_poi_ratio
y = features[:,4] #to_messages_poi_ratio
plt.scatter( x[labels==1], y[labels==1], )
plt.scatter( x[labels==0], y[labels==0], )