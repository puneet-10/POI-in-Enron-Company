#!/usr/bin/python

import sys
import matplotlib.pyplot as plt
import pickle
from sklearn import preprocessing
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
sys.path.append("../tools/")
from sklearn.metrics import accuracy_score
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ["poi", "fraction_from_poi_email", "fraction_to_poi_email", 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
features = ["salary", "bonus"]
#print(data_dict.values())
#print(data_dict.keys())
data_dict.pop('TOTAL',0)

a=[]
b=[]

for key in data_dict:
    if data_dict[key]['salary']!='NaN':
        if int(data_dict[key]['salary'])>600000:
            a.append(key)
    if data_dict[key]['bonus']!='NaN':
        if int(data_dict[key]['bonus'])>4000000:
            b.append(key)
#print(b) 
#print(a)
for j in b:
    data_dict.pop(j)            
for i in a:
    try:
        data_dict.pop(i)
    except:
            continue     
#or point in data_dict:
#    salary = point[0]
#    bonus = point[1]
 #   plt.scatter( salary, bonus )

#plt.xlabel("salary")
#plt.ylabel("bonus")
#plt.show()

my_dataset = data_dict

#print(data_dict['BUY RICHARD B'])
data=featureFormat(data_dict,features)
#print(data)
### Task 2: Remove outliers
outliers=[]
for key in data_dict:
    if data_dict[key]['salary']=='NaN':
        continue
    outliers.append((key,int(data_dict[key]['salary'])))
o=(sorted(outliers,key=lambda x:x[1],reverse=True)[:4])
#print(o)
### Task 3: Create new features

 
### Store to my_dataset for easy export below.

def dict_to_list(key,normalizer):
    l=[]
    for i in data_dict:
        if data_dict[i][key]=="NaN" or data_dict[i][normalizer]=="NaN":
            l.append(0.)
        elif data_dict[i][key]>=0: 
            l.append(float(data_dict[i][key])/float(data_dict[i][normalizer]))
    return l
fraction_from_poi_email=dict_to_list("from_poi_to_this_person","to_messages")
fraction_to_poi_email=dict_to_list("from_this_person_to_poi","from_messages")

count=0
c=[]
d=[]
for i in data_dict:
    data_dict[i]["fraction_from_poi_email"]=fraction_from_poi_email[count]
    data_dict[i]["fraction_to_poi_email"]=fraction_to_poi_email[count]
    count +=1
for s in data_dict:
    if float(data_dict[s]["fraction_from_poi_email"])>0.15:
        c.append(s)
    if float(data_dict[s]["fraction_to_poi_email"])>0.8:
        d.append(s)
#print(c,d)
for w in c:
    try:
        data_dict.pop(w)
    except:
        continue
for q in d:
    try:
        data_dict.pop(q)   
    except:
        continue
#for t in data_dict:
 #   try:
 #       print(data_dict[t]["fraction_from_poi_email"])
 #   except:
 #       continue
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)

for point in data:
    from_poi = point[1]
    to_poi = point[2]
    plt.scatter( from_poi, to_poi )
    if point[0] == 1:
        plt.scatter(from_poi, to_poi, color="r", marker="*")
plt.xlabel("fraction of emails this person gets from poi")
plt.show()
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test =train_test_split(features, labels, test_size=0.3, random_state=42)


from sklearn.cross_validation import KFold
kf=KFold(len(labels),3)
for train,test in kf:
     features_train= [features[ii] for ii in train]
     features_test= [features[ii] for ii in test]
     labels_train=[labels[ii] for ii in train]
     labels_test=[labels[ii] for ii in test]
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()
#clf.fit(features_train, labels_train)
#pred = clf.predict(features_test)
#accuracy = accuracy_score(pred,labels_test)
#print(accuracy)
from sklearn.tree import DecisionTreeClassifier



clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
score = clf.score(features_test,labels_test)
print( 'accuracy before tuning ', score)
importances = clf.feature_importances_
import numpy as np
indices = np.argsort(importances)[::-1]
#print ('Feature Ranking:')
#for i in range(16):
#    print ("{} feature {} ({})".format(i+1,features_list[i+1],importances[indices[i]]))
clf = DecisionTreeClassifier(min_samples_split=8)
clf = clf.fit(features_train,labels_train)
pred= clf.predict(features_test)
acc=accuracy_score(labels_test, pred)
print( "accuracy after tuning = ", acc)
print ('precision = ', precision_score(labels_test,pred))
print( 'recall = ', recall_score(labels_test,pred))


## Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

pickle.dump(clf, open("my_classifier.pkl", "w") )
pickle.dump(data_dict, open("my_dataset.pkl", "w") )
pickle.dump(features_list, open("my_feature_list.pkl", "w") )