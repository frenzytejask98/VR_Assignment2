#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


def sift_surf_generator(choice, image):
    if choice == 'sift':
        sifter = cv2.xfeatures2d.SIFT_create()
        (kp, des) = sifter.detectAndCompute(image, None)
    else:
        surfer = cv2.xfeatures2d.SURF_create()
        (kp, des) = surfer.detectAndCompute(image, None)
    return kp, des

def greyer(path):
    image = cv2.imread(path)
    greyimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return greyimage


# In[ ]:


categories = ['Bikes', 'Horses']
des_arr = []
new_des = []
labels = []
for category in categories:
    for i in range(len(next(os.walk('./' + category))[2])):
        if (next(os.walk('./' + category))[2][i] != '.DS_Store'):
            labels.append(category)
            kp, des = sift_surf_generator('sift', greyer("./" + category + "/" + next(os.walk('./' + category))[2][i]))
            des_arr.append(des)
            for descriptor in des:
                new_des.append(descriptor)


# In[ ]:


data = pd.DataFrame(data = new_des)
clf = MiniBatchKMeans(n_clusters = 20,random_state = 42,batch_size=100)
clf.fit(data)
img_clustered_words = [clf.predict(descriptor) for descriptor in des_arr]


# In[ ]:


bowimg_hist_arr = np.array([np.bincount(clusters, minlength = 20) for clusters in img_clustered_words])
len(bowimg_hist_arr[1])


# In[ ]:


features = pd.DataFrame(data = bowimg_hist_arr)
label_data = pd.DataFrame(data = labels)
label_data.columns = ['label']
label_data = label_data['label'].apply(lambda x: 0 if x == 'Bikes' else 1)
train = pd.concat([features, label_data], axis = 1)


# In[ ]:


scores=[]
clf = LinearSVC(tol=10e-5, max_iter=20000, multi_class='crammer_singer', random_state=42)
train_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
labels = ['label']
clf.fit(train[train_ind], train[labels])
cv_score = np.mean(cross_val_score(clf, train[train_ind], train[labels], cv=10, scoring='roc_auc'))
scores.append(cv_score)
print('Total CV score of SVM is {}'.format(np.mean(scores)))


# In[ ]:


scores1=[]
clf = LogisticRegression(C=2, random_state=42, solver='sag', max_iter=10000, multi_class='auto', penalty='l2', tol=1e-5)
clf.fit(train[train_ind], train[labels])
cv_score = np.mean(cross_val_score(clf, train[train_ind], train[labels], cv=10, scoring='roc_auc'))
scores1.append(cv_score)
print('Total CV score of LR is {}'.format(np.mean(scores1)))


# In[ ]:


scores2=[]
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(train[train_ind], train[labels])
cv_score = np.mean(cross_val_score(clf, train[train_ind], train[labels], cv=10, scoring='roc_auc'))
scores2.append(cv_score)
print('Total CV score for KNN is {}'.format(np.mean(scores2)))


# In[ ]:




