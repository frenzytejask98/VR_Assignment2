import os
import cv2
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

def sift_surf_generator(type, image):
    if type == 'sift':
        sift = cv2.xfeatures2d.SIFT_create()
        (kp, des) = sift.detectAndCompute(image, None)
    else:
        surf = cv2.xfeatures2d.SURF_create()
        (kp, des) = surf.detectAndCompute(image, None)
    return kp, des

def sift_surf_gen_looper(images, label_arr):
    input_data = np.array(images)
    des_arr = []
    new_des = []
    labels = []
    for i, image in enumerate(input_data, 0):
        image_transpose = image.transpose()
        kp, des = sift_surf_generator('sift', cv2.cvtColor(image_transpose, cv2.COLOR_BGR2GRAY))
        if (len(kp) != 0):
            labels.append(label_arr[i])
            des_arr.append(des)
            for descriptor in des:
                new_des.append(descriptor) 
    return new_des, des_arr, labels

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict

def cifar10_importer():
    images = []
    label_arr = []
    for j in range(1,6):
        datadict = unpickle('./cifar-10-batches-py/data_batch_' + str(j))
        data = datadict[b'data'] 
        label_arr.append(datadict[b'labels'])
        for i in range(data.shape[0]):
            image = []
            r = data[i][0:1024].reshape(32, 32)
            g = data[i][1024:2048].reshape(32, 32)
            b = data[i][2048:3072].reshape(32, 32)
            image.append(r)
            image.append(g)
            image.append(b)
            images.append(image)
    label_arr = np.array(label_arr) 
    label_arr = label_arr.flatten().tolist()
    return images, label_arr

def clustering(tot_des_arr, img_des_arr, labels, cluster_size):
    data = pd.DataFrame(data = tot_des_arr)
    clf = MiniBatchKMeans(n_clusters = cluster_size, batch_size=100)
    clf.fit(data)
    img_clustered_words = [clf.predict(descriptor) for descriptor in img_des_arr]
    bow_img_hist_arr = np.array([np.bincount(clusters, minlength = cluster_size) for clusters in img_clustered_words])
    return bow_img_hist_arr, labels

def data_prep(bow_img_hist_arr, labels):
    svm_data = pd.DataFrame(data = bow_img_hist_arr)
    label_data = pd.DataFrame(data = labels)
    label_data.columns = ['label']
    #label_data = label_data['label'].apply(lambda x: 0 if x == 'Bikes' else 1)
    final_data = pd.concat([svm_data, label_data], axis = 1)
    train, test = train_test_split(final_data, test_size = 0.1)
    train_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
        39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,  98,  99]
    labels = ['label']
    print (labels)
    return train, test, train_index, labels

def svm_c(train, test, train_index, labels):
    clf = LinearSVC(max_iter=5000)
    clf.fit(train[train_index], train[labels])
    pred = clf.predict(test[train_index])
    return accuracy_score(pred, test[labels])

def lr(train, test, train_index, labels):
    clf = LogisticRegression(solver='saga', multi_class='multinomial',max_iter=10000)
    clf.fit(train[train_index], train[labels])
    pred = clf.predict(test[train_index])
    return accuracy_score(pred, test[labels])

def knn_c(train, test, train_index, labels):
    clf = KNeighborsClassifier(n_neighbors = 2)
    clf.fit(train[train_index], train[labels])
    pred = clf.predict(test[train_index])
    return accuracy_score(pred, test[labels])

images, label_arr = cifar10_importer()
tot_des_arr, img_des_arr, labels = sift_surf_gen_looper(images, label_arr)
bow_img_hist_arr, labels = clustering(tot_des_arr, img_des_arr, labels, 100)
train, test, train_index, labels = data_prep(bow_img_hist_arr, labels)
print (train)
svm_acc = svm_c(train, test, train_index, labels)
print(svm_acc)
lr_acc = lr(train, test, train_index, labels)
print(lr_acc)
knn_acc = knn_c(train, test, train_index, labels)
print(knn_acc)
