import csv
import numpy
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
import os

from sklearn.preprocessing import StandardScaler

l = 0
for k in range(4):

    minta = ['minta_3', 'minta_5', 'minta_10', 'minta_20']
    path = 'CSV/' + minta[l]
    eredmeny = ['eltolas_3.csv', 'eltolas_5.csv', 'eltolas_10.csv', 'eltolas_20.csv']
    print(path)
    dir_list = os.listdir(path)
    # print("Files and directories in '", path, "' :")
    # print("length: ", len(dir_list))

    # prints all files
    # print(dir_list)

    path_label = 'label'
    dir_list_label = os.listdir(path_label)
    # print("Files and directories in '", path_label, "' :")
    # print("length: ", len(dir_list_label))

    # prints all files
    # print(dir_list_label)

    features_all = []
    labels_all = []

    number = 0.00001
    # for i in range(75):
    #    print(i,dir_list[i].split("_")[0]+dir_list[i].split("_")[1]+dir_list[i].split("_")[2], dir_list_label[i].split("_")[0]+dir_list_label[i].split("_")[1]+dir_list_label[i].split("_")[2])

    file = open(eredmeny[l], 'w', encoding='UTF8', newline='')
    header = ['complexity', 'accuracy']
    with file:
        writer = csv.DictWriter(file, fieldnames=header)

    for i in dir_list:
        with open('CSV/' + minta[l] + '/' + i, encoding='utf-8-sig') as csvfile_train:
            spamreader = csv.reader(csvfile_train, delimiter=',', quotechar='|')
            for row in spamreader:
                features_all.append(row)

    for i in dir_list_label:
        with open('label/' + i, encoding='utf-8-sig') as csvfile_train:
            spamreader = csv.reader(csvfile_train, delimiter=',', quotechar='|')
            for row in spamreader:
                labels_all.append(row)
                # print(row)

    print(len(features_all))
    print(len(labels_all))

    a = numpy.array(features_all)
    b = numpy.array(labels_all)
    # print("a: ", a)
    # print("a: ", a.shape)
    # print(b.shape)
    c = np.reshape(b, 75)
    # print(c.shape)

    # standard
    # a = numpy.array(features_all)
    scaler = StandardScaler()
    SS_train = scaler.fit(a)
    trans = SS_train.transform(a)
    print("standard: ", SS_train)
    me = scaler.mean_
    print("me :", me)
    print(type(me))


    # test train split
    X_train, X_test, y_train, y_test = train_test_split(trans, c, test_size=0.2, random_state=0)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    # SS_train = scaler.fit(X_train)
    # me = scaler.mean_
    # trans = SS_train.transform(X_train)

    # clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    # scores = clf.score(X_test, y_test)

    for i in range(7):
        clf = svm.SVC(kernel='linear', C=number, random_state=42)
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        print("c értéke:", number, "test X: ", X_test, ", test y: ", y_test)
        print("%0.5f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
        print('scores', scores)

        file = open(eredmeny[l], 'a+', encoding='UTF8', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow([number, scores.mean()])
        number = number * 10
        print("c uj erteke: ", number)
    l = l + 1
