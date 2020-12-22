#!/usr/bin/env python
#-*- coding:utf-8 -*-
# datetime:2020/12/8 10:49

import os
import pandas as pd
import numpy as np
import re
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")
from sklearn import svm


# 二十种不同的氨基酸
AminoAcids = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
ChemicalFeatures = [("C", "M"), ("A", "G", "P"), ("I", "L", "V"), ("D", "E"),
                    ("H", "K", "R"), ("F", "W", "Y"), ("N", "Q"), ("S", "T")]
PosLabel = '1'  # 可切割标记为1
NegLabel = '-1'  # 不可切割标记为-1


def __feature_oc(dc, amino_data):
    print("Use orthogonal coding scheme to construct feature vector")
    orthogonal_coding = dict()
    for x, amino in enumerate(AminoAcids[:-1]):
        num = [-1] * 19
        num[x] = 1
        orthogonal_coding[amino] = num[:]
    orthogonal_coding["Y"] = [-1] * 19
    train_x_data = []
    train_y_data = []
    train = []
    for index in range(len(amino_data)):
        feature = []
        for x in amino_data["Amino"][index]:
            feature.extend(orthogonal_coding[x])
        label = amino_data['Label'][index]
        train_x_data.append(feature)
        train_y_data.append(label)
    train.append(train_x_data)
    train.append(train_y_data)
    return train


def __feature_cc(dc, amino_data):
    print("Use chemical coding schemes to construct feature vectors")
    chemical_features = dict()
    for x, amino in enumerate(ChemicalFeatures[:-1]):
        num = [-1] * 7
        num[x] = 1
        chemical_features[amino] = num[:]
    chemical_features[ChemicalFeatures[-1]] = [-1]*7

    train_x_data = []
    train_y_data = []
    train = []
    for index in range(len(amino_data)):
        feature = []
        for x in amino_data["Amino"][index]:
            for group in ChemicalFeatures:
                if x in group:
                    feature.extend(chemical_features [group])
        label = amino_data['Label'][index]
        train_x_data.append(feature)
        train_y_data.append(label)
    train.append(train_x_data)
    train.append(train_y_data)
    return train


def __feature_dc(dc, amino_data):
    print("Use Evocleave coding schemes to construct feature vectors")
    train_x_data = []
    train_y_data = []
    train = []
    for index in range(len(amino_data)):
        feature = []
        for j in range(len(dc)):
            if re.search(dc['AminoT2'][j], amino_data['Amino'][index]):
                feature.append(dc['WeightC'][j])
            else:
                feature.append(0)
        label = amino_data['Label'][index]
        train_x_data.append(feature)
        train_y_data.append(label)
    train.append(train_x_data)
    train.append(train_y_data)
    return train


def __feature_oc_cc(dc, amino_data):
    print("Use orthogonal coding schemes and chemical coding schemes to construct feature vectors")
    orthogonal_coding = dict()
    for x, amino in enumerate(AminoAcids[:-1]):
        num = [-1] * 19
        num[x] = 1
        orthogonal_coding[amino] = num[:]
    orthogonal_coding["Y"] = [-1] * 19

    chemical_features = dict()
    for x, amino in enumerate(ChemicalFeatures[:-1]):
        num = [-1] * 7
        num[x] = 1
        chemical_features[amino] = num[:]
    chemical_features[ChemicalFeatures[-1]] = [-1]*7

    train_x_data = []
    train_y_data = []
    train = []
    for index in range(len(amino_data)):
        feature = []
        for x in amino_data["Amino"][index]:
            feature.extend(orthogonal_coding[x])

        for x in amino_data["Amino"][index]:
            for group in ChemicalFeatures:
                if x in group:
                    feature.extend(chemical_features[group])
        label = amino_data['Label'][index]
        train_x_data.append(feature)
        train_y_data.append(label)
    train.append(train_x_data)
    train.append(train_y_data)
    return train


def __feature_cc_dc(dc, amino_data):
    print("Use chemical coding schemes and Evocleave coding schemes to construct feature vectors")
    chemical_features = dict()
    for x, amino in enumerate(ChemicalFeatures[:-1]):
        num = [-1] * 7
        num[x] = 1
        chemical_features[amino] = num[:]
    chemical_features[ChemicalFeatures[-1]] = [-1]*7

    train_x_data = []
    train_y_data = []
    train = []
    for index in range(len(amino_data)):
        feature = []

        for x in amino_data["Amino"][index]:
            for group in ChemicalFeatures:
                if x in group:
                    feature.extend(chemical_features[group])

        for j in range(len(dc)):
            if re.search(dc['AminoT2'][j], amino_data['Amino'][index]):
                feature.append(dc['WeightC'][j])
            else:
                feature.append(0)

        label = amino_data['Label'][index]
        train_x_data.append(feature)
        train_y_data.append(label)

    train.append(train_x_data)
    train.append(train_y_data)
    return train


def __feature_oc_dc(dc, amino_data):
    print("Use orthogonal coding schemes and Evocleave coding schemes to construct feature vectors")
    orthogonal_coding = dict()
    for x, amino in enumerate(AminoAcids[:-1]):
        num = [-1] * 19
        num[x] = 1
        orthogonal_coding[amino] = num[:]
    orthogonal_coding["Y"] = [-1] * 19

    train_x_data = []
    train_y_data = []
    train = []
    for index in range(len(amino_data)):
        feature = []
        for x in amino_data["Amino"][index]:
            feature.extend(orthogonal_coding[x])

        for j in range(len(dc)):
            if re.search(dc['AminoT2'][j], amino_data['Amino'][index]):
                feature.append(dc['WeightC'][j])
            else:
                feature.append(0)
        label = amino_data['Label'][index]
        train_x_data.append(feature)
        train_y_data.append(label)

    train.append(train_x_data)
    train.append(train_y_data)
    return train


def __feature_mul(dc, amino_data):
    print("Use orthogonal coding schems, chemical coding schemes and Evocleave coding schemes to construct feature vectors")
    orthogonal_coding = dict()
    for x, amino in enumerate(AminoAcids[:-1]):
        num = [-1] * 19
        num[x] = 1
        orthogonal_coding[amino] = num[:]
    orthogonal_coding["Y"] = [-1] * 19

    chemical_features = dict()
    for x, amino in enumerate(ChemicalFeatures[:-1]):
        num = [-1] * 7
        num[x] = 1
        chemical_features[amino] = num[:]
    chemical_features[ChemicalFeatures[-1]] = [-1]*7

    train_x_data = []
    train_y_data = []
    train = []
    for index in range(len(amino_data)):
        feature = []
        for x in amino_data["Amino"][index]:
            feature.extend(orthogonal_coding[x])

        for x in amino_data["Amino"][index]:
            for group in ChemicalFeatures:
                if x in group:
                    feature.extend(chemical_features[group])

        feature_evo = []
        for j in range(len(dc)):
            if re.search(dc['AminoT2'][j], amino_data['Amino'][index]):
                feature_evo.append(dc['WeightC'][j])
            else:
                feature_evo.append(0)
        feature.extend(feature_evo)

        label = amino_data['Label'][index]
        train_x_data.append(feature)
        train_y_data.append(label)
    train.append(train_x_data)
    train.append(train_y_data)
    return train


def __train_test_model(data):
    set_c = [0.03123, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32]
    set_b = [2, 5, 10, 20, 30, 50, 100, 200]
    print("The result of Biased-SVM is running")
    try:
        f = open('../results/result.txt', 'w+', encoding='utf-8')
    except Exception as e:
        print(e)

    (train_data, test_data) = data

    (x_train, y_train) = train_data
    (x_test, y_test) = test_data

    for c in set_c:
        f.write("C={}\n".format(c))
        for b in set_b:
            f.write("\n b={}\n".format(b))
            f.write("label 1 -1\n")
            clf = svm.SVC(kernel='linear', probability=True, random_state=42, max_iter=2000,
                          class_weight={-1: c / b, 1: c})
            clf.fit(x_train, y_train)
            y_test = list(y_test)
            for l, s in enumerate(clf.predict_proba(x_test)):
                f.write("{} {} {}\n".format(y_test[l], s[1], s[0]))
        f.write('\n--------------------------------------------\n')
    f.close()
    print("The result of Biased-SVM method is saved")


def __cross_validation(data):
    set_c = [0.03123, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32]
    set_b = [2, 5, 10, 20, 30, 50, 100, 200]
    print("The result of 10-fold Cross Validation is running")
    try:
        f = open('../results/result.txt', 'w+', encoding='utf-8')
    except Exception as e:
        print(e)

    (data_p, data_n) = data

    kf = KFold(n_splits=10, shuffle=False)

    xp = np.array(data_p[0])
    yp = np.array(data_p[1])
    xp_train = []
    xp_test = []
    for train_index, test_index in kf.split(xp):
        xp_train.append(train_index)
        xp_test.append(test_index)

    xn = np.array(data_n[0])
    yn = np.array(data_n[1])
    xn_train = []
    xn_test = []
    for train_index, test_index in kf.split(xn):
        xn_train.append(train_index)
        xn_test.append(test_index)

    for c in set_c:
        f.write("C={}\n".format(c))
        for b in set_b:
            f.write("\n b={}\n".format(b))
            f.write("label 1 -1\n")
            clf = svm.SVC(kernel='linear', probability=True, random_state=42, max_iter=2000,
                          class_weight={-1: c / b, 1: c})
            for i in range(10):
                x1_train, y1_train, x1_test, y1_test = xp[xp_train[i]], yp[xp_train[i]], xp[xp_test[i]], yp[xp_test[i]]
                x2_train, y2_train, x2_test, y2_test = xn[xn_train[i]], yn[xn_train[i]], xn[xn_test[i]], yn[xn_test[i]]

                x_train = np.append(x1_train, x2_train, axis=0)
                y_train = np.append(y1_train, y2_train, axis=0)
                x_test = np.append(x1_test, x2_test, axis=0)
                y_test = np.append(y1_test, y2_test, axis=0)

                clf.fit(x_train, y_train)
                y_test = list(y_test)
                for l, s in enumerate(clf.predict_proba(x_test)):
                    f.write("{} {} {}\n".format(y_test[l], s[1], s[0]))
        f.write('\n-------------------------------------------------\n')
    f.close()
    print("The result of 10-fold Cross Validation is saved")

def __extract_dc():

    ls = []
    for i in range(7):
        try:
            ls.append(pd.read_table('../sample/rules/1/{}'.format(str(i)), sep=',', names=['AminoT2', 'WeightC']))
        except Exception as e:
            print(e)
            print("Execute the command ‘java -jar EvoCleave.jar Sample 16 0’ to generate the corresponding Evocleave co-evolution patterns")
            exit()
    dc = pd.concat(ls, axis=0)
    dc.index = range(len(dc))
    return dc

def __vector_construction(feature_method, dc, *frame):
    if len(frame) == 1:
        frame = frame[0]
        amino_p = frame[frame['Label'] > 0]
        amino_p.index = range(len(amino_p))

        amino_n = frame[frame['Label'] < 0]
        amino_n.index = range(len(amino_n))

        data = [feature_method(dc, amino_p), feature_method(dc, amino_n)]
    else:
        data = [feature_method(dc, frame[0]), feature_method(dc, frame[1])]
    return data


def run(code, *filename):
    code_feature = {
        "0": __feature_oc,
        "1": __feature_cc,
        "2": __feature_dc,
        "3": __feature_oc_cc,
        "4": __feature_cc_dc,
        "5": __feature_oc_dc,
        "6": __feature_mul
    }
    if code not in code_feature.keys():
        print("the code of feature selection is error")
        print("code 0: feature_oc")
        print("code 1: feature_cc")
        print("code 2: feature_dc")
        print("code 3: feature_oc_cc")
        print("code 4: feature_cc_dc")
        print("code 5: feature_oc_dc")
        print("code 6: feature_mul")
        return
    else:
        feature_method = code_feature[code]
        if filename[0] == filename[1]:
            frame = pd.read_table('../data/{}Data.txt'.format(filename[0]), sep=',', names=['Amino', 'Label'])
            dc= pd.read_table('../data/{}Datadc.txt'.format(filename[0]), sep=',', names=['AminoT2', 'WeightC'])
            data = __vector_construction(feature_method, dc, frame)
            __cross_validation(data)
        else:
            try:
                if filename[0] == "train":
                    frame_train_p = pd.read_table('../sample/train/pos', sep=',', names=['Amino', 'Label'])
                    frame_train_n = pd.read_table('../sample/train/neg', sep=',', names=['Amino', 'Label'])
                    frame_train = pd.concat([frame_train_p, frame_train_n], axis=0)
                    frame_train.index = range(len(frame_train))
                    dc = __extract_dc()
                else:
                    try:
                        frame_train = pd.read_table('../data/{}Data.txt'.format(filename[0]), sep=',', names=['Amino', 'Label'])
                        dc= pd.read_table('../data/{}Datadc.txt'.format(filename[0]), sep=',', names=['AminoT2', 'WeightC'])
                    except Exception as e:
                        print(e)
                        print("train file not found")
                        exit()

                if filename[1] == "test":
                    frame_test_p = pd.read_table('../sample/test/pos', sep=',', names=['Amino', 'Label'])
                    frame_test_n = pd.read_table('../sample/test/neg', sep=',', names=['Amino', 'Label'])
                    frame_test = pd.concat([frame_test_p, frame_test_n], axis=0)
                    frame_test.index = range(len(frame_test))
                else:
                    try:
                        frame_test = pd.read_table('../data/{}Data.txt'.format(filename[1]), sep=',', names=['Amino', 'Label'])
                    except Exception as e:
                        print(e)
                        print("test file not found")
                        exit()
                
                data = __vector_construction(feature_method, dc, frame_train, frame_test)
                __train_test_model(data)
            except FileNotFoundError as e:
                print(e)