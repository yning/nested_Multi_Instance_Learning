#!/usr/bin/env python
#-*- coding:utf-8 -*-

__author__ = "Yue Ning"
__email__ = "yning@vt.edu"
__version__ = "0.0.1"

import sklearn
import sklearn.svm
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.base import ClassifierMixin
from sklearn.grid_search import RandomizedSearchCV
import scipy.stats
from copy import deepcopy
import numpy as np
from numpy import linalg as la
import pdb
import misvm_sklearn
from scipy.sparse import issparse, vstack as parse_vstack
import random
import math
from collections import OrderedDict
from datetime import datetime, timedelta

def sigmoid(x):
  return 1. / (1. + math.exp(-x))


def _vstack(m):
    """
    Return vstack
    """
    if issparse(m[0]) or issparse(m[-1]):
        return parse_vstack(m)
    else:
        return np.vstack(m)


class nMIL:
    def __init__(self, bagsize, beta=3.0, gamma=0.5, m0=0.5, p0=0.5):
        self.bagsize = bagsize
        self.beta = beta
        self.gamma = gamma
        self.m0 = m0
        self.p0 = p0

    def grad_func(X, Y, orgX, w):

        beta = 3.0; m0 =  0.5; p0 = 0.5; beta2 = 6.0; gamma = 0.5
        first_matrix = []
        second_matrix = []
        third_matrix = []
        for i, superbag in enumerate(X):
            jk_plus = []
            P_i_list = []
            m_i = len(X[i])
            P_ijk_list = []
            n_ij_list = []
            
            for j, bag in enumerate(superbag):
                m_i = len(bag)
                k_plus = [k for k, doc in enumerate(bag) 
                        if np.sign(sigmoid(np.dot(w, doc[:, np.newaxis])) - p0) * np.dot(w, doc[:, np.newaxis]) < m0]
                p_ij_list = [sigmoid(np.dot(w, doc[:, np.newaxis])) for k, doc in enumerate(bag)]
                jk_plus.append(k_plus)
                n_ij_list.append(m_i)
                P_ijk_list.append(p_ij_list)
                p_ij = np.mean(p_ij_list)
                P_i_list.append(p_ij)
            
            P_i = np.mean(P_i_list)

            for j, x in enumerate(X[i]):
                if len(x) > 0:
                    row_list = [X[i][j][k] * P_ijk_list[j][k]\
                                * (1. - P_ijk_list[j][k])\
                                * ((Y[i] - P_i) / (P_i * (1.- P_i)))\
                                * (1. / n_ij_list[j])\
                                for k in xrange(len(X[i][j]))]

                    first_matrix.append(np.sum(np.array(row_list), axis=0))

            for j, x in enumerate(X[i]):
                if len(x) > 0:
                    row_list = [(X[i][j][k]\
                                * P_ijk_list[j][k]\
                                * (1. - P_ijk_list[j][k])\
                                * (1. / n_ij_list[j]))\
                                for k in xrange(len(x))]
                    if j > 0 and len(X[i][j-1]) > 0:
                        row_list2 = [(X[i][j-1][k]\
                                    * P_ijk_list[j-1][k]\
                                    * (1. - P_ijk_list[j-1][k])\
                                    * (1. / n_ij_list[j-1]))\
                                    for k in xrange(len(X[i][j-1]))]
                        current_sum = np.sum(np.array(row_list), axis=0)
                        last_sum = np.sum(np.array(row_list2), axis=0)

                        derivative = 2. * (1./ len(X[i]))\
                                        * (P_i_list[j] -  P_i_list[j-1])\
                                        * (current_sum - last_sum)\
                                        * orgX[i][j]['cross_cosine']\
                                        /(len(X[i][j]) * len(X[i][j-1]))
                        second_matrix.append(derivative)

            for idj, kplus in enumerate(jk_plus):
                if len(kplus) > 0:
                    row_list = [X[i][idj][idk]\
                                * np.sign(P_ijk_list[idj][idk] - p0)\
                                * (1. / n_ij_list[idj])\
                                for idk in kplus]

                    sum_row = np.sum(np.array(row_list), axis=0)
                    third_matrix.append(sum_row * (1./ n_ij_list[idj]))


        first_sum = np.sum(np.array(first_matrix), axis=0) * beta
        second_sum = np.sum(np.array(second_matrix), axis=0) * gamma
        third_sum = np.sum(np.array(third_matrix), axis=0) * gamma

        if len(second_matrix) > 0 and len(third_matrix) > 0:
            return -first_sum + second_sum - third_sum 
        else:
             return -first_sum 
   
    def prepare_data(self, docMap, X_docs, dataIndex, X, Y, Z, origianl):

        pos_count = 0
        allcount = 0
        for cVal in dataIndex.viewvalues():
            Z.append(cVal)
            if cVal['Y']:
                Y.append(1)
                pos_count += 1
            else:
                Y.append(0)

            combinedDoc = []
            featureDoc = []
            for i in range(max(1, 11 - self.bagsize), 11):
                if i == 1:
                    cosine_sim = 1.0
                else:
                    cosine_sim = cVal['history'][str(i)]['cross_cosine']

                doc_vec = X_docs[[docMap[d] for d in
                                  cVal['history'][str(i)]['ids']], :]
                doc_ids = cVal['history'][str(i)]['ids']
                x_dict = {'docvec': doc_vec,
                          'docids': doc_ids, 'cross_cosine': cosine_sim}
                combinedDoc.append(x_dict)
                featureDoc.append(doc_vec)

            origianl.append(combinedDoc)
            X.append(featureDoc)

            allcount += 1
        return pos_count, allcount


    def read_data(self, **kwargs):

        trainUnsorted = kwargs['trainIndex']
        testUnsorted = kwargs['testIndex']
        docIndex = kwargs['docIndex']
        trainIndex = OrderedDict(sorted(trainUnsorted.items(), key=lambda x:x[1]['time'][:10]))
        testIndex = OrderedDict(sorted(testUnsorted.items(), key=lambda x:x[1]['time'][:10]))
        print "train data start and end"
        print trainIndex[trainIndex.keys()[0]]['time']
        print trainIndex[trainIndex.keys()[-1]]['time']


        print "test data start and end"
        print testIndex[testIndex.keys()[0]]['time']
        print testIndex[testIndex.keys()[-1]]['time']
        print "number of documents:", len(docIndex)

        # print backward
        docItems = docIndex.items()
        self.x_dimension = len(docIndex.values()[0])
        docMap = {val[0]: index for index, val in enumerate(docItems)}
        X_docs = np.array([k[1] for k in docItems])

        self._trainY = []
        self._trainX = []
        self.original_trainX = []
        self._trainZ = []
        pc_train, c_train = self.prepare_data(docMap, 
                                            X_docs, 
                                            trainIndex, 
                                            self._trainX, 
                                            self._trainY, 
                                            self._trainZ, 
                                            self.original_trainX)
        print "Traing: Positive %d, negative %d" % (pc_train, c_train - pc_train)

        self._testY = []
        self._testX = []
        self.original_testX = []
        self._testZ = []  
        pc_test, c_test = self.prepare_data(docMap, 
                                            X_docs, 
                                            testIndex, 
                                            self._testX, 
                                            self._testY, 
                                            self._testZ, 
                                            self.original_testX) 
        print "Testing: Positive %d, negative %d" % (pc_test, c_test - pc_test)
        return

    def SGD(self):

        train_X = np.array(self._trainX)
        train_Y = np.array(self._trainY)
        print "X: {}".format(train_X.shape)
        print "Y: {}".format(train_Y.shape)
        OriginalTrain_X = self.original_trainX

        test_X = np.array(self._testX)
        test_Y = np.array(self._testY)

        lambd = 0.05
        iteration = 2000
        x_dimension = 300
        n_sgd = 10
        test_f1_arr = []; test_recall_arr = []; test_prec_arr = [];  test_roc_arr = []; test_acc_arr = []
        train_f1_arr = []; train_recall_arr = []; train_prec_arr = []; train_roc_arr = []
        test_score = []
        train_score = []
        for expr in range(n_sgd):
            w = np.random.rand(x_dimension)
            for t in range(iteration):
                eta = 1./((t + 1) * lambd)

                kset = random.sample(range(0, len(train_X) - 1), 10)

                X = np.array([train_X[z] for z in kset])
                Y = np.array([train_Y[z] for z in kset])
                orgX = [OriginalTrain_X[k] for k in kset]
                delta_w = self.grad_func(X, Y, orgX, w)

                new_w = np.dot((1 - eta * lambd), w) - eta * delta_w / len(X)

                rate = (1./np.sqrt(lambd)) * (1./la.norm(new_w))
                if  rate < 1.:
                    w = np.dot(rate, new_w)
                else:
                    w = new_w

            pred_Y = []
            predicted_data = []
            gsr_history_probs = {}
            for idx, testx in enumerate(test_X):
                p_ij_list = []
                originalId = test_idx[idx]
                for j, day in enumerate(testx):
                    p_ijk_list = [sigmoid(np.dot(w, doc[:, np.newaxis])) for k, doc in enumerate(day)]

                    p_ij_list.append(p_ijk_list)

                days_list = [np.mean(p_ijk_list) for p_ijk_list in p_ij_list]

                P_i =  np.mean(days_list)

                bag = self.original_testX[idx]
                bag_gsrId = self._testZ[idx]['Id']

                gsr_history_probs[bag_gsrId] = {}
                gsr_history_probs[bag_gsrId]['trueY'] = data_Y[originalId]
                for idx, histday in enumerate(bag):
                    
                    doc_vec = histday['docvec']
                    doc_ids = histday['docids']
                    today_probs = [sigmoid(np.dot(w.T, doc)) for k, doc in enumerate(doc_vec)]

                    a = zip(doc_ids, today_probs)
                    gsr_history_probs[bag_gsrId][idx] = a
                                originalnews = self._testZ[idx]
            test_f1 = sklearn.metrics.f1_score(test_Y, pred_Y)
            test_recall = sklearn.metrics.recall_score(test_Y, pred_Y)
            test_precision = sklearn.metrics.precision_score(test_Y, pred_Y)
            test_roc = sklearn.metrics.roc_auc_score(test_Y, pred_Y)
            test_acc = sklearn.metrics.accuracy_score(test_Y, pred_Y)

            test_f1_arr.append(test_f1)
            test_recall_arr.append(test_recall)
            test_prec_arr.append(test_precision)
            test_roc_arr.append(test_roc)
            test_acc_arr.append(test_acc)

        test_score = [np.mean(np.array(test_acc_arr)),
                      np.mean(np.array(test_recall_arr)),
                      np.mean(np.array(test_prec_arr)),
                      np.mean(np.array(test_f1_arr)),
                      np.mean(np.array(test_roc_arr))]
        print "test bagsize:", self.bagsize,
        print "accuracy:", np.mean(np.array(test_acc_arr)),
        print "recall:",  np.mean(np.array(test_recall_arr)),
        print "recall:",  np.mean(np.array(test_prec_arr)),
        print "f1-score:", np.mean(np.array(test_f1_arr))

        return self, test_score, train_score, gsr_history_probs

def main(args):
    import json
    import os

    trainfname = "input_forClassification/country-%s/leadtime-%d/%s"\
            % (args.country, args.leadtime, args.train)
    trainf =  args.path + trainfname

    # Testing data
    testfname = "input_forClassification/country-%s/leadtime-%d/%s"\
            % (args.country, args.leadtime, args.test)
    testf = args.path + testfname  


    dfname = "news_deepfeature/news_doc2vec_%s.json" % args.country
    docf = args.path + dfname

    ### Output Files:
    resultf = '../result/{}_{}_lt-{}.txt'.format(args.resultfile, args.country, args.leadtime)
    trainMap = {}
    with open(trainf) as infile:
        for line in infile:
            j = json.loads(line.strip())
            trainMap[len(trainMap)] = j

    testMap = {}
    with open(testf) as infile:
        for line in infile:
            j = json.loads(line.strip())
            testMap[len(testMap)] = j

    with open(docf) as infile:
        docMap = {j['Id']: j['doc2vec'] for j in
                  (json.loads(l) for l in infile)}

    day = args.historyDays


    start  = datetime.now()
    model = nMIL(bagsize=day, beta=args.beta, gamma=args.gamma, m0=args.m0, p0=args.p0)
    print "Learning for Bag Size: %d" % day
    model.read_data(histIndex=historyMap, docIndex=docMap, feature=args.feature)
    model, perf1, perf2, gsrHistoryProbs = model.SGD()

    timediff = datetime.now() - start
    timecom.append(timediff)
    w1 = open(resultf, 'a')
    w1.write('\t'.join([str(score) for score in perf1]) + '\n')
    w1.close()
    outf1 = "../result/{}_{}_lt-{}_hd-{}_.json".format(args.outfile, args.country, args.leadtime, day)
    w2 = open(outf1, 'wb')
    w2.write(json.dumps(gsrHistoryProbs))
    w2.close()


    print "Running nMIL model for forecasting on country %s, leadtime %d" % (args.country, args.leadtime)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", help="path of data")
    ap.add_argument("-c", "--country", help="country")
    ap.add_argument("--train", help="path of training data")
    ap.add_argument("--test", help="path of testing data")
    ap.add_argument("--resultfile", help="path of result file")
    ap.add_argument("--outfile", help="path of precursor file")
    ap.add_argument("-l", "--leadtime", type=int, default=1, help="k days before GSR to forecast")
    ap.add_argument("-d", "--historyDays", type=int, default=10, help="number of history days to be used for training")
    ap.add_argument("-m0", type=float, default=.5, help="hyper parameter in hinge loss")
    ap.add_argument("-p0", type=float, default=.5, help="hyper parameter in hinge loss")
    ap.add_argument("--gamma", type=float, default=.5, help="parameter in SGD")
    ap.add_argument("--beta", type=float, default=3.0, help="parameter in SGD")
    args = ap.parse_args()
    main(args)

