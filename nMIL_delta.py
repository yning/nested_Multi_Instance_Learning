#!/usr/bin/env python
#-*- coding:utf-8 -*-


__author__ = "Yue Ning, Sathappan Muthiah"
__email__ = "yning@vt.edu, sathap1@vt.edu"
__version__ = "0.0.1"

import sklearn
import sklearn.cross_validation
from scipy.special import expit as scipy_sig
import numpy as np
from numpy import linalg as la
import pdb
import random
import math
import time
from datetime import datetime, timedelta
from dateutil.parser import parse as dateparser
from collections import OrderedDict

def sigmoid(x):
    return 1. / (1. + math.exp(-x))


class nMIL_delta:
    def __init__(self, bagsize, beta=3.0, gamma=0.5, m0=0.5, p0=0.5):
        self.bagsize = bagsize
        self.beta = beta
        self.gamma = gamma
        self.m0 = m0
        self.p0 = p0

    def grad_func(self, X, Y, orgX, w, beta, gamma, m0, p0):

        first_matrix = []
        second_matrix = []
        third_matrix = []

        for i, superbag in enumerate(X):
            P_i_list = []
            combinedbag_loss = []
            combinedDoc_loss = []
            for j, bag in enumerate(superbag):
                bag_dotp = np.dot(bag, w[:, np.newaxis])
                p_ij_list = scipy_sig(bag_dotp)
                hinge_loss = np.sign(p_ij_list - p0)
                P_i_list.append(np.mean(p_ij_list))
                bagloss = np.dot((p_ij_list*(1 - p_ij_list)).T, bag) / len(bag)
                combinedbag_loss.append(bagloss)
                mask = ((hinge_loss * bag_dotp) < m0)
                combinedDoc_loss.append(np.dot((hinge_loss * mask).T, bag)
                                            / len(bag))

            combinedbag_loss = np.concatenate(combinedbag_loss, axis=0)
            combinedDoc_loss = np.concatenate(combinedDoc_loss, axis=0)
            P_i = np.mean(P_i_list)
            neg_loglh = (Y[i] - P_i) / (P_i * (1. - P_i))

            superbagloss = neg_loglh * combinedbag_loss
            first_matrix.append(np.sum(superbagloss, axis=0))

            diffbag = np.concatenate([np.zeros((1,self.x_dimension)), combinedbag_loss], axis=0)
            crossbag_loss = np.diff(diffbag, axis=0)
            if len(superbag) == 1:
                cross_cosine_combined = 1
            else:
                cross_cosine_combined = np.array([0.] + [(orgX[i][j]['cross_cosine'] /
                                                  (len(X[i][j]) * len(X[i][j - 1])))
                                                  for j in
                                                  range(1, len(superbag))])[:, np.newaxis]

            crossbag_derivative = 2*((np.diff([0]+ P_i_list)[:, np.newaxis] *
                                     crossbag_loss * cross_cosine_combined))
            crossbag_derivative = (crossbag_derivative / len(X[i])).sum(axis=0)
            second_matrix.append(crossbag_derivative)
            third_matrix.append(combinedDoc_loss.sum(axis=0))

        first_sum = np.sum(np.array(first_matrix), axis=0) * beta
        second_sum = np.sum(np.array(second_matrix), axis=0) * gamma
        third_sum = np.sum(np.array(third_matrix), axis=0) #* gamma
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

    def SGD(self, leadtime):

        train_X = np.array(self._trainX)
        train_Y = np.array(self._trainY)
        print "X: {}".format(train_X.shape)
        print "Y: {}".format(train_Y.shape)
        OriginalTrain_X = self.original_trainX

        test_X = np.array(self._testX)
        test_Y = np.array(self._testY)

        lambd = 0.05
        iteration = 2000
        n_sgd = 10
        test_f1_arr = []
        test_recall_arr = []
        test_prec_arr = []
        test_roc_arr = []
        test_acc_arr = []
        test_score = []
        train_score = []
        print "using params:\n gamma:{}, beta:{}, m0:{}, p0:{}".format(self.gamma, self.beta,
                                                                       self.m0, self.p0)
        for expr in range(n_sgd):
            w = np.random.rand(self.x_dimension)
            for t in range(iteration):
                eta = 1./((t + 1) * lambd)

                kset = random.sample(range(0, len(train_X) - 1), 10)
                X = train_X[kset]
                Y = train_Y[kset]
                orgX = [OriginalTrain_X[k] for k in kset]
                delta_w = self.grad_func(X, Y, orgX, w, self.beta,
                                    self.gamma, self.m0, self.p0)

                new_w = np.dot((1 - eta * lambd), w) - eta * delta_w / len(X)

                rate = (1./np.sqrt(lambd)) * (1./la.norm(new_w))
                if rate < 1.:
                    w = np.dot(rate, new_w)
                else:
                    w = new_w

            pred_Y = []
            pred_probs = []
            predicted_data = []
            gsr_history_probs = {}
            for idx, testx in enumerate(test_X):
                p_ij_list = []
                days_list = [np.mean(scipy_sig(np.dot(bag, w[:, np.newaxis])))
                             for j, bag in enumerate(testx)]

                P_i = np.mean(days_list)
                pred_probs.append(P_i)
                if P_i > 0.5:
                    pred_Y.append(1)
                else:
                    pred_Y.append(0)

                bag = self.original_testX[idx]
                bag_gsrId = self._testZ[idx]['Id']
                gsr_history_probs[bag_gsrId] = {'trueY': test_Y[idx]}
                for idx, histday in enumerate(bag):
                    doc_vec = histday['docvec']
                    doc_ids = histday['docids']
                    today_probs = [sigmoid(np.dot(w.T, doc)) for k, doc
                                   in enumerate(doc_vec)]
                    a = zip(doc_ids, today_probs)
                    gsr_history_probs[bag_gsrId][idx] = a

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
        
        return self, test_score, train_score, gsr_history_probs, test_Y, pred_probs


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

    start  = time.time()
    model = nMIL_delta(day, beta=args.beta, gamma=args.gamma, m0=args.m0, p0=args.p0)
    print "Learning for Bag Size: %d" % day
    model.read_data(trainIndex=trainMap, testIndex=testMap, docIndex=docMap)

    model, perf1, perf2, gsrHistoryProbs, test_Y, pred_probs = model.SGD(args.leadtime)
    w1 = open(resultf, 'a')
    w1.write('\t'.join([str(score) for score in perf1]) + '\n')
    w1.close()
    outf1 = "../result/{}_{}_lt-{}_hd-{}_.json".format(args.outfile, args.country, args.leadtime, day)
    w2 = open(outf1, 'wb')
    w2.write(json.dumps(gsrHistoryProbs))
    w2.close()

    print "Running nMIL delta model for forecasting on country %s, leadtime %d" % (args.country, args.leadtime)
    print "run-time:{}s".format(time.time() - start)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", help="path of data")
    ap.add_argument("-c", "--country", help="country")
    ap.add_argument("--train", help="path of training data")
    ap.add_argument("--test", help="path of testing data")
    ap.add_argument("--resultfile", help="path of result file")
    ap.add_argument("--outfile", help="path of precursor file")
    ap.add_argument("-l", "--leadtime", type=int, default=1, help="k days before events to forecast")
    ap.add_argument("-d", "--historyDays", type=int, default=10, help="number of history days to be used for training")
    ap.add_argument("-m0", type=float, default=.5, help="hyper parameter in hinge loss")
    ap.add_argument("-p0", type=float, default=.5, help="hyper parameter in hinge loss")
    ap.add_argument("--gamma", type=float, default=.5, help="parameter in SGD")
    ap.add_argument("--beta", type=float, default=3.0, help="parameter in SGD")
    args = ap.parse_args()
    main(args)
