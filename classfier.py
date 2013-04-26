########################################################
# Final Project for TAMU CSCE670 
# Project Name: SmartCalendar
# Purpose: tf-idf; vectorization; SVM classifier;
# Group Members: Xixu Cai, Yue Zhuo, Lin Mu
# April 23, 2013 
########################################################
import sys
import os
import re
import json
import array
from collections import defaultdict
from numpy import *
import math
import time
import pickle
import requests # Get from https://github.com/kennethreitz/requests
import string
import random
from sklearn import svm
import pylab as pl
import get_rss

class classifier:
    def __init__(self):
        self.inFile = ''
        self.tf_index = defaultdict(dict)
        self.df_index = defaultdict(int)
        self.total_terms = []
        self.doc_words = defaultdict(list)  #doc_words[docID] = [term1, term2, term3,...] easy to find what terms in a doc
        self.doc_vector = [] #normalized array
        
        #a list with each documents, text[0] = {'title': title, 'contents': description}
        #the train_text and test_text will be assigned from the data parsing step
        self.train_text = []
        self.test_text = get_rss.generate_test_data()
        #a classification vector for the training docs;
        #it will be given by the data parsing step
        self.train_vec = []

        (self.train_text, self.train_vec) = get_rss.generate_training_data()
        print "self.train_vec: ", self.train_vec
        #self.test_vec is initially empty
        self.test_vec = []
        self.total_docs = len(self.train_text) + len(self.test_text)

        
    def test_func(self):
        print "test succeed!"
        
    def getTerms(self, line):
        """helper function; parse and get terms from a string"""
        line = line.lower()
        line = re.split(r'[\W]+', line, 0, re.UNICODE)
        return line
    def tf_idf(self, tf, df, total):
        """given two ints: tf and df, return the tf_idf value"""
        return ((1+log2(tf))*log2(total/float(df)))
    
    def build_tf_idf(self):
        """parse the infile documents; build tf_index and df_index"""
        for doc in self.train_text:
            docID = self.train_text.index(doc)
            #print "build_tf_idf() ",doc
            if doc['Content'] == None:
                text = doc['Title']
            elif doc['Title'] == None:
                text = doc['Content']
            else:
                text = doc['Title'] + doc['Content']
            words = self.getTerms(text)
            for word in words:
                try:
                    self.tf_index[word][docID] += 1
                except:
                    self.tf_index[word][docID] = 1
                    self.df_index[word] += 1
                    if not word in self.total_terms:
                        self.total_terms.append(word)                    
                if not word in self.doc_words[docID]:
                    try:
                        self.doc_words[docID].append(word)
                    except:
                        self.doc_words[docID] = [word]
    def vectorization(self):
        """build vector for each document"""
        for doc in self.train_text:
            vector = []
            docID = self.train_text.index(doc)
            for term in self.total_terms:
                if term in self.doc_words[docID]:
                    vector.append(self.tf_idf(self.tf_index[term][docID], self.df_index[term], self.total_docs))
                else:
                    vector.append(0)
            mag = 0
            for i in vector:
                mag += math.pow(i, 2)
            mag = sqrt(mag)
            self.doc_vector.append( array(vector)/mag )  #does sklearn svm accept numpy array? or only list??? ---make sure!!!
    def svm_train_linear(self):
        """use svm in sklearning to train the classifier; kernel = 'linear'"""
        svc = svm.SVC(kernel = 'linear')
        print "self.doc_vector: type: ", type(self.doc_vector), "length: ", len(self.doc_vector)
        svc.fit(self.doc_vector, self.train_vec)
        return svc
    def svm_train_polynomial(self):
        """use svm in sklearning to train the classifier; kernel = 'linear'"""
        svc = svm.SVC(kernel = 'poly', degree = 3) #degree: polynomial degree
        svc.fit(self.doc_vector, self.train_vec)
        return svc
    def svm_train_rbf(self):
        """use svm in sklearning to train the classifier; kernel = 'linear'"""
        svc = svm.SVC(kernel = 'rbf') #gamma: inverse of size of radial kernel
        svc.fit(self.doc_vector, self.train_vec)
        return svc
    def svm_test_one_doc(self, doc, svc):
        """given a test document, parse it for tf and idf, classify it with svm and return its class"""
        #doc = {'title':'some title', 'content':some contents}
        tf = {}
        vector = []
        if doc['Content'] == None:
            text = doc['Title']
        elif doc['Title'] == None:
            text = doc['Content']
        else:
            text = doc['Title'] + doc['Content']

        words = self.getTerms(text)
        for word in words:
            try:
                tf[word] += 1
            except:
                tf[word] = 1
        for term in self.total_terms:
            if term in words:
                vector.append(self.tf_idf(tf[term], self.df_index[term], self.total_docs))
            else:
                vector.append(0)
        mag = 0
        for i in vector:
            mag += math.pow(i, 2)
        mag = sqrt(mag)
        if mag == 0:
            print doc
        else: 
            vector = array(vector)/mag
        group = svc.predict(vector)
        return group
        
    def svm_test(self, svc):
        """given a vector of document; classify it as "food, seminar or movie"""
        for doc in self.test_text:
            self.test_vec.append(self.svm_test_one_doc(doc,svc))
            
        
def main():
    c1 = classifier()
    c1.build_tf_idf()
    c1.vectorization()
    svc = c1.svm_train_linear()
    c1.svm_test(svc)
    print "the classification results are: "
    print c1.test_vec
    print "type: ", type(c1.test_vec[3]), c1.test_vec[3]
    for i in xrange(len(c1.test_vec)):
        if c1.test_vec[i] == 2:
            print "class 2, movie: ", c1.test_text[i]
        elif c1.test_vec[i] == 1:
            print "class 1: ", c1.test_text[i]
        elif c1.test_vec[i] == 3:
            print "class 3: ", c1.test_text[i]
        elif c1.test_vec[i] == 5:
            print "class 5: ", c1.test_text[i]
    
if __name__ == '__main__':
    main()
