########################################################
# Final Project for TAMU CSCE670 
# Project Name: SmartCalendar
# Purpose: tf-idf; vectorization; SVM classifier; 
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

class classifier:
    def __init__(self):
        self.inFile = ''
        self.tf_index = defaultdict(list)
        self.df_index = defaultdict(int)
        self.total_terms = []
        self.doc_words = defaultdict(list)  #doc_words[docID] = [term1, term2, term3,...] easy to find what terms in a doc
        self.doc_vectors = defaultdict(array) #??? normalize or not???? array or list????
        
        #a list with each documents, text[0] = {'title': title, 'contents': description}
        #the train_text and test_text will be assigned from the data parsing step
        self.train_text = []
        self.test_text = []
        #a classification vector for the training docs;
        #it will be given by the data parsing step
        self.train_vec = [] 
        
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
            text = doc['title'] + doc['content']
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
        for docID in self.doc_words:
            vector = []
            for term in self.total_terms:
                if term in self.doc_words[docID]:
                    vector.append(self.tf_idf(self.tfIndex[term][docID], self.dfIndex[term], self.totalDocs))
                else:
                    vector.append(0)
            mag = 0
            for i in vector:
                mag += math.pow(i, 2)
            mag = sqrt(mag)
            self.doc_vector[docID] = array(vector)/mag   #does sklearn svm accept numpy array? or only list??? ---make sure!!!
    def svm_train(self):
        """use svm in sklearning to train the classifier"""
        pass
        
    def svm_test_one_doc(self):
        """given a test document, parse it for tf and idf, classify it with svm and return its class"""
        pass
    def svm_test(self):
        """given a vector for a document; classify it as "food, seminar or movie"""
        pass
def main():
    c1 = classifier()
    c1.test_func()
    
if __name__ == '__main__':
    main()
