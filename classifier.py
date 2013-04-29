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
#import requests # Get from https://github.com/kennethreitz/requests
import string
import random
from sklearn import svm
import pylab as pl
import get_rss

class classifier:
    def __init__(self):
        
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
        self.class_df = defaultdict(list)
        self.chi_score = defaultdict(dict)

        (self.train_text, self.train_vec) = get_rss.generate_training_data()
        print "self.train_vec: ", self.train_vec
        #self.test_vec is initially empty
        self.test_vec = []
        self.test_food_vec = []
        self.test_movie_vec = []
        self.total_docs = len(self.train_text) + len(self.test_text)

        self.documents = []
        self.new_docs = self.feature_selection()

        self.food_results = []
        self.movie_results = []
      
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
        i = 0
        for doc in self.train_text:
            #docID = self.train_text.index(doc)
            docID = i
            i += 1
            #print "build_tf_idf() ",doc
            if doc['Content'] == None:
                text = doc['Title']
            elif doc['Title'] == None:
                text = doc['Content']
            else:
                text = doc['Title'] + doc['Content']
            words = self.getTerms(text)
            self.documents.append(words)
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

    def build_tf_idf_feature_selection(self):
        """parse the infile documents; build tf_index and df_index"""
        i = 0
        for doc in self.new_docs:
            #docID = self.train_text.index(doc)
            docID = i
            i += 1
            
            words = doc
            #self.documents.append(words)
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
        i = 0
        for doc in self.train_text:
            vector = []
            docID = i
            i += 1
            for term in self.total_terms:
                if term in self.doc_words[docID]:
                    vector.append(self.tf_idf(self.tf_index[term][docID], self.df_index[term], self.total_docs))
                else:
                    vector.append(0)
            mag = 0
            for m in vector:
                mag += math.pow(m, 2)
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
            pass
            #print doc
        else: 
            vector = array(vector)/mag
        group = svc.predict(vector)
        return group
        
    def svm_test_food(self, svc):
        """given a vector of document; classify it as "food, seminar or movie"""
        for doc in self.test_text:
            self.test_food_vec.append(self.svm_test_one_doc(doc,svc))
        for i in xrange(len(self.test_food_vec)):
            if self.test_food_vec[i] == 1:
                self.food_results.append(self.test_text[i])
    def svm_test_movie(self, svc):
        """given a vector of document; classify it as "food, seminar or movie"""
        for doc in self.test_text:
            self.test_movie_vec.append(self.svm_test_one_doc(doc,svc))
        #print "test_movie_vec:", self.test_movie_vec
        for i in xrange(len(self.test_movie_vec)):
            if self.test_movie_vec[i] == 2:
                self.movie_results.append(self.test_text[i])            
    def svm_train_food(self):
        y = []
        for i in self.train_vec:
            if i == 1:
                y.append(i)
            else:
                y.append(0)
        svc = svm.SVC(kernel = 'linear')
        #print "self.doc_vector: type: ", type(self.doc_vector), "length: ", len(self.doc_vector)
        svc.fit(self.doc_vector, y)
        return svc
    def svm_train_movie(self):
        y = []
        for i in self.train_vec:
            if i == 2:
                y.append(i)
            else:
                y.append(0)
        svc = svm.SVC(kernel = 'linear')
        #print "self.doc_vector: type: ", type(self.doc_vector), "length: ", len(self.doc_vector)
        svc.fit(self.doc_vector, y)
        return svc
    def extract_terms(self, data):
        ''' (list) -> dict
 
        Return a dictionary invert_term consisted of {terms: [term_id, DF]}
        '''
        invert_table = []
        for i in range(len(data)):
            ''' [term, document_id] '''
            tmp_dict = {term: i for term in data[i]}
            for key in tmp_dict:
                invert_table.append((key, tmp_dict[key]))
        ''' sort by term '''
        invert_table = sorted(invert_table, key = lambda x : x[0])
        cur_word = invert_table[0][0]
        invert_terms = {cur_word: [0, 1]}
        count = 1
        for i in range(1, len(invert_table)):
            if invert_table[i][0] == cur_word:
                ''' add document frequency '''
                invert_terms[cur_word][1] += 1
            else:
                ''' new term '''
                cur_word = invert_table[i][0]
                invert_terms[cur_word] = [count, 1]
                count += 1
        return invert_terms

    def feature_selection(self):
        ''' (list, list, int) -> list
 
        Return list of documents of selected features. The feature selection is achieved
        by computing the mutual information
        '''
        
        i = 0
        for doc in self.train_text:
            #docID = self.train_text.index(doc)
            docID = i
            i += 1
            #print "build_tf_idf() ",doc
            if doc['Content'] == None:
                text = doc['Title']
            elif doc['Title'] == None:
                text = doc['Content']
            else:
                text = doc['Title'] + doc['Content']
            words = self.getTerms(text)
            self.documents.append(words)
        documents = self.documents
                   
        classification = self.train_vec
        print "len of documents: ", len(documents), "len of classification: ", len(classification)
        num_classes = len(unique(classification))
        invert_terms = self.extract_terms(documents)    
        N = len(documents) + 4
        for i in invert_terms:
            invert_terms[i] = []
        max_score = []
        for i in range(num_classes):
            max_score.append(0)
            for j in invert_terms:
                ''' add one smooth '''
                N10 = 1
                N11 = 1
                N01 = 1
                N00 = 1
                for k in range(len(documents)):
                    if j in documents[k] and classification[k] != i:
                        N10 += 1
                    elif j in documents[k] and classification[k] == i:
                        N11 += 1
                    elif j not in documents[k] and classification[k] == i:
                        N01 += 1
                    elif j not in documents[k] and classification[k] != i:
                        N00 += 1            
            
                N1_ = N10 + N11 * 1.0
                N_1 = N11 + N01 * 1.0
                N0_ = N01 + N00 * 1.0
                N_0 = N10 + N00 * 1.0            
                invert_terms[j].append(N11 * 1.0 / N * math.log(N * N11 / N1_ / N_1, 2) + \
                            N01 * 1.0 / N * math.log(N * N01 / N0_ / N_1, 2) + \
                            N10 * 1.0 / N * math.log(N * N10 / N1_ / N_0, 2) + \
                            N00 * 1.0 / N * math.log(N * N00 / N0_ / N_0, 2))
                max_score[i] = max(max_score[i], invert_terms[j][i])
            print sorted(invert_terms, key = lambda x : invert_terms[x][i], reverse = True)[:10]
            ''' print selected_feature '''
        new_documents = []
        cnt = 0
        for i in range(len(documents)):
            new_documents.append([])
            cls = classification[i]
            for j in documents[i]:
                #print "j: ", j, "cls: ", cls
                if invert_terms[j][cls] >= max_score[cls] * 0.45:                
                    new_documents[i].append(j)
            if new_documents[i] == []:
                cnt += 1
                new_documents[i].append('for_eliminating_null')
            
        return new_documents
    def find_food(self):
        """find food in class 0"""
        for i in xrange(len(self.documents)):
            doc = self.documents[i]
            if self.train_vec[i] == 0:
                if 'glasscock' in doc:
                    print i, doc
                elif 'refreshments' in doc:
                    print i, doc

    def calculate_df_in_classes(self):
        """calculate how many documents a term appear in each class"""
        for doc in self.doc_words:
            cls = self.train_vec[doc]
            #print doc, self.doc_words[doc]
            for word in self.doc_words[doc]:
                if word in self.class_df:
                    self.class_df[word][cls] += 1
                else:
                    for i in xrange(len(unique(self.train_vec))):
                        if i == cls:
                            self.class_df[word].append(1)
                        else:
                            self.class_df[word].append(0)
    
    def chi_square(self):
        """ compute chi square scores for each term in each class"""
        num_docs_in_class = []
        for i in xrange(len(unique(self.train_vec))):
            num_docs_in_class.append(self.train_vec.count(i))
        for doc in self.doc_words:
            cls = self.train_vec[doc]
            for term in self.doc_words[doc]:
                if not term in self.chi_score[cls]:
                    n11 = float(self.class_df[term][0])
                    n10 = float(self.class_df[term][1] + self.class_df[term][2])
                    n01 = float(num_docs_in_class[cls] - n11)
                    n00 = 0.0
                    for i in xrange(len(unique(self.train_vec))):
                        if i != cls:
                            n00 += num_docs_in_class[i]
                    
                    a = n11 + n10 + n01 + n00
                    b = math.pow(((n11 * n00) - (n10 * n01)), 2)
                    c = (n11 + n01) * (n11 + n10) * (n10 + n00) * (n01 + n00)
                    chi = (a * b) / c
                    self.chi_score[cls][term] = chi

    def chi_feature_list(self, k):
        """return the feature list with top k terms in each classes; total <= 3*k"""
        #self.chi_score[cls], self.score['bus'], self.score['pol']; self.features = []
        num_classes = len(unique(self.train_vec))
        features = [[],[],[]]
        for i in xrange(num_classes):
            count = 0
            for key,value in sorted(self.chi_score[i].iteritems(), key=lambda (k,v): (v,k)):
                
                if count < k:
                    features[i].append(key)
                count += 1
       
        #print "length of feature_list:", len(self.features)
        print "number of features: ", k
        for i in xrange(num_classes):
            print i, ":",features[i]
        return features

    def write_results(self):
        f1 = open('food.txt','wb')
        pickle.dump(self.food_results, f1)
        f1.close()

        f2 = open('movie.txt', 'wb')
        pickle.dump(self.movie_results, f2)
        f2.close()
def main():
    c1 = classifier()
    #c1.build_tf_idf()
    #c1.feature_selection()
    c1.build_tf_idf_feature_selection()
    c1.vectorization()
#    c1.calculate_df_in_classes()
#    c1.chi_square()
#    c1.chi_feature_list(50)

    svc1 = c1.svm_train_food()
    c1.svm_test_food(svc1)
    print "============the classification results for food are:=================== "
    
    for i in c1.food_results:
        print i['Title'],"\n", i['Content'],"\n"

    
    svc2 = c1.svm_train_movie()
    c1.svm_test_movie(svc2)
    print "=============the classification results for movie are: ==============="
    for i in c1.movie_results:
        print i['Title'],"\n", i['Content'],"\n"

    c1.write_results()
    
    print "type of self.food_resutls: ", type(c1.food_results), "len:", len(c1.food_results)
    print "type of self.food_results[0]: ", type(c1.food_results[0]), "len:", len(c1.food_results[0])
    print c1.food_results[0]
    print "type of self.movie_resutls: ", type(c1.movie_results), "len:", len(c1.movie_results)
    print "type of self.movie_results[0]: ", type(c1.movie_results[0]), "len:", len(c1.movie_results[0])
    print c1.movie_results[0]
if __name__ == '__main__':
    main()
