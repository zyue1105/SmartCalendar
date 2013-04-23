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
from sklearning import svm
import pylab as pl

class classifier:
    def __init__(self):
        self.inFile = ''
        self.tf_index = defaultdict(list)
        self.df_index = defaultdict(int)
    def test_func(self):
        print "test succeed!"
        
    def getTerms(self, line):
        """helper function; parse and get terms from a string"""
        line = line.lower()
        line = re.split(r'[\W]+', line, 0, re.UNICODE)
        return line
        
def main():
    c1 = classifier()
    c1.test_func()
    
if __name == '__main__':
    main()
