# author: audreyjchang

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import string
import os
import csv
import pickle
from collections import Counter
import xlsxwriter

text_file = 'data/SeptVerbatimTest.txt'

# workbook = xlsxwriter.Workbook('/data/nps_test.xlsx')
# worksheet = workbook.add_worksheet()

clf = joblib.load('data/nps_model_file.pkl')
vectorizer = joblib.load('data/nps_vectorizer.pkl')

total_verbatim_count = 0
with open(text_file, encoding='utf-8') as f:
    for line in f:
        total_verbatim_count += 1
        verbatim_one = line.split('\n')[0]
        verbatim_two = verbatim_one.lower()
        verbatim = ''.join([l for l in verbatim_two if l not in string.punctuation])
        x_test = vectorizer.transform([verbatim])
        prediction = clf.predict(x_test)
        print(verbatim, prediction)
