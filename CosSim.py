# author: audreyjchang
# date: May 2, 2017

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pickle
from collections import Counter
import numpy as np
from nltk.corpus import stopwords
import numpy.linalg as LA
import re, math
import random
import math


def Cosine(vec1, vec2):
    result = InnerProduct(vec1, vec2) / (VectorSize(vec1) * VectorSize(vec2))
    return result


def VectorSize(vec):
    return math.sqrt(sum(math.pow(v, 2) for v in vec))


def InnerProduct(vec1, vec2):
    return sum(v1*v2 for v1,v2 in zip(vec1,vec2))


def Euclidean(vec1, vec2):
    return math.sqrt(sum(math.pow((v1-v2),2) for v1,v2 in zip(vec1, vec2)))


def Theta(vec1, vec2):
    return math.acos(Cosine(vec1,vec2)) + 10


def Triangle(vec1, vec2):
    theta = math.radians(Theta(vec1,vec2))
    return (VectorSize(vec1) * VectorSize(vec2) * math.sin(theta)) / 2


def Magnitude_Difference(vec1, vec2):
    return abs(VectorSize(vec1) - VectorSize(vec2))


def Sector(vec1, vec2):
    ED = Euclidean(vec1, vec2)
    MD = Magnitude_Difference(vec1, vec2)
    theta = Theta(vec1, vec2)
    return math.pi * math.pow((ED+MD),2) * theta/360


def TS_SS(vec1, vec2):
    return Triangle(vec1, vec2) * Sector(vec1, vec2)


file_loc = 'siuf_negative.txt'
file_out = 'blahblah.pkl'
if not os.path.exists(file_out):
    verbatim_list = []
    with open(file_loc, encoding='utf-8') as f:
        for line in f:
            verbatim_list.append(line)
    print('number of articles: ', len(verbatim_list))
    # pickle.dump(verbatim_list, open(file_out, 'wb'))
    # print('pickle dumped to: ', file_out)
else:
    print('opening: ', file_out)
    verbatim_list = pickle.load(open(file_out, 'rb'))

outfile = open('output.tsv', 'w', encoding='utf-8')

# vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer(max_features=100)
#
trainVectorizerArray = vectorizer.fit_transform(verbatim_list).toarray()

output_counter = Counter()
output_dict = {}
used_list = []
for e, vector in enumerate(trainVectorizerArray):
    match_list = []
    for f, testV in enumerate(trainVectorizerArray):
        if f in used_list or f == e:
            continue
        similarity = TS_SS(vector, testV)
        if similarity > 0.2:
            match_list.append(f)
        print(verbatim_list[f], ' <==> ', verbatim_list[e], ' || ', similarity)
    # print(len(match_list), (float(len(verbatim_list))*0.02))
    if float(len(match_list)) > (float(len(verbatim_list))*0.01):
        output_counter[e] = len(match_list)
        output_dict[e] = match_list
        used_list.extend(match_list)
        # print("Matching: ", verbatim_list[e])
        # print("length text list: ", len(verbatim_list))
        # for g in match_list:
        #    print(verbatim_list[g])
        # input('------------------  ' + str(len(match_list)) + '  -------------')

already_matched_index = []
output_length = 0
for z in output_counter.most_common():
    v_index = z[0]
    if v_index in already_matched_index:
        continue
    print('(' + str(z[1]) + ')' + verbatim_list[z[0]])
    outfile.write('(' + str(z[1]) + ')' + verbatim_list[z[0]])
    already_matched_index.append(z[0])
    output_length += 1
    for i in output_dict[v_index]:
        already_matched_index.append(i)
        outfile.write('  --  ' + verbatim_list[i])
        print('  --  ' + verbatim_list[i])

print('article number of verbatims: ' + str(len(verbatim_list)) + ' ---- num results: ' + str(output_length))
print('matched verbatims: ' + str(len(already_matched_index)) +
      ' --- unmatched: ' + str(len(verbatim_list) - len(already_matched_index)))

for e, c in enumerate(verbatim_list):
    if e not in already_matched_index:
        print(c)
        outfile.write(c)
