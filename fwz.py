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
from fuzzywuzzy import fuzz


file_loc = 'siuf_negative.txt'
verbatim_list = []
with open(file_loc, encoding='utf-8') as f:
    for line in f:
        verbatim_list.append(line)
print('number of articles: ', len(verbatim_list))

outfile = open('output.tsv', 'w', encoding='utf-8')

output_counter = Counter()
output_dict = {}
used_list = []
for e, vector in enumerate(verbatim_list):
    match_list = []
    for f, testV in enumerate(verbatim_list):
        if f in used_list or f == e:
            continue
        match_number = fuzz.token_set_ratio(vector, testV)
        if match_number > 70:
            match_list.append(f)
    # print(len(match_list), (float(len(verbatim_list))*0.02))
    if float(len(match_list)) > (float(len(verbatim_list))*0.02):
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