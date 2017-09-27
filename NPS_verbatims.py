# author: audreyjchang
# date: May 2, 2017

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import csv
import pickle
from collections import Counter
import numpy as np
from nltk.corpus import stopwords
import numpy.linalg as LA
import random

psuedo_categories = {'null response': ['no primary category', 'no feedback', 'neutral', 'non-english', 'nps prompt',
                                       '"no" recommendation'],
                     'positive': ['positive', 'positivie', '"yes" recommendation'],
                     'general displeasure': ['general displeasure'],
                     'compete': ['pc compete', 'ps4', 'xbox 360', 'compete', 'scorpio', '360', '4k'],
                     'enforcement': ['enforcement', 'cheat'],
                     'support': ['support'],
                     'cost/value': ['cost/gold value', 'cost', 'gold'],
                     'reliability': ['reliability', 'service reliability'],
                     'network/connection': ['connection', 'network', 'offline'],
                     'hardware': ['hardware', 'controller', ' controller', 'mic', 'keyboard', 'storage',
                                  'headset', 'sound quality', 'vr', 'kinect'],
                     'store': ['purchase', 'store', 'retail'],
                     'UI/UX': ['ui/ux', 'snap', 'installation/download', 'delete', 'download/installation',
                               'settings', 'home', 'apps', 'updates', 'update', 'notification',
                               'ads', 'advertisements'],
                     'mixer': ['mixer', 'beam', 'streaming', 'bitstream', 'broadcast', 'twitch', 'gamedvr'],
                     'performance': ['performance'],
                     'non-game': ['bluray', 'achievements', 'livetv', 'cortana', 'music'],
                     'social': ['social experience', 'party', 'messaging', 'friends', 'activity feed'],
                     'games': ['games', 'game catalog', 'backwards compatibility', 'fps', 'preview'],
                     'account/profile': ['accounts', 'profile', 'privacy', 'security', 'safety', 'sign-in']}

file_loc = 'Compiled Verbatims.csv'
file_out = 'ginas_verbatims.pkl'
if not os.path.exists(file_out):
    file_count = 0
    verbatim_list = []
    category_list = []
    category_counter = Counter()
    month_counter = Counter()
    null_month_counter = Counter()
    with open(file_loc, encoding='utf-8') as f:
        reader = csv.reader(f)
        for p in reader:
            file_count += 1
            if file_count == 1:
                print(p)
                continue

            category_list.append(p[2])
            mapped_cat = [f for f, k in psuedo_categories.items() if p[2].lower() in k]
            if len(mapped_cat) == 1:
                mapped_cat = mapped_cat[0]
            elif len(mapped_cat) == 0:
                mapped_cat = 'null response'
                print(p[2])
            else:
                print(mapped_cat)
            if mapped_cat == 'null response':
                null_month_counter[p[0]] += 1
            month_counter[p[0]] += 1
            category_counter[mapped_cat] += 1
            category_list.append(mapped_cat)
            verbatim_list.append({'month': p[0], 'rating': p[1], 'category': p[2],
                                  'verbatim': p[3], 'pcat': mapped_cat})
    print('file length: ', file_count)
    print('number of articles: ', len(verbatim_list))
    print(null_month_counter)
    print(month_counter)
    print(category_counter)
    print(verbatim_list[1436])
    print(sum(category_counter.values()))
    pickle.dump(verbatim_list, open(file_out, 'wb'))
    print('pickle dumped to: ', file_out)
else:
    print('opening: ', file_out)
    verbatim_list = pickle.load(open(file_out, 'rb'))
