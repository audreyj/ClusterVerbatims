# author: audreyjchang

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.utils.extmath import density
from sklearn import metrics
import os
import csv
import pickle
from collections import Counter
import numpy as np
from time import time
import string
from sklearn.externals import joblib


psuedo_categories = {'null response': ['no primary category', 'no feedback', 'neutral', 'non-english', 'nps prompt',
                                       '"no" recommendation', 'general displeasure'],
                     'pc compete': ['pc compete'],
                     'ps4': ['ps4'],
                     'xbox 360': ['xbox 360', '360'],
                     'enforcement': ['enforcement', 'cheat'],
                     'support': ['support'],
                     'cost/value': ['cost/gold value', 'cost', 'gold', 'purchase', 'store', 'retail'],
                     'reliability': ['reliability', 'service reliability'],
                     'performance': ['performance'],
                     'network/connection': ['connection', 'network', 'offline'],
                     'hardware': ['hardware', 'mic', 'keyboard', 'storage', 'headset', 'sound quality', 'vr', 'kinect'],
                     'controller': ['controller', ' controller'],
                     'UI/UX': ['ui/ux', 'snap', 'installation/download', 'delete', 'download/installation', 'ads',
                               'settings', 'home', 'apps', 'updates', 'update', 'notification', 'advertisements'],
                     'mixer/social': ['mixer', 'beam', 'streaming', 'bitstream', 'broadcast', 'twitch', 'friends',
                                      'gamedvr', 'social experience', 'party', 'messaging', 'activity feed'],
                     'games': ['games', 'game catalog', 'backwards compatibility', 'fps', 'preview'],
                     'account/profile': ['accounts', 'profile', 'privacy', 'security', 'safety', 'sign-in']}

file_loc = 'data/Compiled Verbatims.csv'
file_out = 'data/ginas_verbatims_dict.pkl'
if not os.path.exists(file_out):
    file_count = 0
    verbatim_list = []
    category_list = []
    category_counter = Counter()
    month_counter = Counter()
    null_month_counter = Counter()
    outfile = open('ginas_verbatims.txt', 'w', encoding='utf-8')
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
            clean_one = p[3].lower()
            clean_two = ''.join([l for l in clean_one if l not in string.punctuation])
            verbatim_list.append({'month': p[0], 'rating': p[1], 'category': p[2],
                                  'raw verbatim': p[3], 'pcat': mapped_cat, 'verbatim': clean_two})

            outfile.write(p[3] + '\n')
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

x_list = []
y_list = []
for v_dict in verbatim_list:
    x_list.append(v_dict['verbatim'])
    y_list.append(v_dict['pcat'])
X_train, X_test, y_train, y_test = train_test_split(x_list, y_list, test_size=0.1, random_state=12)
vectorizer = TfidfVectorizer(ngram_range=(1, 4), stop_words='english')
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
joblib.dump(vectorizer, 'data/nps_vectorizer.pkl')
print("train samples: %d, train features: %d" % X_train.shape)
feature_names = vectorizer.get_feature_names()
feature_names = np.asarray(feature_names)

# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)
    joblib.dump(clf, 'data/nps_model_file.pkl')

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        print("top 20 keywords per class:")
        for i, label in enumerate(psuedo_categories.keys()):
            top10 = np.argsort(clf.coef_[i])[-20:]
            print("%s: %s" % (label, ", ".join(feature_names[top10])))

    print("classification report:")
    print(metrics.classification_report(y_test, pred))

    # print("confusion matrix:")
    # print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


benchmark(LinearSVC(penalty="l1", dual=False, tol=1e-3))


