from time import time
import pickle
import sys
import random

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def print_top_words(model, feature_names, n_top_words=20):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % (topic_idx+1)
        message += ", ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


def load_data(data_in, semantics_switch):
    if semantics_switch:
        data_out = {'Positive': [], 'Neutral': [], 'Negative': []}
        sid = SentimentIntensityAnalyzer()
    else:
        data_out = {'All Verbatims': []}
    raw_verbatim_list = []
    if isinstance(data_in, str):
        print("Loading from file: ", data_in)
        total_verbatim_count = 0
        with open(data_in, encoding='utf-8') as f:
            for line in f:
                total_verbatim_count += 1
                raw_verbatim_list.append(line.split('\n')[0])
        print("Total Verbatims in file: ", str(total_verbatim_count))
    elif isinstance(data_in, list):
        print("Data input as list with length: ", str(len(data_in)))
        for d in data_in:
            if isinstance(d, str):
                raw_verbatim_list.append(d.split('\n')[0])
            else:
                print("list instance not a string")
    for r in raw_verbatim_list:
        if semantics_switch:
            ss = sid.polarity_scores(r)
            if ss['compound'] > 0.3:
                data_out['Positive'].append(r)
            elif ss['compound'] > -0.3:
                data_out['Neutral'].append(r)
            else:
                data_out['Negative'].append(r)
        else:
            data_out['All Verbatims'].append(r)

    return data_out


def lda(verbatim_file='', semantics_on=True, number_topics=4, print_option=True):
    if not verbatim_file:
        print("No verbatim file specified, using example verbatim file: siuf_list.txt")
        print("Semantics option is ON, number of topics set to: 4, print option is ON")
        verbatim_file = 'siuf_list.txt'

    data_samples = load_data(verbatim_file, semantics_on)

    fileout = open('siuf_negative.txt', 'w', encoding='utf-8')

    for v_type, v_list in data_samples.items():
        print('===============================================')
        n_samples = len(v_list)
        print("%s Verbatims (%d)" % (v_type, n_samples))

        print("Extracting tf features for LDA...")
        tf_vectorizer = TfidfVectorizer(ngram_range=(1, 4), stop_words='english')
        t0 = time()
        tf = tf_vectorizer.fit_transform(v_list)
        print("done in %0.3fs." % (time() - t0))
        print()

        max_score = (0, -1000000.0)

        print("Fitting LDA model, n_samples=%d and n_topics=%d..." % (n_samples, number_topics))
        lda = LatentDirichletAllocation(max_iter=2500, learning_offset=50., random_state=0, n_topics=number_topics)
        t0 = time()
        lda.fit(tf)
        sc = lda.score(tf)
        print("done in %0.3fs." % (time() - t0))
        # print("topics = %d, score = %f" % (number_topics, sc))
        if sc > max_score[1]:
            max_score = (number_topics, sc)
        print("\nTopics in LDA model:")
        tf_feature_names = tf_vectorizer.get_feature_names()
        print_top_words(lda, tf_feature_names, 10)
        print("5 Randomly Selected Verbatims from: %s" % v_type)
        print("   ----------------------------")
        for r in range(5):
            print(random.choice(v_list))

        if print_option and v_type == 'Negative':
            for v in v_list:
                fileout.write(v + '\n')

if __name__ == "__main__":
    lda(sys.argv[1:])
