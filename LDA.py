from time import time
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from nltk.sentiment.vader import SentimentIntensityAnalyzer

n_top_words = 20
topics = range(2, 12, 2)


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += ", ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

print("Loading dataset...")
t0 = time()
# data_samples = pickle.load(open('verbatims.pkl', 'rb'))
data_samples_positive = []
data_samples_neutral = []
data_samples_negative = []
sid = SentimentIntensityAnalyzer()
with open('SeptVerbatimTest.txt', encoding='utf-8') as f:
    for line in f:
        ss = sid.polarity_scores(line)
        if ss['compound'] > 0.3:
            data_samples_positive.append(line)
        elif ss['compound'] > -0.3:
            data_samples_neutral.append(line)
        else:
            data_samples_negative.append(line)

for data_samples in [data_samples_positive, data_samples_neutral, data_samples_negative]:
    print('===============================================')
    n_samples = len(data_samples)
    print("done in %0.3fs." % (time() - t0))

    # Use tf (raw term count) features for LDA.
    print("Extracting tf features for LDA...")
    tf_vectorizer = TfidfVectorizer(ngram_range=(1, 4), stop_words='english')
    t0 = time()
    tf = tf_vectorizer.fit_transform(data_samples)
    print("done in %0.3fs." % (time() - t0))
    print()

    max_score = (0, -1000000.0)
    for t in topics:
        print("Fitting LDA models with tf features, n_samples=%d and n_topics=%d..."
              % (n_samples, t))
        lda = LatentDirichletAllocation(max_iter=2500, learning_method='online', learning_offset=50., random_state=0,
                                        n_topics=t)
        t0 = time()
        lda.fit(tf)
        sc = lda.score(tf)
        print("done in %0.3fs." % (time() - t0))
        print("topics = %d, score = %f" % (t, sc))
        if sc > max_score[1]:
            max_score = (t, sc)
        print("\nTopics in LDA model:")
        tf_feature_names = tf_vectorizer.get_feature_names()
        print_top_words(lda, tf_feature_names, n_top_words)

    # lda = LatentDirichletAllocation(max_iter=2000, learning_method='online', learning_offset=50., random_state=0,
    #                                 n_topics=max_score[0])
    # lda.fit(tf)
    # print("\nTopics in LDA model:")
    # tf_feature_names = tf_vectorizer.get_feature_names()
    # print_top_words(lda, tf_feature_names, n_top_words)