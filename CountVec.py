from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import string


def load_data(data_in, semantics_switch=True):
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
    for s in raw_verbatim_list:
        r = ''.join(l for l in s if l not in string.punctuation)
        r = r.lower()
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


verbatim_file = 'siuf_list.txt'
data_samples = load_data(verbatim_file, False)

tf_vectorizer = TfidfVectorizer(ngram_range=(1, 6), max_features=1000, stop_words='english')
tf = tf_vectorizer.fit_transform(data_samples['All Verbatims'])

# print(tf_vectorizer.get_feature_names())
# print(tf_vectorizer.get_stop_words())

try_me = tf_vectorizer.build_analyzer()
print(try_me('this is a test of the tokenization system'))
