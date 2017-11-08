import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import csv
from nltk.corpus import words
from nltk.corpus import stopwords
import gensim
import random


def semantics(input_string, sid=''):
    if not sid:
        sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(input_string)
    return ss['compound']


def clean_text(input_string):
    output_one = input_string.replace("-", " ")
    output_two = ''.join([l for l in output_one if l not in string.punctuation])
    output_three = output_two.lower()
    return output_three


def gibberish_check(input_list, wordset):
    if any([l in wordset for l in input_list]):
        return 0
    else:
        return 1


def tokenize(input_list, stopset, bigram, trigram, stemmer):
    step_one = [z for z in input_list if z not in stopset]
    step_two = trigram[bigram[step_one]]
    step_three = [stemmer.stem(w) for w in step_two]
    return step_three


def load_data(file_name):
    verbatim_list = []
    print("Loading from file: ", file_name)
    total_verbatim_count = 0
    removed_for_gibberish = []
    sid = SentimentIntensityAnalyzer()
    wordset = set(words.words())
    stopset = set(stopwords.words('english'))
    bigram = gensim.models.phrases.Phraser.load('bigram_verbatims.pkl')
    trigram = gensim.models.phrases.Phraser.load('trigram_verbatim.pkl')
    stemmer = gensim.parsing.porter.PorterStemmer()
    with open(file_name, encoding='utf-8') as f:
        for line in f:
            total_verbatim_count += 1
            raw = line.split('\n')[0]
            verbatim_list.append({'raw': raw})
            verbatim_list[-1]['semantics'] = semantics(raw, sid)
            cleaned = clean_text(raw).split()
            verbatim_list[-1]['list'] = cleaned
            if gibberish_check(cleaned, wordset):
                removed_for_gibberish.append(raw)
                del verbatim_list[-1]
                continue
            verbatim_list[-1]['tokens'] = tokenize(cleaned, stopset, bigram, trigram, stemmer)
    print("Total Verbatims in file: ", str(total_verbatim_count))
    print("Removed by Gibberish Filter: ", str(len(removed_for_gibberish)))

    return verbatim_list


verbatims = load_data('siuf_list.txt')
texts = [v['tokens'] for v in verbatims]
dictionary = gensim.corpora.Dictionary(texts)
print(dictionary)
corp = [dictionary.doc2bow(text) for text in texts]
lsi = gensim.models.lsimodel.LsiModel(corpus=corp, id2word=dictionary, num_topics=10)
print(lsi.show_topics())
# doc = 'increase compatibility with older games'
# vec_bow = dictionary.doc2bow(doc.split())
vec_lsi = lsi[random.choice(corp)]
index = gensim.similarities.MatrixSimilarity(lsi[corp])
sims = index[vec_lsi]
sims = sorted(enumerate(sims), key=lambda item: -item[1])
for s in sims:
    print(s, verbatims[s[0]]['raw'])
