import gensim
import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import words
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import random
from collections import Counter
import time


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
    remove_list = ['game']
    step_one = [z for z in input_list if z not in stopset]
    step_two = trigram[bigram[step_one]]
    step_three = [stemmer.stem(w) for w in step_two]
    step_four = [l for l in step_three if l not in remove_list]
    return step_four


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
            # verbatim_list[-1]['semantics'] = semantics(raw, sid)
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


def random_test(the_list, test_against, df):
    output_dict = {'test string': test_against, 'match count': 0, 'match list': []}
    same_count = 0
    distance_total = 0
    for teststr in the_list:
        distance = word2vec_model.wmdistance(test_against, teststr)
        if distance < df:
            same_count += 1
            distance_total += df-distance
            output_dict['match list'].append((teststr, distance))
    output_dict['match count'] = same_count
    output_dict['distance metric'] = distance_total
    return output_dict


def get_max(the_list, ind):
    stems = Counter()
    for sub_list in the_list:
        for w in sub_list:
            stems[w] += 1
    print(stems.most_common(20))
    looking_for = stems.most_common(20)[ind][0]
    for sub_list in the_list:
        if looking_for in sub_list:
            return sub_list


def get_list(the_list, num_returned=10):
    tf_vectorizer = TfidfVectorizer(ngram_range=(2, 6), max_features=num_returned, stop_words='english')
    tf_vectorizer.fit_transform(the_list)

    return tf_vectorizer.build_analyzer()


start = time.time()
verbatim_file = 'siuf_list.txt'
data_samples_dicts = load_data(verbatim_file)
data_samples = [r['tokens'] for r in data_samples_dicts]
sample_range = int(len(data_samples) * 0.9)
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)
word2vec_model.init_sims(replace=True)  # normalizes vectors

distance_factor = 0.9
copy_list = data_samples[:]
saved_out = [{'distance metric': 0, 'test string': []} for x in range(10)]

# test_against_list = get_list(copy_list)
test_against_list = copy_list
for e, a in enumerate(test_against_list):
    output_d = random_test(copy_list, a, distance_factor)
    for ind, s in enumerate(saved_out):
        if output_d['distance metric'] > s['distance metric'] \
                and not any(g in s['test string'] for g in output_d['test string']):
            saved_out.insert(ind, output_d)
            del saved_out[-1]
            break

total_matched_verbatims = []
for e, t in enumerate(saved_out):
    print("#%d: (%d, %f) %s" % (e, t['match count'], t['distance metric'], t['test string']))
    for i, v in enumerate(t['match list']):
        raw_words = data_samples_dicts[data_samples.index(v[0])]['raw']
        print('   - ', i, raw_words, v[1])
        total_matched_verbatims.append(raw_words)

total_matched_verbatims = set(total_matched_verbatims)
end = time.time()
print('TIME: ', end-start)
for ind, v in enumerate([x['raw'] for x in data_samples_dicts if x['raw'] not in total_matched_verbatims]):
    print("#%d: %s" % (ind, v))

