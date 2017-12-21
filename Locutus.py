import sys
import gensim
import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import words
from nltk.corpus import stopwords
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

    # no_words = ['no', 'na', 'none', 'nothing', 'nada', 'n', 'n/a'
    #             'idk', 'dunno', 'ok', 'yes', 'y', 'ya', 'its ok', 'k']
    # if input_list.lower() in no_words:
    #     return 1
    # else:
    #     return 0


def tokenize(input_list, stopset, bigram, trigram, stemmer):
    remove_list = ['game', 'fuck', 'would']
    step_one = [z for z in input_list if z not in stopset]
    step_two = trigram[bigram[step_one]]
    step_three = [stemmer.stem(w) for w in step_two]
    step_four = [l for l in step_three if l not in remove_list]
    return step_four


def load_data(file_name, gib_flag=0):
    verbatim_list = []
    total_verbatim_count = 0
    removed_for_gibberish = []
    blank_lines = 0
    # sid = SentimentIntensityAnalyzer()
    wordset = set(words.words())
    stopset = set(stopwords.words('english'))
    bigram = gensim.models.phrases.Phraser.load('data/bigram_verbatims.pkl')
    trigram = gensim.models.phrases.Phraser.load('data/trigram_verbatim.pkl')
    stemmer = gensim.parsing.porter.PorterStemmer()
    with open(file_name, encoding='utf-8') as f:
        for line in f:
            total_verbatim_count += 1
            raw = line.split('\n')[0]
            if raw == '':
                blank_lines += 1
                continue
            verbatim_list.append({'raw': raw})
            # verbatim_list[-1]['semantics'] = semantics(raw, sid)
            cleaned = clean_text(raw).split()
            verbatim_list[-1]['list'] = cleaned
            if gib_flag and gibberish_check(raw, wordset):
                removed_for_gibberish.append(raw)
                del verbatim_list[-1]
                continue
            verbatim_list[-1]['tokens'] = tokenize(cleaned, stopset, bigram, trigram, stemmer)

    print('blank lines: ', blank_lines)
    print('removed for gibberish: ', len(removed_for_gibberish))
    print('total verbatims: ', total_verbatim_count)

    return verbatim_list, removed_for_gibberish


def random_test(the_list, test_against, df):
    output_dict = {'test string': test_against, 'match count': 0, 'match list': []}
    same_count = 0
    distance_total = 0
    for teststr in the_list:
        distance = word2vec.wmdistance(test_against, teststr)
        if distance < df:
            same_count += 1
            distance_total += df-distance
            output_dict['match list'].append((teststr, distance))
    output_dict['match count'] = same_count
    output_dict['distance metric'] = distance_total
    return output_dict


def run_all(verbatim_file, output_file, num_topics=20, df=1.0, test_percent=100, remove_gibberish=0):
    start = time.time()
    outfile = open("data/" + output_file, 'w', encoding='utf-8')
    input_verbatims, gibberish = load_data(verbatim_file, remove_gibberish)
    v_tokens = [r['tokens'] for r in input_verbatims]

    tokens_duplicate = random.sample(v_tokens, int(test_percent / 100 * len(v_tokens)))
    print("length of list: ", len(tokens_duplicate))

    start_message = "Verbatim File Used: %s\nNumber Topics Set: %d\nDistance Factor Set: %f\n" \
                    "Number Verbatims Assessed: %d\nRemoved by Gibberish Filter: %d\n" % \
                    (verbatim_file, num_topics, df, len(tokens_duplicate), len(gibberish))
    outfile.write(start_message)
    print(start_message)
    outfile.write(str(gibberish) + '\n')

    topic_final = []
    for i in range(num_topics):
        # test_against_list = random.sample(tokens_duplicate, sample_range)
        test_against_list = tokens_duplicate
        if len(test_against_list) < 10:
            break
        test_len = 0
        for e, a in enumerate(test_against_list):
            output_d = random_test(tokens_duplicate, a, df)
            if output_d['distance metric'] > test_len:
                test_len = output_d['distance metric']
                saved_out = output_d
        original_verbatim = input_verbatims[v_tokens.index(saved_out['test string'])]['raw']
        outfile.write("#%d: (%d, %f) %s\n" % (i, saved_out['match count'], saved_out['distance metric'],
                                              original_verbatim))
        topic_final.append((saved_out['match count'], original_verbatim))
        for e, t in enumerate(sorted(saved_out['match list'], key=lambda x: x[1])):
            outfile.write('   - %d %s (%f)\n' % (e, input_verbatims[v_tokens.index(t[0])]['raw'], t[1]))
            tokens_duplicate.remove(t[0])

    for ind, v in enumerate(tokens_duplicate):
        outfile.write("#%d: %s\n" % (ind, input_verbatims[v_tokens.index(v)]['raw']))

    # print out topic summary
    outfile.write("\n-----------------\n")
    for i, topic_pairing in enumerate(topic_final):
        outfile.write("#%d: (%d matches) %s\n" % (i+1, topic_pairing[0], topic_pairing[1]))

    end = time.time()
    outfile.write('TIME: %f\n' % ((end-start) / 60.0))
    outfile.close()


if __name__ == "__main__":
    word2vec = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)
    word2vec.init_sims(replace=True)  # normalizes vectors

    # run_all('data/XIH_reviews_pos.txt', 'insider_30_10_pos.txt', 20, 1.0, 100, 1)
    # run_all('data/XIH_reviews_neg.txt', 'insider_30_10_neg.txt', 20, 1.0, 100, 1)
    run_all('data/PUBG_data_list3.txt', 'pubg_20_11_all3.txt', 20, 1.1, 100, 1)
    # run_all('data/KI_verbatims.txt', 'KI_30_11_(10).txt', 30, 1.1, 10, 1)
    # run_all('data/KI_verbatims.txt', 'KI_30_11_(20).txt', 30, 1.1, 20, 1)
    # run_all('data/KI_verbatims.txt', 'KI_30_11_(50).txt', 30, 1.1, 50, 1)
    # run_all('data/KI_verbatims.txt', 'KI_30_11_(100).txt', 30, 1.1, 100, 1)

    # input_file = sys.argv[1]
    # output_file = sys.argv[2]
    # num_topics = sys.argv[3]
    # distance_factor = sys.argv[4]
    # test_percent = sys.argv[5]
    #
    # run_all(input_file, output_file, num_topics, distance_factor, test_percent, 1)
