from nltk.sentiment.vader import SentimentIntensityAnalyzer


def semantics(input_string, sid=''):
    if not sid:
        sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(input_string)
    # ss['compound'] will return -1 to 1
    # the math below will change that to 0 to 1
    c_output = (ss['compound'] + 1) / 2
    return c_output


print(semantics('i love xbox! :)'))
print(semantics('i hate everything'))
print(semantics('meh'))
