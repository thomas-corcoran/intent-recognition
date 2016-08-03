import nltk
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()

def makeTagged(fn='data/intents/text_search.txt'):
    with open(fn) as fi:
        for line in fi:
            tokens = nltk.word_tokenize(line) # this isn't the best tokenizer
            tagged = nltk.pos_tag(tokens) # this isn't the best tagger
            yield ' '.join('/'.join(x) for x in tagged)+'\n\n' # join tags into `token/tag token2/tag2 ...` format
        
with open('data/chunks/text_search.pos','w') as fo:
    for sentence in makeTagged():
        fo.write(sentence)
