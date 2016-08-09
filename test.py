import glob
import os
import pickle

import nltk
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score

tknzr = TweetTokenizer()
cnt = CountVectorizer(tokenizer=tknzr.tokenize)
chunker = pickle.load(open("models/chunker.pkl"))

sentences = []
intention_files = glob.glob('data/intents/*.txt')
intentions = {}
y = []
for label,fn in enumerate(intention_files):
    intentions[label]=os.path.splitext(os.path.split(fn)[-1])[0]
    with open(fn) as fi:
        for line in fi:
            sentences.append(line.strip())
            y.append(label)

def xval(n_folds=10):
    X = cnt.fit_transform(sentences)
    clf = LogisticRegression()
    
    print "Mean score accorss {} folds: {}".format(n_folds,cross_val_score(clf,X,y=y,scoring='f1_weighted',cv=n_folds))

def getIntention(a_string):
    X = cnt.transform([a_string])
    pred = clf.predict(X)
    return intentions[pred[0]]
def getChunks(a_string):
    tokens = nltk.pos_tag(nltk.word_tokenize(a_string))
    tree = chunker.parse(tokens)
    search_words = ''
    for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
        search_words += ' '.join(x[0] for x in subtree.leaves())
    return search_words
if __name__=='__main__':
    X = cnt.fit_transform(sentences)
    clf = LogisticRegression()
    clf.fit(X,y)
    while True:
        a_string = raw_input(">")
        intention = getIntention(a_string)
        keywords = ''
        if intention == 'text_search':
            keywords = getChunks(a_string)
        print "Intention: {}, keywords: {}".format(intention,keywords)
