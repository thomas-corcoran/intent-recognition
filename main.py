import glob
import os

from nltk.tokenize import TweetTokenizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

tknzr = TweetTokenizer()
cnt = CountVectorizer(tokenizer=tknzr.tokenize)

sentences = []
intention_files = glob.glob('data/*.txt')
intentions = {}
y = []
for label,fn in enumerate(intention_files):
    intentions[label]=os.path.splitext(os.path.split(fn)[-1])[0]
    with open(fn) as fi:
        for line in fi:
            sentences.append(line.strip())
            y.append(label)

X = cnt.fit_transform(sentences)
clf = LogisticRegression()


