import glob
from nltk.tokenize import TweetTokenizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

tknzr = TweetTokenizer()
cnt = CountVectorizer(tokenizer=tknzr.tokenize)

sentences = []
intentions = glob.glob('data/*.txt')
y = []
for label,fn in enumerate(intentions):
    with open(fn) as fi:
        for line in fi:
            sentences.append(line.strip())
            y.append(label)

X = cnt.fit_transform(sentences)
clf = LogisticRegression()


