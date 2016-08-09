import pickle
import json
import glob
import os
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.chunk import ChunkParserI
from nltk.corpus.reader import ChunkedCorpusReader
from nltk.chunk.util import conlltags2tree, tree2conlltags
from nltk.tag import UnigramTagger, BigramTagger
from nltk.tokenize import TweetTokenizer

cnt = CountVectorizer(tokenizer=nltk.word_tokenize)

class TagChunker(ChunkParserI):
	'''Chunks tagged tokens using Ngram Tagging.
	source: https://github.com/japerk/nltk-trainer/blob/master/nltk_trainer/chunking/chunkers.py 
	'''
	def __init__(self, train_chunks, tagger_classes=[UnigramTagger, BigramTagger]):
		'''Train Ngram taggers on chunked sentences'''
		train_sents = conll_tag_chunks(train_chunks)
		self.tagger = None
		
		for cls in tagger_classes:
			self.tagger = cls(train_sents, backoff=self.tagger)
	
	def parse(self, tagged_sent):
		'''Parsed tagged tokens into parse Tree of chunks'''
		if not tagged_sent: return None
		(words, tags) = zip(*tagged_sent)
		chunks = self.tagger.tag(tags)
		# create conll str for tree parsing
		return conlltags2tree([(w,t,c) for (w,(t,c)) in zip(words, chunks)])
def conll_tag_chunks(chunk_sents):
	'''Convert each chunked sentence to list of (tag, chunk_tag) tuples,
	so the final result is a list of lists of (tag, chunk_tag) tuples.
	>>> from nltk.tree import Tree
	>>> t = Tree('S', [Tree('NP', [('the', 'DT'), ('book', 'NN')])])
	>>> conll_tag_chunks([t])
	[[('DT', 'B-NP'), ('NN', 'I-NP')]]
	Source: https://github.com/japerk/nltk-trainer/blob/master/nltk_trainer/chunking/chunkers.py
	'''
	tagged_sents = [tree2conlltags(tree) for tree in chunk_sents]
	return [[(t, c) for (w, t, c) in sent] for sent in tagged_sents]

def createTagged(fn='data/intents/text_search.txt'):
    """Create pos tagged senteces
    input: text file where each sentence newline-delimited, sentence can be unprocessed
    returns: a string in format `token1/pos_tag token2/pos_tag2 token3/pos_tag ... ` followed
    by two newline chars
    """
    with open(fn) as fi:
        for line in fi:
            tokens = nltk.word_tokenize(line) # this isn't the best tokenizer
            tagged = nltk.pos_tag(tokens) # this isn't the best tagger
            yield ' '.join('/'.join(x) for x in tagged)+'\n\n' # join tags into `token/tag token2/tag2 ...` format
def writeTagged(fn='data/chunks/text_search.pos'):
    """writes strings given by createTagged to a file, ready to be chunked
    CAREFUL: this will overwrite previously chunked files
    """
    with open(fn,'w') as fo:
        for sentence in createTagged():
            fo.write(sentence)


def createIntentionModel(intention_glob_path='data/intents/*.txt'):
    sentences = []
    intention_files = glob.glob(intention_glob_path)
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
    clf.fit(X,y)
    return clf, intentions,cnt
def writeIntentionModel(clf,intentions_dict,cnt,clf_fn='models/intention.pkl',intentions_fn='models/intentions.json',
                        vector_fn='models/cnt.pkl'):
	with open(clf_fn,'w') as fo:
		pickle.dump(clf,fo)
	with open(intentions_fn,'w') as fo:
		json.dump(intentions_dict,fo)
	with open(vector_fn,'w') as fo:
		pickle.dump(cnt,fo)
def createChunker():
    chunks = ChunkedCorpusReader('data/chunks/','text_search.pos')
    tagger_classes = [UnigramTagger, BigramTagger]
    train_chunks = chunks.chunked_sents()
    chunker = TagChunker(train_chunks, tagger_classes)
    return chunker
def writeChunkerModel(chunker,fn='models/chunker.pkl'):
    with open(fn,'w') as fo:
        pickle.dump(chunker,fo)
def createTSVFile(raw_in='data/ner/ner.txt',progress='data/ner/ner_train_progress.json',tsv_out='data/ner/ner.tsv'):
    """Create TSV file for annotating NEs
    then run 
    java -cp ../stanford/stanford-ner-2015-12-09/stanford-ner.jar:../stanford/stanford-ner-2015-12-09/lib/* edu.stanford.nlp.ie.crf.CRFClassifier -prop config/ner.prop
    to create NER model
    """
    with open(progress) as fi:
        trained_to_line = json.load(fi)['trained_to_line']
    sents = []
    with open(raw_in) as fi:
        for i,line in enumerate(fi):
            if i >= trained_to_line:
                sents.append(line)
    with open(progress,'w') as fo:
        trained_to_line = json.dump({'trained_to_line':i+1},fo)
    with open(tsv_out,'a') as fo:
        fo.write('\n')
        for line in sents:
            toks = nltk.word_tokenize(line)
            fo.write('\tO\n'.join(toks)+'\tO\n\n')
