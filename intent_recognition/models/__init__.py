import pickle
import json
import os
from nltk.tag import StanfordNERTagger
this_dir = os.path.split(os.path.realpath(__file__))[0]

with open(os.path.join(this_dir,'intention.pkl')) as fi:
    intention_model = pickle.load(fi)
with open(os.path.join(this_dir,'intentions.json')) as fi:
    intention_mapping = json.load(fi)
with open(os.path.join(this_dir,'cnt.pkl')) as fi:
    cnt = pickle.load(fi)
ner_tagger = StanfordNERTagger(os.path.join(this_dir,'ner-model.ser.gz'))
