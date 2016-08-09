import pickle
import json
from nltk import word_tokenize
from intent_recognition.models import intention_model,intention_mapping,cnt,ner_tagger

class Intent(object):
    def __init__(self,actions,merge_func):
        self.actions = actions
        self.merge = merge_func
    def run_actions(self, chat_id, message, context):
        X = cnt.transform([message])
        pred = intention_model.predict(X)
        intention_str = intention_mapping[str(pred[0])]
        tagged = ner_tagger.tag(word_tokenize(message))
        self.merge(chat_id,context,message,tagged)
        self.actions[intention_str](chat_id,context)

