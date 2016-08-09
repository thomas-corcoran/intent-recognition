import pickle
import json

with open('models/intention.pkl') as fi:
    intention_model = pickle.load(fi)
with open('models/intentions.json') as fi:
    intention_mapping = json.load(fi)
with open('models/cnt.pkl') as fi:
    cnt = pickle.load(fi)
class Intention(object):
    def __init__(self,actions):
        self.actions = actions
    def run_actions(self, chat_id, message, context):
        X = cnt.transform([message])
        pred = intention_model.predict(X)
        intention_str = intention_mapping[str(pred[0])]
        self.actions[intention_str](chat_id,context)

                
