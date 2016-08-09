from distutils.core import setup

setup(
    name='intent_recognition',
    version='0.1dev',
    packages = ['intent_recognition',
                'intent_recognition.intent',
                'intent_recognition.models'],
    install_requires = ['nltk==3.2.1','scikit-learn==0.17.1']
    )
