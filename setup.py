from distutils.core import setup

setup(
    name='intent_recognition',
    version='0.1dev',
    packages = ['intent'],
    install_requires = ['nltk','sklearn']
    )
