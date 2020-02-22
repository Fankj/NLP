#!/usr/bin/env python3
"""
ENLP A2: Perceptron

Usage: python perceptron.py NITERATIONS

@author: Alan Ritter, Nathan Schneider

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""
import sys, os, glob

from collections import Counter
from math import log
from numpy import mean
import numpy as np
import nltk
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer

from evaluation import Eval

def load_docs(direc, lemmatize, labelMapFile='labels.csv'):
    """Return a list of word-token-lists, one per document.
    Words are optionally lemmatized with WordNet."""


    labelMap = {}   # docID => gold label, loaded from mapping file
    with open(os.path.join(direc, labelMapFile)) as inF:
        for ln in inF:
            docid, label = ln.strip().split(',')

            assert docid not in labelMap
            labelMap[docid] = label

    # create parallel lists of documents and labels
    docs, labels = [], []
    for file_path in sorted(glob.glob(os.path.join(direc, '*.txt'))):

        filename = os.path.basename(file_path)

        # open the file at file_path, construct a list of its word tokens,
        # and append that list to 'docs'.
        # look up the document's label and append it to 'labels'.

        doc = open(file_path)
        tokens = []
        labels.append(labelMap[filename])
        for line in doc:
            sentence = line.split(' ')
            for word in sentence:
                tokens.append(word)
        docs.append(tokens)
    return docs, labels

def extract_feats(doc):
    """
    Extract input features (percepts) for a given document.
    Each percept is a pairing of a name and a boolean, integer, or float value.
    A document's percepts are the same regardless of the label considered.
    """
    ff = Counter()
    ff['bias'] = 1
    prev_word = ""
    lemmatizer = WordNetLemmatizer()

    for token in doc:
        if token not in ff:
            ff[token] = 1

        if prev_word:
            ff[(prev_word, token)] = 1
        prev_word = token

        lemma = lemmatizer.lemmatize(token)
        ff[lemma] = 1

    return ff

def load_featurized_docs(datasplit):
    rawdocs, labels = load_docs(datasplit, lemmatize=False)
    assert len(rawdocs)==len(labels)>0,datasplit
    featdocs = []
    for d in rawdocs:
        featdocs.append(extract_feats(d))
    return featdocs, labels

class Perceptron:
    def __init__(self, train_docs, train_labels, MAX_ITERATIONS=100, dev_docs=None, dev_labels=None):
        self.CLASSES = ['ARA', 'DEU', 'FRA', 'HIN', 'ITA', 'JPN', 'KOR', 'SPA', 'TEL', 'TUR', 'ZHO']
        self.MAX_ITERATIONS = MAX_ITERATIONS
        self.dev_docs = dev_docs
        self.dev_labels = dev_labels
        self.weights = {l: Counter() for l in self.CLASSES}
        self.learn(train_docs, train_labels)

    def copy_weights(self):
        """
        Returns a copy of self.weights.
        """
        return {l: Counter(c) for l,c in self.weights.items()}

    def learn(self, train_docs, train_labels):
        """
        Train on the provided data with the perceptron algorithm.
        Up to self.MAX_ITERATIONS of learning.
        At the end of training, self.weights should contain the final model
        parameters.
        """
        for t in range(self.MAX_ITERATIONS):
            updates = 0
            for i in range(len(train_docs)):
                gold = train_labels[i]
                pred = self.predict(train_docs[i])
                if gold != pred:
                    self.weights[gold] += train_docs[i]
                    self.weights[pred] -= train_docs[i]
                    updates+=1
            #print("iteration: ", t, " updates=", updates, ", trainAcc=", self.test_eval(train_docs, train_labels),", devAcc=", self.test_eval(dev_docs,dev_labels), ", params=", sum(len(self.weights[l]) for l in self.CLASSES))


    def score(self, doc, label):
        """
        Returns the current model's score of labeling the given document
        with the given label.

        """
        ans = 0
        for token in doc:
            ans += self.weights[label][token] * doc[token]
        return ans

    def predict(self, doc):
        """
        Return the highest-scoring label for the document under the current model.
        """
        return max(self.CLASSES, key=lambda k: self.score(doc,k))

    def test_eval(self, test_docs, test_labels):
        pred_labels = [self.predict(d) for d in test_docs]
        ev = Eval(test_labels, pred_labels)
        return ev.accuracy(), ev.precision(), ev.recall(), ev.F1(), ev.confusion_matrix()

    def highest_weighted_features(self):
        lans = {l: {} for l in self.CLASSES}
        for l in lans:
            features =  self.weights[l].most_common(10)
            lans[l] = features
        return lans

    def lowest_weighted_features(self):
        lans = {l: {} for l in self.CLASSES}
        for l in lans:
            features =  self.weights[l].most_common()[:-11:-1]
            lans[l] = features
        return lans

    def bias_weight(self):
        w = {l: {} for l in self.CLASSES}
        for l in w:
            w[l] = self.weights[l]['bias']
        return w

if __name__ == "__main__":
    args = sys.argv[1:]
    niters = int(args[0])

    train_docs, train_labels = load_featurized_docs('train')
    print(len(train_docs), 'training docs with',
        sum(len(d) for d in train_docs)/len(train_docs), 'percepts on avg', file=sys.stderr)

    dev_docs,  dev_labels  = load_featurized_docs('dev')
    print(len(dev_docs), 'dev docs with',
        sum(len(d) for d in dev_docs)/len(dev_docs), 'percepts on avg', file=sys.stderr)


    test_docs,  test_labels  = load_featurized_docs('test')
    print(len(test_docs), 'test docs with',
        sum(len(d) for d in test_docs)/len(test_docs), 'percepts on avg', file=sys.stderr)

    train_counter = Counter(train_labels)
    dev_counter = Counter(dev_labels)
    print("train counter: ",train_counter)
    print("dev counter: ", dev_counter)


    ptron = Perceptron(train_docs, train_labels, MAX_ITERATIONS=niters, dev_docs=dev_docs, dev_labels=dev_labels)

    acc, prec, rec, f1, conf = ptron.test_eval(test_docs, test_labels)
    print(conf)
    print(ptron.highest_weighted_features())
    print(ptron.lowest_weighted_features())
    print(ptron.bias_weight())
    #acc, prec, rec, f1 = ptron.test_eval(test_docs, test_labels)
    print(prec)
    print("\n")
    print(rec)
    print("\n")
    print(f1)
    print("\n")
    print(acc, file=sys.stderr)

