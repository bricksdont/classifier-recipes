#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Mathias Mueller / mmueller@cl.uzh.ch

from __future__ import unicode_literals

import codecs
import sys
import logging

# preprocessing
import pandas as pd
import csv
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# work around keras' inability to process unicode
import keras.preprocessing.text
import string


def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    if lower:
        text = text.lower()
    if type(text) == unicode:
        translate_table = {ord(c): ord(t) for c, t in zip(filters, split * len(filters))}
    else:
        translate_table = string.maketrans(filters, split * len(filters))
    text = text.translate(translate_table)
    seq = text.split(split)
    return [i for i in seq if i]


keras.preprocessing.text.text_to_word_sequence = text_to_word_sequence

sys.stdout = codecs.getwriter('utf-8')(sys.__stdout__)
sys.stderr = codecs.getwriter('utf-8')(sys.__stderr__)
sys.stdin = codecs.getreader('utf-8')(sys.__stdin__)

logger = logging.getLogger(__name__)


class Preprocessor(object):
    """
    Prepares texts to be used in classification tasks.
    """
    def __init__(self,
                 lowercase=False,
                 ngram_range=1,
                 unknown_label_id=0,):
        """
        Define common preprocessing behaviour for training and testing.
        """
        self._lowercase = lowercase
        self._ngram_range = ngram_range
        self._unknown_label_id = unknown_label_id

    def _tokenize(self, string_):
        """
        Tokenize and represent results as string again.
        """
        return " ".join(word_tokenize(string_))

    def _split_string(self, string_):
        """
        Split tokenized string at word boundaries so that max length of
        each output string is approximately self._max_seq_length.
        """
        strings = []
        temp = []

        for token in string_.split(" "):
            temp.append(token)
            temp_string = " ".join(temp)
            if len(temp_string) >= self._max_seq_length:
                strings.append(temp_string)
                temp = []
        # remaining text
        if temp != []:
            temp_string = " ".join(temp)
            strings.append(temp_string)

        return strings

    def _ngrams(self, string_):
        """
        Find all possible n-grams in the range (1, self._ngram_range).

        Source of code:
        http://locallyoptimal.com/blog/2013/01/20/elegant-n-gram-generation-in-python/
        """
        def find_ngrams(input_list, n):
            return zip(*[input_list[i:] for i in range(n)])

        ngrams = []
        tokens = string_.split()

        for size in range(1, self._ngram_range + 1):
            tuples = find_ngrams(tokens, size)
            concatenated = ["_".join(tuple_) for tuple_ in tuples]
            ngrams.extend(concatenated)

        return " ".join(ngrams)

    def preprocess_file(self,
                        file_path,
                        with_labels=True,
                        return_ids=False,
                        fit=True,
                        min_seq_length=None,
                        max_seq_length=None):
        """
        Takes a file path as input, returns preprocessed X and y.

        :param with_labels: indicate whether the CSV file has a label (= class) column
        """
        self._min_seq_length = min_seq_length
        self._max_seq_length = max_seq_length

        self._file_handle = codecs.open(file_path, "r", encoding="utf-8")

        data_frame = pd.read_csv(self._file_handle, delimiter=",", encoding="utf-8", header="infer")

        X, y, ids = [], [], []

        for _, row in data_frame.iterrows():
            ids.append(row.id)

            strings = []
            string_ = self._tokenize(row.text)

            if (self._max_seq_length is not None) and len(string_) > self._max_seq_length:
                # split into several examples and append them to `strings`
                strings.extend(self._split_string(string_))
            else:
                strings.append(string_)

            for string_ in strings:
                if (self._min_seq_length is not None) and len(string_) < self._min_seq_length:
                    # do not use this example for training
                    continue

                string_ = self._ngrams(string_)
                X.append(string_)
                if with_labels:
                    y.append(row.author)

        # encode X strings to sequence ids
        # keras tokenizer does not actually tokenize, rather encodes sequences
        if fit:
            self.encoder = Tokenizer(filters="", lower=self._lowercase)
            self.encoder.fit_on_texts(X)
        X_sequences = self.encoder.texts_to_sequences(X)

        # pad without max length at the moment
        X_sequences = pad_sequences(X_sequences)

        # encode y labels
        if fit:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(y)
            logger.info("Classes found in the training data: %s" % str(self.label_encoder.classes_))

        if with_labels:
            y_sequences = self.label_encoder.transform(y)
        else:
            y_sequences = []

        # debug samples
        for name, subset in zip(["ids", "X", "y", "X_sequences", "y_sequences"],
                                [samples[:5] for samples in [ids, X, y, X_sequences, y_sequences]]):
            logger.debug("Samples from %s: %s" % (name, str(subset)))

        logger.info("Preprocessed dataset has size: %d" % len(X))

        return ids, X, y, X_sequences, y_sequences

    def inverse_transform_labels(self, y):
        """
        Convert labels ids to string labels.
        """
        return self.label_encoder.inverse_transform(y)

    def write_file(self, file_path, ids, X_texts, y_probs, y_labels, verbose=False):
        """
        Write predictions to a CSV file.
        """
        frame_list = []
        for id_, X_text, y_label, y_probs in zip(ids, X_texts, y_labels, y_probs):
            if verbose:
                row = [id_, " ".join(X_text), y_label] + list(y_probs)
                columns = [u"id", u"text", u"label"] + list(self.label_encoder.classes_)
            else:
                row = [id_] + list(y_probs)
                columns = ["id"] + list(self.label_encoder.classes_)
            frame_list.append(row)

        data_frame = pd.DataFrame(frame_list, columns=columns)

        logger.info("Writing predictions to file '%s'." % file_path)
        data_frame.to_csv(file_path, encoding="utf-8", index=False, quoting=csv.QUOTE_NONNUMERIC)
