#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Mathias Mueller / mmueller@cl.uzh.ch
# Based on DyNet biLSTM tagger tutorial example:
# https://github.com/clab/dynet/blob/master/examples/tagger/bilstmtagger.py

from __future__ import unicode_literals
from collections import Counter

import argparse
import random
import codecs
import dynet as dy
import numpy as np
import sys
import time

import logging
import utils  # helper module
import iterator

logger = logging.getLogger(__name__)

sys.stdout = codecs.getwriter('utf-8')(sys.__stdout__)
sys.stderr = codecs.getwriter('utf-8')(sys.__stderr__)
sys.stdin = codecs.getreader('utf-8')(sys.__stdin__)


class CNNClassifier(object):
    """
    Sequence classification model in Dynet.
    """
    def __init__(self,
                 train_file,
                 dev_file,
                 lowercase=False,
                 max_epochs=20,
                 validation_interval=1000,
                 validation_metric='log_loss',
                 early_stop=True,
                 early_stop_patience=5,
                 char_level=False,
                 char_occurrence_threshold=None,
                 char_embedding_size=20,
                 word_embedding_size=20,
                 cnn_filter_size=64,
                 cnn_window_size=3,
                 outfile_prefix="model"
                 ):
        """
        Loads data, builds vocabularies, initializes params, builds model.
        """
        self._train_file = train_file
        self._dev_file = dev_file
        self._lowercase = lowercase

        self._max_epochs = max_epochs

        self._validation_interval = validation_interval
        self._validation_metric = validation_metric
        self._early_stop = early_stop
        self._early_stop_patience = early_stop_patience

        self._char_level = char_level
        self._char_occurrence_threshold = char_occurrence_threshold
        self._char_embedding_size = char_embedding_size
        self._word_embedding_size = word_embedding_size
        self._cnn_filter_size = cnn_filter_size
        self._cnn_window_size = cnn_window_size

        self._outfile_prefix = outfile_prefix

        # variables hold all lines read from each file respectively
        self.train, self.dev = [], []

        # class distribution and number of classes estimated from training data
        self.class_dist, self.num_classes = None, None

        self._load_data()

        self._init_vocabs()
        self._init_params()

    def _load_data(self):
        """
        Load data from files.
        """
        train_iterator = iterator.DataIterator(self._train_file,
                                               class_dist=None,
                                               lowercase=self._lowercase)

        # load all data at once for now, change in a low-memory setting
        self.train = list(train_iterator.read())
        self.class_dist = train_iterator.class_dist
        self.num_classes = train_iterator.num_classes

        self._dev_iterator = iterator.DataIterator(self._dev_file,
                                                   class_dist=self.class_dist,
                                                   lowercase=self._lowercase)
        self.dev = list(self._dev_iterator.read())

        for name, dataset in zip(["train", "dev"], [self.train, self.dev]):
            length = len(dataset)
            logger.info("Length of dataset '%s': %d" % (name, length))

    def _init_vocabs(self):
        """
        Builds vocabularies for train, dev and test data.
        """
        all_tokens = []
        all_labels = []
        char_counter = Counter()

        for sentence in self.train:
            tokens, label = sentence

            all_tokens.extend(tokens)
            all_labels.append(label)
            char_counter.update("".join(tokens))

        for char in char_counter.keys():
            if self._char_occurrence_threshold and \
               char_counter[char] < self._char_occurrence_threshold:
                del char_counter[char]

        all_chars = set(char_counter)
        # add padding symbol
        all_chars.add("<*>")
        # add whitespace symbol
        all_chars.add(" ")

        # add unknown token symbol (UNK)
        all_tokens.append("_Spooky!_")

        self.word_vocab = utils.Vocab.from_corpus(all_tokens)
        self.label_vocab = utils.Vocab.from_corpus(all_labels)
        self.char_vocab = utils.Vocab.from_corpus(all_chars)

        # index for unkown tokens
        self._unknown_token_id = self.word_vocab.w2i["_Spooky!_"]

        self.num_words = self.word_vocab.size()
        self.num_labels = self.label_vocab.size()
        self.num_chars = self.char_vocab.size()

        logger.debug('Word vocabulary size: %d' % self.num_words)
        logger.debug('Number of different labels: %d' % self.num_labels)
        logger.debug('Character vocabulary size: %d' % self.num_chars)

        logger.debug("Label vocab i2w: %s" % self.label_vocab.i2w)
        logger.debug("Label vocab w2i: %s" % self.label_vocab.w2i)

    def _word_rep(self, word):
        """
        Look up a word index and return a word embedding.
        """
        w_index = self.word_vocab.w2i.get(word, self._unknown_token_id)

        return self.word_lookup[w_index]

    def _char_rep(self, word):
        """
        Representation of words as character vectors.
        """
        pad_char = self.char_vocab.w2i["<*>"]
        char_ids = [pad_char] + [self.char_vocab.w2i[c] for c in word if c in self.char_vocab.w2i] + [pad_char]
        char_embs = [self.chars_lookup[char_id] for char_id in char_ids]

        return char_embs

    def _chars_rep(self, words):
        """
        Representation for the whole sentence.
        """
        string_ = " ".join(words)

        return self._char_rep(string_)

    def _init_params(self):
        """
        Defines all model parameters.
        """

        self.model = dy.Model()
        self.trainer = dy.AdamTrainer(self.model)

        self.word_lookup = self.model.add_lookup_parameters((self.num_words, 1, 1, self._word_embedding_size))
        self.chars_lookup = self.model.add_lookup_parameters((self.num_chars, 1, 1, self._char_embedding_size))

        if self._char_level:
            W_size = self._char_embedding_size
        else:
            W_size = self._word_embedding_size

        self.W_cnns = []
        self.b_cnns = []
        for window_size in range(self._cnn_window_size):
            # convolution weights
            W_cnn = self.model.add_parameters((1, window_size + 1, W_size, self._cnn_filter_size))
            self.W_cnns.append(W_cnn)

            # convolution bias
            b_cnn = self.model.add_parameters((self._cnn_filter_size))
            self.b_cnns.append(b_cnn)

        self.pO = self.model.add_parameters((self.num_labels, self._cnn_filter_size *
                                             self._cnn_window_size))

        # uncomment once we update our dynet library version
        # logger.debug("Model parameter collection: %s" % str(self.model.parameters_list()))

    def _build_tagging_graph(self, words, train_mode=True):
        """
        Builds the computational graph.

        Model similar to http://aclweb.org/anthology/D/D14/D14-1181.pdf.
        """
        dy.renew_cg()
        # turn parameters into expressions
        mlp_output = dy.parameter(self.pO)

        W_cnn_expressions = []
        b_cnn_expressions = []

        for W_cnn, b_cnn in zip(self.W_cnns, self.b_cnns):
            W_cnn_expressions.append(dy.parameter(W_cnn))
            b_cnn_expressions.append(dy.parameter(b_cnn))

        if len(words) < self._cnn_window_size:
            pad_char = "<*>"
            words += [pad_char] * (self._cnn_window_size - len(words))

        if self._char_level:
            cnn_in = dy.concatenate(self._chars_rep(words), d=1)
        else:
            word_reps = [self._word_rep(word) for word in words]
            cnn_in = dy.concatenate(word_reps, d=1)


        pools_out = []
        for W_cnn_express, b_cnn_express in zip(W_cnn_expressions, b_cnn_expressions):
            cnn_out = dy.conv2d_bias(cnn_in, W_cnn_express, b_cnn_express, stride=(1, 1), is_valid=False)

            # max-pooling
            pool_out = dy.max_dim(cnn_out, d=1)
            pool_out = dy.reshape(pool_out, (self._cnn_filter_size,))

            pools_out.append(pool_out)

        pools_concat = dy.concatenate(pools_out)

        return mlp_output * pools_concat

    def _tag_sent(self, words, train_mode):
        """
        Returns a labelled sentence.
        """
        vector = self._build_tagging_graph(words, train_mode)
        vector = dy.softmax(vector)
        probs = vector.npvalue()

        label = np.argmax(probs)
        label = self.label_vocab.i2w[label]

        return words, label, probs

    def _tag_dataset(self, sentences, train_mode, with_labels=True):
        """
        For each sequence of words in a list of sequences,
        return the words themselves, and the label, and the
        probabilities for all labels.

        If gold labels are given, also returns classification accuracy.
        """
        tagged_sentences = []

        if with_labels:
            num_correct = 0

            for sentence in sentences:
                tokens, gold_label = sentence
                tagged_sentence = self._tag_sent(tokens, train_mode)

                _, predicted_label, __ = tagged_sentence
                if predicted_label == gold_label:
                    num_correct += 1

                tagged_sentences.append(tagged_sentence)
            return tagged_sentences, num_correct / float(len(tagged_sentences))
        else:
            for sentence in sentences:
                tokens = sentence
                tagged_sentence = self._tag_sent(tokens, train_mode)
                tagged_sentences.append(tagged_sentence)
            return tagged_sentences

    def _validate(self, updates=0):
        """
        Validate on development set.
        """
        tagged_dev_sentences, accuracy = self._tag_dataset(self.dev, train_mode=False)

        loss = utils.compute_loss(tagged_dev_sentences, self.dev, 'dev')

        iterator.write(sentences=tagged_dev_sentences,
                       ids=self._dev_iterator.ids,
                       file_name=self._outfile_prefix + '_devset.csv',
                       verbose=True)

        logger.info("Update %r: dev loss/sent=%.4f, acc=%.4f" % (updates, loss, accuracy))

        # Early stop here
        if self._early_stop:
            if loss < self._best_dev_loss:
                logger.info("Dev loss improved to %.4f" % loss)
                self._best_dev_loss = loss
                self._dev_loss_not_improved = 0

                # Save best model
                model_path = self._outfile_prefix + ".m"
                logger.info("Saving best model to '%s'." % model_path)
                self.model.save(model_path)
            else:
                self._dev_loss_not_improved += 1
                if self._dev_loss_not_improved > self._early_stop_patience:
                    logger.info("Model has not improved for %d validation steps, stopping." % self._dev_loss_not_improved)
                    logger.info("Best dev loss: %.4f" % self._best_dev_loss)
                    self._training_stopped = True
                else:
                    logger.info("Model has not improved for %d validation steps." % self._dev_loss_not_improved)

    def test_model(self, test_file):
        """
        After model training has stopped, tag test set.
        """
        self._test_file = test_file
        logger.info("Reading in test set.")
        test_iterator = iterator.DataIterator(self._test_file,
                                              class_dist=self.class_dist,
                                              with_labels=False,
                                              lowercase=self._lowercase)
        self.test = list(test_iterator.read())

        logger.info("Length of dataset 'test': %d" % (len(self.test)))

        logger.info("Start labelling the test set.")

        # Load the best model
        model_path = self._outfile_prefix + ".m"
        logger.info("Loading best model from '%s'." % model_path)
        self._init_params()
        self.model.populate(model_path)

        tagged_test_sentences = self._tag_dataset(self.test, train_mode=False, with_labels=False)

        iterator.write(sentences=tagged_test_sentences,
                       ids=test_iterator.ids,
                       file_name=self._outfile_prefix + '_testset.csv',
                       verbose=False)  # non-verbose: write only ids and probs of all labels

        logger.info("Finished labelling the test set.")

    def _train_sentence(self, sentence):
        """
        Returns the sentence loss.
        """
        tokens, gold_label = sentence

        vector = self._build_tagging_graph(tokens)
        gold_label_id = self.label_vocab.w2i[gold_label]

        # count correctly predicted labels for accuracy
        predicted_label_id = np.argmax(vector.npvalue())
        if predicted_label_id == gold_label_id:
            self._num_correct += 1

        # loss for back-propagation
        sentence_loss = dy.pickneglogsoftmax(vector, gold_label_id)

        self._cum_loss += sentence_loss.scalar_value()
        self._num_trained += 1
        sentence_loss.backward()
        self.trainer.update()

    def train_model(self):
        """
        Trains a sequence tagging model.
        """
        logger.info("Start training model")

        all_updates = 0
        start_time = time.time()

        self._best_dev_loss = 10000  # nonsense number
        self._dev_loss_not_improved = 0
        self._training_stopped = False

        for iteration in xrange(self._max_epochs):
            self._num_trained, self._num_correct, self._cum_loss = 0, 0, 0.
            # shuffle for each epoch
            random.shuffle(self.train)

            for sentence in self.train:
                all_updates += 1
                if (self._num_trained % 1000 == 0) and not self._num_trained == 0:
                    logger.info("Epoch %r: train loss/sent=%.4f, acc=%.4f, time=%.2fs" %
                                (iteration, self._cum_loss / self._num_trained,
                                 self._num_correct / float(self._num_trained), time.time() - start_time))

                if all_updates % self._validation_interval == 0:
                    self._validate(all_updates)
                    if self._training_stopped:
                        return

                # train on the current sentence
                self._train_sentence(sentence)


def parse_args():
    """
    Command line arguments.
    """
    parser = argparse.ArgumentParser(description='dynet sequence tagging.')

    io_args = parser.add_argument_group("Input and output arguments")

    io_args.add_argument('-l', '--logfile', default=None,
                         help='write log to FILE (default: --outfile_prefix + ".log")')
    io_args.add_argument('-q', '--quiet', action='store_true', default=False,
                         help='do not print status messages to stderr (default: %(default)s)')
    io_args.add_argument('-d', '--debug', action="store_true", default=False,
                         help='print debug information (default: %(default)s)')
    io_args.add_argument('-M', '--outfile_prefix', default='model',
                         help='prefix for output files (default: %(default)s)')
    io_args.add_argument('-T', '--train_file', required=True,
                         help='Training file (PENN format) (default: %(default)s)')
    io_args.add_argument('-D', '--dev_file',
                         help='development file (PENN format) (default: %(default)s)')
    io_args.add_argument('-E', '--test_file', default=None,
                         help='test file (PENN format) (default: %(default)s)')
    io_args.add_argument('--lowercase', action="store_true", default=False,
                         help='lowercase all strings before training and prediction (default: %(default)s)')

    model_args = parser.add_argument_group("Model parameters")

    model_args.add_argument('-S', '--dynet-seed', type=int, default=42,
                            help='Dynet seed value (default: %(default)s)')
    model_args.add_argument('--dynet-memory', type=int,
                            help='Dynet memory value in MB (default: %(default)s)')
    model_args.add_argument('--char_occurrence_threshold', type=int, default=None,
                            help="Filter characters that occur fewer times than the threshold. (default: %(default)s)")

    model_args.add_argument('--char_embedding_size', default=20, type=int,
                            help='dimensions of character embeddings (default: %(default)s)')
    model_args.add_argument('--word_embedding_size', default=20, type=int,
                            help='dimensions of word embeddings (default: %(default)s)')

    model_args.add_argument('--char_level_embeddings', action="store_true", default=False,
                            dest="char_level",
                            help="Use character-level embeddings (default: %(default)s)")

    model_args.add_argument('--cnn_filter_size', default=64, type=int,
                            help='size of CNN filters (default: %(default)s)')
    model_args.add_argument('--cnn_window_size', default=3, type=int,
                            help='CNN window size (default: %(default)s)')

    train_args = parser.add_argument_group("Training parameters")

    train_args.add_argument('--max_epochs', type=int, default=20,
                            help="Maximum number of epochs the model is trained. (default: %(default)s)")
    train_args.add_argument('--validation_interval', type=int, default=10000, metavar="INT",
                            help="Validate with development data after INT updates. (default: %(default)s)")
    train_args.add_argument('--validation_metric', type=str, default='log_loss',
                            help="Metric used for validation. (default: %(default)s)")
    train_args.add_argument('--early_stop', action="store_true", default=False,
                            help="Early stop training with development set. (default: %(default)s)")

    train_args.add_argument('--patience', type=int, default=5, metavar="INT",
                            dest="early_stop_patience",
                            help="Early stop after model has not improved for over INT validation steps. (default: %(default)s)")

    args = parser.parse_args()

    return args


def main():
    """
    Invoke this module as a script.
    """
    args = parse_args()

    # log to logfile
    logfile = args.logfile if args.logfile else args.outfile_prefix + ".log"
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(asctime)s:%(levelname)s:%(name)s:%(funcName)s] %(message)s',
                        filename=logfile,
                        filemode="w")

    if args.quiet:
        level = logging.WARNING
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    # log to STDERR
    console = logging.StreamHandler()
    console.setLevel(level)
    formatter = logging.Formatter('[%(asctime)s:%(levelname)s:%(name)s:%(funcName)s] %(message)s')
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    logger.info(args)

    random.seed(a=args.dynet_seed)

    sequence_model = CNNClassifier(train_file=args.train_file,
                                   dev_file=args.dev_file,
                                   lowercase=args.lowercase,
                                   max_epochs=args.max_epochs,
                                   validation_interval=args.validation_interval,
                                   validation_metric=args.validation_metric,
                                   early_stop=args.early_stop,
                                   early_stop_patience=args.early_stop_patience,
                                   char_level=args.char_level,
                                   char_occurrence_threshold=args.char_occurrence_threshold,
                                   char_embedding_size=args.char_embedding_size,
                                   word_embedding_size=args.word_embedding_size,
                                   cnn_filter_size=args.cnn_filter_size,
                                   cnn_window_size=args.cnn_window_size,
                                   outfile_prefix=args.outfile_prefix)
    sequence_model.train_model()

    if args.test_file:
        sequence_model.test_model(args.test_file)


if __name__ == '__main__':
    main()
