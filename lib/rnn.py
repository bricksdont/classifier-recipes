#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Mathias Mueller / mmueller@cl.uzh.ch
# Based on DyNet biLSTM tagger tutorial example:
# https://github.com/clab/dynet/blob/master/examples/tagger/bilstmtagger.py

from __future__ import unicode_literals

import argparse
import random
import codecs
import dynet as dy
import numpy as np
import sys
import time

import logging

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from sklearn import metrics

# preprocessing helper module
from preprocessing import Preprocessor

logger = logging.getLogger(__name__)

sys.stdout = codecs.getwriter('utf-8')(sys.__stdout__)
sys.stderr = codecs.getwriter('utf-8')(sys.__stderr__)
sys.stdin = codecs.getreader('utf-8')(sys.__stdin__)


class RNNClassifier(BaseEstimator, ClassifierMixin):
    """
    Classiy sequences with simple embeddings and pooling.

    Classifier object is compatible with scikit-learn's API,
    i.e. it can be used in Random Search or pipelines.
    """
    def __init__(self,
                 max_epochs=50,
                 validation_interval=5000,
                 validation_metric='log_loss',
                 early_stop=True,
                 early_stop_patience=5,
                 early_stop_tol=0.01,
                 char_embedding_size=20,
                 word_embedding_size=40,
                 add_char_noise=False,
                 add_word_noise=True,
                 word_hidden_output_size=64,
                 word_num_hidden_layers=2,
                 char_num_hidden_layers=2,
                 word_dropout=False,
                 char_dropout=False,
                 pooling_method='average',
                 average_dropout=None,
                 random_state=None,
                 model_path="model.m"):

        # training parameters
        self.max_epochs = max_epochs
        self.validation_interval = validation_interval
        self.validation_metric = validation_metric
        self.early_stop = early_stop
        self.early_stop_patience = early_stop_patience
        self.early_stop_tol = early_stop_tol

        # model parameters
        self.char_embedding_size = char_embedding_size
        self.word_embedding_size = word_embedding_size
        self.add_char_noise = add_char_noise
        self.add_word_noise = add_word_noise
        self.word_hidden_output_size = word_hidden_output_size
        self.word_num_hidden_layers = word_num_hidden_layers
        self.char_num_hidden_layers = char_num_hidden_layers
        self.word_dropout = word_dropout
        self.char_dropout = char_dropout
        self.pooling_method = pooling_method
        self.average_dropout = average_dropout

        # other
        self.random_state = random_state
        self.model_path = model_path

    def fit(self, X, y):
        """
        Learning the model.
        """
        # Check that X and y have correct shapes
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.num_labels = len(self.classes_)

        # vocabulary size: all known words plus the unknown token
        self.num_words = len(np.unique(X)) + 1
        self.unknown_token_id = 0

        # if early stop, then train_test_split
        if self.early_stop:
            self.X_, self.X_dev, self.y_, self.y_dev = train_test_split(X, y, test_size=0.15,
                                                                        random_state=self.random_state)
        else:
            self.X_ = X
            self.y_ = y

        self._init_params()
        self._train_model()

        # Return the classifier
        return self

    def _load_best_model(self):
        """
        Load best model after training has ended.
        """
        if not self._loaded_best_model:
            logger.info("Loading best model from '%s'." % (self.model_path + ".m"))
            self._init_params()
            self.model.populate(self.model_path + ".m")

            self._loaded_best_model = True

    def predict(self, X, probs=None):
        """
        Predict with a fitted model.

        :param X: array-like with shape [n_samples, n_features]
        :param probs: specify probs if already known
        """
        # Check if fit was called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        # check if called during validation
        if self._training_stopped:
            # called during actual prediction, load best model
            self._load_best_model()

        y_preds = []

        if probs is not None:
            y_preds = self._predict_labels_given_probs(X, probs, train_mode=False)
        else:
            for sentence in X:
                y_pred = self._predict_label(sentence, train_mode=False)
                y_preds.append(y_pred)

        # return only the label here
        return y_preds

    def predict_proba(self, X):
        """
        For input sentences, return probabilites for all labels.
        """
        # Check if fit was called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        # check if called during validation
        if self._training_stopped:
            # called during actual prediction, load best model
            self._load_best_model()

        y_probs = []

        for sentence in X:
            # called `y_prob` although it contains several values
            y_prob = self._predict_probs(sentence, train_mode=False)
            y_probs.append(y_prob)

        # return the probabilities here
        return y_probs

    def score(self, X, y, metric="log_loss"):
        """
        Return a score, e.g. for Random Search.
        """
        # Check if fit was called
        check_is_fitted(self, ['X_', 'y_'])

        if metric == "log_loss":
            y_probs = self.predict_proba(X)
            return metrics.log_loss(y, y_probs)
        elif metric == "accuracy":
            y_pred = self.predict(X)
            return metrics.accuracy_score(y, y_pred)
        else:
            raise NotImplementedError

    def _word_rep(self, word_index):
        """
        Look up a word index and return a word embedding.
        """
        return self.word_lookup[word_index]

    def _init_params(self):
        """
        Defines all model parameters.
        """

        self.model = dy.Model()
        self.trainer = dy.AdamTrainer(self.model)

        self.word_lookup = self.model.add_lookup_parameters((self.num_words, self.word_embedding_size))
        # self.chars_lookup = self.model.add_lookup_parameters((self.num_chars, self.char_embedding_size))

        # word-level LSTMs
        self.word_lstm_input_size = self.word_embedding_size  # + 2 * self.char_embedding_size

        self.fwd_word_rnn = dy.CoupledLSTMBuilder(self.word_num_hidden_layers,   # number of layers
                                                  self.word_lstm_input_size,     # input dimension
                                                  self.word_hidden_output_size,  # output dimension
                                                  self.model)
        self.bwd_word_rnn = dy.CoupledLSTMBuilder(self.word_num_hidden_layers,
                                                  self.word_lstm_input_size,
                                                  self.word_hidden_output_size,
                                                  self.model)

        # char-level LSTMs
        # self.fwd_char_rnn = dy.CoupledLSTMBuilder(self.char_num_hidden_layers,
        #                                          self.char_embedding_size,
        #                                          self.char_embedding_size,
        #                                          self.model)
        # self.bwd_char_rnn = dy.CoupledLSTMBuilder(self.char_num_hidden_layers,
        #                                          self.char_embedding_size,
        #                                          self.char_embedding_size,
        #                                          self.model)

        # set variational dropout
        if self.word_dropout:
            self.fwd_word_rnn.set_dropout(0.2)
            self.bwd_word_rnn.set_dropout(0.2)
        # if self._char_dropout:
        #    self.fwd_char_rnn.set_dropout(0.2)
        #    self.bwd_char_rnn.set_dropout(0.2)

        self.softmax_weight = self.model.add_parameters((self.num_labels, self.word_hidden_output_size * 2))
        self.softmax_bias = self.model.add_parameters((self.num_labels,))

        # uncomment once we update our dynet library version
        # logger.debug("Model parameter collection: %s" % str(self.model.parameters_list()))

    def _build_computation_graph(self, words, train_mode=True):
        """
        Builds the computational graph.
        """
        dy.renew_cg()
        # turn parameters into expressions
        softmax_weight_exp = dy.parameter(self.softmax_weight)
        softmax_bias_exp = dy.parameter(self.softmax_bias)

        # initialize the RNNs
        f_init = self.fwd_word_rnn.initial_state()
        b_init = self.bwd_word_rnn.initial_state()

        # cf_init = self.fwd_char_rnn.initial_state()
        # cb_init = self.bwd_char_rnn.initial_state()

        # only use word-level for now
        word_reps = [self._word_rep(word) for word in words]

        if train_mode and self.add_word_noise:
            word_reps = [dy.noise(word_rep, 0.05) for word_rep in word_reps]

        # feed word vectors into biLSTM
        fw_exps = f_init.transduce(word_reps)
        bw_exps = b_init.transduce(reversed(word_reps))

        if self.pooling_method == "last":
            average_lstm = dy.concatenate([fw_exps[-1], bw_exps[-1]])
        else:
            bi_exps = [dy.concatenate([f, b]) for f, b in zip(fw_exps, reversed(bw_exps))]
            bi_exps = dy.concatenate(bi_exps, d=1)

            if self.pooling_method == "average":
                average_lstm = dy.mean_dim(bi_exps, d=1)
            elif self.pooling_method == "max":
                average_lstm = dy.max_dim(bi_exps, d=1)
            else:
                raise NotImplementedError

        if self.average_dropout is not None:
            average_lstm = dy.dropout(average_lstm, p=self.average_dropout)

        return softmax_weight_exp * average_lstm + softmax_bias_exp

    def _predict_probs(self, words, train_mode):
        """
        Return the probabilities for all labels given a sentence.
        """
        vector = self._build_computation_graph(words, train_mode)
        vector = dy.softmax(vector)

        return vector.npvalue()

    def _predict_label(self, words, train_mode, probs=None):
        """
        Return the most probable label id for an input sentence.
        """
        if probs is None:
            probs = self._predict_probs(words, train_mode)
        label = np.argmax(probs)

        return label

    def _predict_labels_given_probs(self, X, probs, train_mode):
        """
        Predict labels if probs are already known.
        """
        y_preds = []
        for sentence, prob in zip(X, probs):
            y_pred = self._predict_label(sentence, train_mode=train_mode, probs=prob)
            y_preds.append(y_pred)

        return y_preds

    def _validate(self, updates=0):
        """
        Validate on development set.
        """
        y_dev_probs = self.predict_proba(self.X_dev)
        # avoid rebuilding computational graph
        y_dev_pred = self.predict(self.X_dev, probs=y_dev_probs)

        accuracy = metrics.accuracy_score(self.y_dev, y_dev_pred)
        loss = metrics.log_loss(self.y_dev, y_dev_probs)

        logger.info("Update %r: dev loss/sent=%.4f, acc=%.4f" % (updates, loss, accuracy))

        # Early stop here
        if self.early_stop:
            if (loss + self.early_stop_tol) <= self._best_dev_loss:
                logger.info("Dev loss improved to %.4f" % loss)
                self._best_dev_loss = loss
                self._dev_loss_not_improved = 0

                # Save best model
                logger.info("Saving best model to '%s'." % (self.model_path + ".m"))
                self.model.save(self.model_path + ".m")
            else:
                self._dev_loss_not_improved += 1
                if self._dev_loss_not_improved > self.early_stop_patience:
                    logger.info("Model has not improved for %d validation steps, stopping." % self._dev_loss_not_improved)
                    logger.info("Best dev loss: %.4f" % self._best_dev_loss)
                    self._training_stopped = True
                else:
                    logger.info("Model has not improved for %d validation steps." % self._dev_loss_not_improved)

    def _train_sentence(self, X, y):
        """
        Returns the sentence loss.
        """
        vector = self._build_computation_graph(X)

        # count correctly predicted labels for accuracy
        y_pred = np.argmax(vector.npvalue())
        if y_pred == y:
            self._num_correct += 1

        # loss for back-propagation
        sentence_loss = dy.pickneglogsoftmax(vector, y)

        self._cum_loss += sentence_loss.scalar_value()
        self._num_trained += 1
        sentence_loss.backward()
        self.trainer.update()

    def _train_model(self):
        """
        Trains a sequence classification model.
        """
        logger.info("Start training model")

        all_updates = 0
        start_time = time.time()

        self._best_dev_loss = 10000  # nonsense number
        self._dev_loss_not_improved = 0
        self._training_stopped = False

        for iteration in xrange(self.max_epochs):
            self._num_trained, self._num_correct, self._cum_loss = 0, 0, 0.

            # shuffle for each epoch
            tuples = list(zip(self.X_, self.y_))
            random.shuffle(tuples)
            X_current, y_current = zip(*tuples)

            for sentence, label in zip(X_current, y_current):
                all_updates += 1
                if (self._num_trained % 1000 == 0) and not self._num_trained == 0:
                    logger.info("Epoch %r: train loss/sent=%.4f, acc=%.4f, time=%.2fs" %
                                (iteration, self._cum_loss / self._num_trained,
                                 self._num_correct / float(self._num_trained), time.time() - start_time))

                if self.early_stop and (all_updates % self.validation_interval == 0):
                    self._validate(all_updates)
                    if self._training_stopped:
                        logger.info("Stop training model.")
                        return

                # train on the current sentence
                self._train_sentence(sentence, label)
        logger.info("Running out of epochs. Stop training model.")


def parse_args():
    """
    Command line arguments.
    """
    parser = argparse.ArgumentParser(description='dynet sequence classification.')

    io_args = parser.add_argument_group("Input and output arguments")

    io_args.add_argument('-l', '--logfile', default="log", metavar="FILE",
                         help='write log to FILE (default: %(default)s)')
    io_args.add_argument('-q', '--quiet', action='store_true', default=False,
                         help='do not print status messages to stderr (default: %(default)s)')
    io_args.add_argument('-d', '--debug', action="store_true", default=False,
                         help='print debug information (default: %(default)s)')
    io_args.add_argument('-M', '--model_path', default='model',
                         help='path where model should be saved (default: %(default)s)')
    io_args.add_argument('-T', '--train_file', required=True,
                         help='Training file (CSV format) (default: %(default)s)')
    io_args.add_argument('-E', '--test_file', default=None,
                         help='test file (CSV format) (default: %(default)s)')
    io_args.add_argument('--lowercase', action="store_true", default=False,
                         help='lowercase all strings before training and prediction (default: %(default)s)')

    io_args.add_argument('--min_seq_length', type=int, default=None, metavar="INT",
                         help='remove (= do not use for training) sequences shorter than INT characters (default: %(default)s)')
    io_args.add_argument('--max_seq_length', type=int, default=None, metavar="INT",
                         help='split sequences longer than INT characters into several examples (default: %(default)s)')
    io_args.add_argument('--ngram_range', type=int, default=1, metavar="INT",
                         help='add ngrams up to length INT to sequences (default: %(default)s)')

    model_args = parser.add_argument_group("Model parameters")

    model_args.add_argument('-S', '--seed', type=int, default=None,
                            help='Random seed value (default: %(default)s)')
    model_args.add_argument('--dynet-memory', type=int,
                            help='Dynet memory value in MB (default: %(default)s)')

    model_args.add_argument('--char_embedding_size', default=10, type=int,
                            help='dimensions of character embeddings (default: %(default)s)')
    model_args.add_argument('--word_embedding_size', default=20, type=int,
                            help='dimensions of word embeddings (default: %(default)s)')

    model_args.add_argument('--add_char_noise', action="store_true", default=False,
                            help='Add a small amount of noise to character embeddings. (default: %(default)s)')
    model_args.add_argument('--add_word_noise', action="store_true", default=True,
                            help='Add a small amount of noise to word embeddings. (default: %(default)s)')

    model_args.add_argument('--word_hidden_output_size', default=32, type=int,
                            help='dimensions of word LSTM output vectors (default: %(default)s)')

    model_args.add_argument('--char_num_hidden_layers', default=1, type=int,
                            help='hidden layers of character LSTMs (default: %(default)s)')
    model_args.add_argument('--word_num_hidden_layers', default=1, type=int,
                            help='hidden layers of word LSTMs (default: %(default)s)')
    model_args.add_argument('--word_dropout', default=False, action="store_true",
                            help='add variational dropout to word LSTMs (default: %(default)s)')
    model_args.add_argument('--char_dropout', default=False, action="store_true",
                            help='add variational dropout to word LSTMs (default: %(default)s)')
    model_args.add_argument('--pooling', default="average", type=str,
                            dest="pooling_method", choices=["last", "average", "max"],
                            help="method that pools embeddings (default: %(default)s)")
    model_args.add_argument('--average_embedding_dropout', default=None, type=float,
                            metavar="FLOAT", dest="average_dropout",
                            help='with probability FLOAT drop out nodes from averaged embedding vector (default: %(default)s)')

    train_args = parser.add_argument_group("Training parameters")

    train_args.add_argument('--max_epochs', type=int, default=50,
                            help="Maximum number of epochs the model is trained. (default: %(default)s)")
    train_args.add_argument('--validation_interval', type=int, default=10000, metavar="INT",
                            help="Validate with development data after INT updates. (default: %(default)s)")
    train_args.add_argument('--validation_metric', type=str, default='log_loss',
                            help="Metric used for early stopping. (default: %(default)s)")
    train_args.add_argument('--early_stop', action="store_true", default=False,
                            help="Early stop training with development set. (default: %(default)s)")
    train_args.add_argument('--patience', type=int, default=5, metavar="INT",
                            dest="early_stop_patience",
                            help="Early stop after model has not improved for over INT validation steps. (default: %(default)s)")
    train_args.add_argument('--early_stop_tol', type=float, default=0.01, metavar="FLOAT",
                            help="if the dev loss decreases by less than FLOAT, this will not be regarded as an improvement (default: %(default)s)")

    args = parser.parse_args()

    return args


def _set_up_logging(args):
    # log to logfile
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(asctime)s:%(levelname)s:%(name)s:%(funcName)s] %(message)s',
                        filename=args.logfile,
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


def main():
    """
    Invoke this module as a script.
    """
    args = parse_args()
    _set_up_logging(args)

    random.seed(a=args.seed)

    preprocessor = Preprocessor(lowercase=args.lowercase,
                                unknown_label_id=0,
                                ngram_range=args.ngram_range)

    rnn_classifier = RNNClassifier(max_epochs=args.max_epochs,
                                           validation_interval=args.validation_interval,
                                           validation_metric=args.validation_metric,
                                           early_stop=args.early_stop,
                                           early_stop_patience=args.early_stop_patience,
                                           early_stop_tol=args.early_stop_tol,
                                           char_embedding_size=args.char_embedding_size,
                                           word_embedding_size=args.word_embedding_size,
                                           add_char_noise=args.add_char_noise,
                                           add_word_noise=args.add_word_noise,
                                           word_hidden_output_size=args.word_hidden_output_size,
                                           word_num_hidden_layers=args.word_num_hidden_layers,
                                           char_num_hidden_layers=args.char_num_hidden_layers,
                                           word_dropout=args.word_dropout,
                                           char_dropout=args.char_dropout,
                                           pooling_method=args.pooling_method,
                                           average_dropout=args.average_dropout,
                                           random_state=args.seed,
                                           model_path=args.model_path)

    _, _, _, X, y = preprocessor.preprocess_file(file_path=args.train_file,
                                                 with_labels=True,
                                                 min_seq_length=args.min_seq_length,
                                                 max_seq_length=args.max_seq_length)

    rnn_classifier.fit(X, y)

    logger.info("Preparing test file.")
    # prepare test file entries into X_test
    ids, X_texts, _, X, y = preprocessor.preprocess_file(file_path=args.test_file,
                                                         with_labels=False,
                                                         fit=False)

    logger.info("Predicting test file.")
    # make predictions for entries in test file
    y_probs = rnn_classifier.predict_proba(X)
    y_label_ids = rnn_classifier.predict(X)

    # converting label ids back to string labels
    y_labels = preprocessor.inverse_transform_labels(y_label_ids)

    preprocessor.write_file(file_path=args.model_path + ".test.predictions.csv",
                            ids=ids,
                            X_texts=X_texts,
                            y_probs=y_probs,
                            y_labels=y_labels,
                            verbose=False)


if __name__ == '__main__':
    main()
