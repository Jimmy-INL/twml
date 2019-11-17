# Copyright 2018, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified to use logistic regression instead of CNN
# and synthetic data instead of MNIST by Antti Honkela, 2019

"""Training a logistic regression model with differentially private SGD optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import sys
import pandas as pd
import numpy as np
import numpy.random as npr
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from tensorflow_privacy.privacy.analysis import privacy_ledger
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp_from_ledger
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from tensorflow_privacy.privacy.optimizers import dp_optimizer


AdamOptimizer = tf.compat.v1.train.AdamOptimizer

FLAGS = flags.FLAGS
try:
    flags.FLAGS['dpsgd']
except KeyError:
    flags.DEFINE_boolean(
        'dpsgd', True, 'If True, train with DP-SGD. If False, train with vanilla SGD.')
    flags.DEFINE_float('learning_rate', .0001, 'Learning rate for training')
    flags.DEFINE_float('noise_multiplier', 2.0,
                       'Ratio of the standard deviation to the clipping norm')
    flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
    flags.DEFINE_integer('batch_size', 64, 'Batch size')
    flags.DEFINE_integer('epochs', 2, 'Number of epochs')
    flags.DEFINE_integer('training_data_size', 2000, 'Training data size')
    flags.DEFINE_integer('test_data_size', 2000, 'Test data size')
    flags.DEFINE_integer('input_dimension', 5, 'Input dimension')
    flags.DEFINE_string('model_dir', None, 'Model directory')

class EpsilonPrintingTrainingHook(tf.estimator.SessionRunHook):
    """Training hook to print current value of epsilon after an epoch."""

    def __init__(self, ledger):
        """Initalizes the EpsilonPrintingTrainingHook.
        Args:
        ledger: The privacy ledger.
        """
        self._samples, self._queries = ledger.get_unformatted_ledger()

    def end(self, session):
        orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        samples = session.run(self._samples)
        queries = session.run(self._queries)
        formatted_ledger = privacy_ledger.format_ledger(samples, queries)
        rdp = compute_rdp_from_ledger(formatted_ledger, orders)
        eps = get_privacy_spent(orders, rdp, target_delta=1e-5)[0]
        sys.stdout.write(',%s' % eps)
        sys.stdout.flush()


def lr_model_fn(features, labels, mode):
    """Model function for a LR."""

    # Define logistic regression model using tf.keras.layers.
    logits = tf.keras.layers.Dense(2).apply(features['x'])

    # Calculate loss as a vector (to support microbatches in DP-SGD).
    vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    # Define mean of loss across minibatch (for reporting through tf.Estimator).
    scalar_loss = tf.reduce_mean(input_tensor=vector_loss)

    # Configure the training op (for TRAIN mode).
    if mode == tf.estimator.ModeKeys.TRAIN:

        if FLAGS.dpsgd:
            ledger = privacy_ledger.PrivacyLedger(
                population_size=FLAGS.training_data_size,
                selection_probability=(FLAGS.batch_size / FLAGS.training_data_size))

            # Use DP version of AdamOptimizer. Other optimizers are
            # available in dp_optimizer. Most optimizers inheriting from
            # tf.train.Optimizer should be wrappable in differentially private
            # counterparts by calling dp_optimizer.optimizer_from_args().
            # Setting num_microbatches to None is necessary for DP and
            # per-example gradients
            optimizer = dp_optimizer.DPAdamGaussianOptimizer(
                l2_norm_clip=FLAGS.l2_norm_clip,
                noise_multiplier=FLAGS.noise_multiplier,
                num_microbatches=None,
                ledger=ledger,
                learning_rate=FLAGS.learning_rate)
            training_hooks = [
                EpsilonPrintingTrainingHook(ledger)
            ]
            opt_loss = vector_loss
        else:
            optimizer = AdamOptimizer(learning_rate=FLAGS.learning_rate)
            training_hooks = []
            opt_loss = scalar_loss

        global_step = tf.compat.v1.train.get_global_step()
        train_op = optimizer.minimize(loss=opt_loss, global_step=global_step)
        # In the following, we pass the mean of the loss (scalar_loss) rather than
        # the vector_loss because tf.estimator requires a scalar loss. This is only
        # used for evaluation and debugging by tf.estimator. The actual loss being
        # minimized is opt_loss defined above and passed to optimizer.minimize().
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=scalar_loss,
                                          train_op=train_op,
                                          training_hooks=training_hooks)

    # Add evaluation metrics (for EVAL mode).
    elif mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            'accuracy':
                tf.compat.v1.metrics.accuracy(
                    labels=labels,
                    predictions=tf.argmax(input=logits, axis=1))
        }

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=scalar_loss,
                                      eval_metric_ops=eval_metric_ops)


def wrangle_the_data(data):

    # education
    data.drop(['education-num'], 1, inplace=True) # We can drop education-num, education is the same

    # age
    data['age'] = pd.cut(data.age, range(0, 105, 10), right=False)
    data['age'] = data['age'].astype("category").cat.codes

    # hours-per-week
    data['hours-per-week'] = pd.cut(data.age, range(0, 100, 10), right=False)
    data['hours-per-week'] = data['hours-per-week'].astype("category").cat.codes

    # workclass
    data['workclass'].replace([" Without-pay", " Never-worked"], "Jobless", inplace=True)
    data['workclass'].replace([" State-gov", " Federal-gov", " Local-gov"], "Govt", inplace=True)
    data['workclass'].replace([" Self-emp-not-inc", " Self-emp-inc"], "Self-emp", inplace=True)

    # marital status
    data['marital-status'].replace([" Married-AF-spouse"," Married-civ-spouse"," Married-spouse-absent"], "Married", inplace=True)
    data['marital-status'].replace([" Divorced"," Separated"," Widowed"," Never-married"], "Not-Married", inplace=True)

    # country
    data['native-country'].replace([" Canada"," Cuba"," Dominican-Republic", " El-Salvador", " Guatemala", " Haiti", " Honduras", " Jamaica", " Mexico", " Nicaragua", " Outlying-US(Guam-USVI-etc)", " Puerto-Rico", " Trinadad&Tobago", " United-States"], "North-America", inplace=True)
    data['native-country'].replace([" Cambodia", " China", " Hong", " India", " Iran", " Japan", " Laos", " Philippines", " Taiwan", " Thailand", " Vietnam"], "Asia", inplace=True)
    data['native-country'].replace([" Columbia", " Ecuador", " Peru"], "South-America", inplace=True)
    data['native-country'].replace([" England", " France", " Germany", " Greece", " Holand-Netherlands", " Hungary", " Ireland", " Italy", " Poland", " Portugal", " Scotland", " Yugoslavia"], "Europe", inplace=True)
    data['native-country'].replace([" South", " ?"], "Other", inplace=True)

    # Y
    data['salary'] = data['salary'].astype("category").cat.codes

    #print(data.dtypes.sample(10))
    return data.iloc[:,:-1], data['salary']


def get_uci_data():
    #data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data')
    #test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test')
    column_names = ['age', 'workclass', 'fnlwgt', 'education',
                    'education-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain',
                    'capital-loss', 'hours-per-week', 'native-country','salary']
    data = pd.read_csv('./data/adult.data', header=None)
    data.columns = column_names
    test = pd.read_csv('./data/adult.test', header=None, skiprows=1)
    test.columns = column_names

    X, Y = wrangle_the_data(data)
    test_X, test_Y = wrangle_the_data(test)

    X = pd.get_dummies(X)
    Y = np.array(Y, dtype=np.int32)
    test_X = pd.get_dummies(X)
    test_Y = np.array(Y, dtype=np.int32)
    X, test_X = X.align(test_X, join='left', axis=1)

    # You should pass df.values instead of df to tensorflow functions.
    return X.values, Y, test_X.values, test_Y


def main(unused_argv):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Load training and test data.
    train_data, train_labels, test_data, test_labels = get_uci_data()

    # Instantiate the tf.Estimator.
    lr_classifier = tf.estimator.Estimator(model_fn=lr_model_fn, model_dir=FLAGS.model_dir)   # <= SUDDENLY STOPPED WORKING!!!

    # Create tf.Estimator input functions for the training and test data.
    train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={'x': train_data},
        y=train_labels,
        batch_size=FLAGS.batch_size,
        num_epochs=FLAGS.epochs,
        shuffle=True)
    eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={'x': test_data},
        y=test_labels,
        num_epochs=1,
        shuffle=False)

    # Approximate a good learning rate with fewer rounds
    # Use the best rate for evaluating the rest of the hyperparameters
    evaluate_lr = False
    if evaluate_lr:
        print('---------------------- Learning rate -------------------------')
        #hparam = [.1, .05, .025, .0125, .01, .0075, .005, .001]
        hparam = [.002375, .00225, .002125]
        for i in range(0,3):
            FLAGS.learning_rate = hparam[i]
            print('\n# Learning rate %s' % hparam[i])
            # Training loop.
            accuracy_arr = []
            sys.stdout.write('e'+str(i)+'=[')
            steps_per_epoch = FLAGS.training_data_size // FLAGS.batch_size / 10
            for epoch in range(1, 3*FLAGS.epochs + 1):
                # Train the model for one epoch.
                lr_classifier.train(input_fn=train_input_fn, steps=steps_per_epoch)
                # Evaluate the model and print results
                eval_results = lr_classifier.evaluate(input_fn=eval_input_fn)
                test_accuracy = eval_results['accuracy']
                accuracy_arr.append(test_accuracy)
            print(']\na'+str(i)+' =', accuracy_arr)
        return

    if False:
        FLAGS.learning_rate = .002125
        print('\n---------------------- Clipping norm -------------------------')
        hparam = [0.8,1.0,1.2]
        for i in range(0,3):
            FLAGS.l2_norm_clip = hparam[i]
            print('\n# Clipping norm %s' % hparam[i])
            # Training loop.
            accuracy_arr = []
            sys.stdout.write('e'+str(i)+'=[')
            steps_per_epoch = FLAGS.training_data_size // FLAGS.batch_size / 10
            for epoch in range(1, 10*FLAGS.epochs + 1):
                # Train the model for one epoch.
                lr_classifier.train(input_fn=train_input_fn, steps=steps_per_epoch)
                # Evaluate the model and print results
                eval_results = lr_classifier.evaluate(input_fn=eval_input_fn)
                test_accuracy = eval_results['accuracy']
                accuracy_arr.append(test_accuracy)
            print(']\na'+str(i)+' =', accuracy_arr)

    if False:
        print('\n---------------------- Noise multiplier -------------------------')
        hparam = [1.5,2.0,4.0]
        for i in range(0,3):
            FLAGS.noise_multiplier = hparam[i]
            print('\n# Noise multiplier %s' % hparam[i])
            # Training loop.
            accuracy_arr = []
            sys.stdout.write('e'+str(i)+'=[')
            steps_per_epoch = FLAGS.training_data_size // FLAGS.batch_size / 10
            #for epoch in range(1, 10*FLAGS.epochs + 1):
            for epoch in range(1, FLAGS.epochs + 1):
                # Train the model for one epoch.
                lr_classifier.train(input_fn=train_input_fn, steps=steps_per_epoch)
                # Evaluate the model and print results
                eval_results = lr_classifier.evaluate(input_fn=eval_input_fn)
                test_accuracy = eval_results['accuracy']
                accuracy_arr.append(test_accuracy)
            print(']\na'+str(i)+' =', accuracy_arr)

    if False:
        print('\n---------------------- Batch size -------------------------')
        hparam = [16, 32, 64]
        for i in range(0,3):
            FLAGS.batch_size = hparam[i]
            print('\n# Batch size %s' % hparam[i])
            # Training loop.
            accuracy_arr = []
            sys.stdout.write('e'+str(i)+'=[')
            steps_per_epoch = FLAGS.training_data_size // FLAGS.batch_size / 10
            #for epoch in range(1, 10*FLAGS.epochs + 1):
            for epoch in range(1, FLAGS.epochs + 1):
                # Train the model for one epoch.
                lr_classifier.train(input_fn=train_input_fn, steps=steps_per_epoch)
                # Evaluate the model and print results
                eval_results = lr_classifier.evaluate(input_fn=eval_input_fn)
                test_accuracy = eval_results['accuracy']
                accuracy_arr.append(test_accuracy)
            print(']\na'+str(i)+' =', accuracy_arr)


if __name__ == '__main__':

    print('Started')
    app.run(main)
