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

import math
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

from sklearn.model_selection import train_test_split

AdamOptimizer = tf.compat.v1.train.AdamOptimizer



FLAGS = flags.FLAGS
try:
    flags.FLAGS['dpsgd']
except KeyError:
    flags.DEFINE_boolean(
        'dpsgd', True, 'If True, train with DP-SGD. If False, train with vanilla SGD.')
    flags.DEFINE_float('learning_rate', .05, 'Learning rate for training')
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
        #print('For delta=1e-5, the current epsilon is: %.2f' % eps)
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

    # Leave: num, exang, oldpeak, cp, slope, sex and age
    data.drop(['trestbps','chol','fbs','restecg','thalach','ca','thal'], 1, inplace=True) # We can drop education-num, education is the same

    # age
    key, bucket = 'age', 10
    #describe(data[key],bins)
    data[key] = pd.cut(data[key], range(math.floor(data[key].min()), math.ceil(data[key].max())+1, bucket), right=False)

    # Categorical
    for key in ['sex', 'exang', 'cp', 'slope']:
        data[key] = data[key].astype('category')

    # Y: (angiographic disease status: Values 0, 1, 2, 3, 4)
    data['num'] = data['num'].astype('bool').astype('int')

    return data.iloc[:,:-1], data['num']


def get_uci_hd_data():
    column_names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']
    #data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data')
    data = pd.read_csv('../data/processed.cleveland.data', header=None)
    data.columns = column_names

    X, Y = wrangle_the_data(data)
    X = pd.get_dummies(X)
    X, X_test, Y, Y_test = train_test_split(X,Y,test_size=0.2,train_size=0.8)

    return X.values, np.array(Y, dtype=np.int32), X_test.values, np.array(Y_test, dtype=np.int32)


def main(unused_argv):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    hparams = ["learning_rate", "l2_norm_clip", "noise_multiplier", "batch_size"]
    hvalues = [[.5, .1, .05, .01, .005, .001],[0.5,1.5,2.5],[1.5,2.5,3.0],[32, 128]]

    for j in range(1,4):

        FLAGS.learning_rate = .1
        FLAGS.l2_norm_clip = 1.0
        FLAGS.noise_multiplier = 2.0
        FLAGS.batch_size = 64

        print('----------------------' + hparams[j] + '-------------------------')
        for i in range(0,len(hvalues[j])):

            if hparams[j]=='learning_rate':
                FLAGS.learning_rate = hvalues[j][i]
            elif hparams[j]=='l2_norm_clip':
                FLAGS.l2_norm_clip = hvalues[j][i]
            elif hparams[j]=='noise_multiplier':
                FLAGS.noise_multiplier = hvalues[j][i]
            else:
                FLAGS.batch_size = hvalues[j][i]
            print('\n# %s %s' % (hparams[j], hvalues[j][i]))

            # Load training and test data.
            train_data, train_labels, test_data, test_labels = get_uci_hd_data()

            # Instantiate the tf.Estimator.
            lr_classifier = tf.estimator.Estimator(model_fn=lr_model_fn, model_dir=FLAGS.model_dir)

            # Create tf.Estimator input functions for the training and test data.
            train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
                x={'x': train_data}, y=train_labels, batch_size=FLAGS.batch_size, num_epochs=FLAGS.epochs, shuffle=True)
            eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
                x={'x': test_data},y=test_labels,num_epochs=1,shuffle=False)

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
                if (test_accuracy<.3):
                    break
                if (test_accuracy>.868):
                    FLAGS.learning_rate =  hvalues[j][i]/2
            print(']\na'+str(i)+' =', accuracy_arr)


if __name__ == '__main__':

    app.run(main)
