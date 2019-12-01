import time
import datetime

import pandas as pd
import numpy as np
import tensorflow as tf

from utils import fc_layer, bilinear_layer, rmse
from extract_features_for_model import extract_features


class GameRating:
    def __init__(self,
                 file='',
                 batch_size=32,
                 num_epochs=None,
                 iterations=5e4,
                 learning_rate=5e-4):
        # original data
        self.file = file
        # hyper-parameters
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.iterations = int(iterations)
        self.learning_rate = learning_rate
        # initialization of tensors
        with tf.variable_scope(name_or_scope='init'):
            # indicator of training
            self.training = tf.placeholder(tf.bool, name='training')
            # input tensors
            self._semantic = tf.placeholder(tf.float32, [None, 20, 768], name='semantic')
            self._features = tf.placeholder(tf.float32, [None, 54], name='features')
            # ground truth
            self._playtime = tf.placeholder(tf.float32, [None, 1], name='playtime')

    @staticmethod
    def pre_process(data, to_standardize=False):
        # define keys of 'genres' and 'categories'
        keys = ['price', 'purchase_date', 'release_date', 'total_positive_reviews', 'total_negative_reviews', 'Action',
                'Adventure', 'Animation & Modeling', 'Audio Production', 'Casual', 'Design & Illustration',
                'Early Access', 'Free to Play', 'Gore', 'Indie', 'Massively Multiplayer', 'Nudity', 'RPG', 'Racing',
                'Sexual Content', 'Simulation', 'Sports', 'Strategy', 'Utilities', 'Violent',
                'Captions available', 'Co-op', 'Commentary available', 'Cross-Platform Multiplayer',
                'Full controller support', 'In-App Purchases', 'Includes Source SDK', 'Includes level editor',
                'Local Co-op', 'Local Multi-Player', 'MMO', 'Multi-player', 'Online Co-op', 'Online Multi-Player',
                'Partial Controller Support', 'Remote Play on Phone', 'Remote Play on TV', 'Remote Play on Tablet',
                'Shared/Split Screen', 'Single-player', 'Stats', 'Steam Achievements', 'Steam Cloud',
                'Steam Leaderboards', 'Steam Trading Cards', 'Steam Workshop', 'SteamVR Collectibles', 'VR Support',
                'Valve Anti-Cheat enabled']
        # perform one-hot encoding
        genres_dummies = data['genres'].str.get_dummies(',')
        categories_dummies = data['categories'].str.get_dummies(',')
        # replace 'genres' by 'genres_dummies', as well as replace 'categories' with 'categories_dummies'
        data = pd.concat([data, genres_dummies, categories_dummies], axis=1)
        # deal with the missing key(s)
        for k in keys:
            if data.get(k) is None:
                data[k] = 0
        # sort columns by keys
        data = data[keys]

        # convert a date into a timestamp
        def date2timestamp(schema='', date=''):
            # define a mapping from english months to numbers
            month2num = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
                         'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
            if type(date) is str:
                # convert english months into numbers
                for month, number in month2num.items():
                    date = date.replace(month, number)
                # return
                return time.mktime(datetime.datetime.strptime(date, schema).timetuple())
            else:
                # return
                return 1438531200.0

        # replace 'purchase_date'
        data['purchase_date'] = data['purchase_date'].apply(lambda d: date2timestamp(schema='%m %d, %Y', date=d))

        # replace 'release_date'
        data['release_date'] = data['release_date'].apply(lambda d: date2timestamp(schema='%d %m, %Y', date=d))

        # fill NANs
        data['total_positive_reviews'] = data['total_positive_reviews'].fillna(value=0)
        data['total_negative_reviews'] = data['total_negative_reviews'].fillna(value=0)

        # perform standardization
        if to_standardize:
            # reference table
            ref = {'price': (0.0, 15999900.0), 'purchase_date': (1438531200.0, 1567785600.0),
                   'release_date': (1156262400.0, 1564416000.0), 'total_positive_reviews': (0.0, 440902.0),
                   'total_negative_reviews': (0.0, 436046.0)}
            # perform min-max standardization
            for key, (minimum, maximum) in ref.items():
                data[key] = np.clip((data[key] - minimum) / (maximum - minimum), 0.0, 1.0)

        # return
        return data

    def naive_model(self):
        with tf.variable_scope(name_or_scope='game_rating'):
            # pre-processing layers
            fc_0 = fc_layer(self._features, 128, training=self.training, name='fc_0')
            # bi-linear layers
            bi_0 = bilinear_layer(fc_0, 128, training=self.training, name='bi_0')
            # output layer
            playtime_output = fc_layer(bi_0, 1, training=self.training, name='playtime_output')
            # return
            return playtime_output

    def model(self):
        with tf.variable_scope(name_or_scope='game_rating'):
            # pre-processing layers
            semantic_fc_0 = fc_layer(tf.layers.flatten(self._semantic), 512, training=self.training,
                                     name='semantic_fc_0')
            features_fc_0 = fc_layer(self._features, 128, training=self.training, name='features_fc_0')
            # bi-linear layers
            semantic_bi_0 = bilinear_layer(semantic_fc_0, 512, training=self.training, name='semantic_bi_0')
            features_bi_0 = bilinear_layer(features_fc_0, 128, training=self.training, name='features_bi_0')
            # post-processing layers
            semantic_fc_1 = fc_layer(semantic_bi_0, 128, training=self.training, name='semantic_fc_1')
            features_fc_1 = fc_layer(features_bi_0, 128, training=self.training, name='features_fc_1')
            # merge user features and movie features by an add_n operation, following by a fully-connected layer
            merge = tf.multiply(semantic_fc_1, features_fc_1, name='merge')
            # merge = tf.add_n([semantic_fc_1, features_fc_1], name='merge')
            # output layer
            playtime_output = fc_layer(merge, 1, training=self.training, name='playtime_output')
            # return
            return playtime_output

    def train(self):
        # load data
        train = pd.read_csv('train.csv')
        train['tags'] = train['tags'].apply(lambda x: x.replace(',', ' '))
        train, validation = train[:285], train[285:]
        test = pd.read_csv('test.csv')
        test['tags'] = test['tags'].apply(lambda x: x.replace(',', ' '))

        # extract features
        features = self.pre_process(train, True)
        semantic = np.array(extract_features(train['tags'], 20))
        playtime = np.array(train.get('playtime_forever')).reshape((-1, 1))

        fv = self.pre_process(validation, True)
        sv = np.array(extract_features(validation['tags'], 20))
        pv = np.array(validation.get('playtime_forever')).reshape((-1, 1))

        ft = self.pre_process(test, True)
        st = np.array(extract_features(test['tags'], 20))

        # process the schema of testing set
        test['playtime_forever'] = 0
        test = test[['id', 'playtime_forever']]
        test.set_index('id', inplace=True)

        # generate model
        optimal = np.inf
        output = self.model()

        with tf.variable_scope(name_or_scope='optimizer'):
            # loss function
            total_loss = tf.reduce_mean(rmse(self._playtime, output), name='total_loss')

            # optimizer
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_ops = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(total_loss)

        config = tf.ConfigProto()

        print('Start training')
        with tf.Session(config=config) as sess:
            # initialization
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            # data set
            queue = tf.train.slice_input_producer([semantic, features, playtime],
                                                  num_epochs=self.num_epochs, shuffle=True)
            s_batch, f_batch, y_batch = tf.train.batch(queue, batch_size=self.batch_size, num_threads=1,
                                                       allow_smaller_final_batch=False)

            # enable coordinator
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)

            try:
                for i in range(self.iterations):
                    s, f, y = sess.run([s_batch, f_batch, y_batch])

                    _, loss = sess.run([train_ops, total_loss], feed_dict={self.training: True,
                                                                           self._semantic: s, self._features: f,
                                                                           self._playtime: y})

                    if i % 100 == 0:
                        print('iteration %d: loss = %f' % (i, loss))
                    if i % 500 == 0:
                        result, loss = sess.run([output, total_loss], feed_dict={self.training: False,
                                                                                 self._semantic: sv, self._features: fv,
                                                                                 self._playtime: pv})
                        print('validation loss = %f' % loss)
                        if loss < optimal:
                            optimal = loss
                            result = sess.run(output, feed_dict={self.training: False,
                                                                 self._semantic: st, self._features: ft})
                            result = np.reshape(result, newshape=(-1,))

                            # dump csv file
                            test['playtime_forever'] = result
                            test.to_csv('submission_%d_%f.csv' % (i, optimal))

            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')

            finally:
                coord.request_stop()

            coord.join(threads)

            # print('Start testing')
            # # extract features
            # features = self.pre_process(test, True)
            # semantic = np.mean(np.array(extract_features(test['tags'], 20)), axis=1)
            #
            # # calculate result
            # result = sess.run(output, feed_dict={self.training: False,
            #                                      self._semantic: semantic, self._features: features})
            # result = np.reshape(result, newshape=(-1,))
            #
            # # dump csv file
            # test['playtime_forever'] = result
            # test.to_csv('prediction.csv')


if __name__ == '__main__':
    model = GameRating(batch_size=16, iterations=1e5)
    model.train()
