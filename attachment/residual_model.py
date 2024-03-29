import time
import datetime
import pickle

import pandas as pd
import numpy as np
import tensorflow as tf

from utils import fc_layer, bilinear_layer, rmse

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
            self._features = tf.placeholder(tf.float32, [None, 370], name='features')
            # ground truth
            self._playtime = tf.placeholder(tf.float32, [None, 1], name='playtime')

    @staticmethod
    def pre_process(data, to_standardize=False, use_pca=None):
        # define keys of 'genres' and 'categories'
        keys = [ # original features
                 'price', 'purchase_year', 'purchase_month', 'purchase_day', 'release_year', 'release_month',
                 'release_day', 'total_positive_reviews', 'total_negative_reviews',
                 # 'price', 'purchase_date', 'release_date', 'total_positive_reviews', 'total_negative_reviews',
                 # genres
                 'Action_g', 'Adventure_g', 'Animation & Modeling_g', 'Audio Production_g', 'Casual_g',
                 'Design & Illustration_g', 'Early Access_g', 'Free to Play_g', 'Gore_g', 'Indie_g',
                 'Massively Multiplayer_g', 'Nudity_g', 'RPG_g', 'Racing_g', 'Sexual Content_g', 'Simulation_g',
                 'Sports_g', 'Strategy_g', 'Utilities_g', 'Violent_g',
                 # categories
                 'Captions available_c', 'Co-op_c', 'Commentary available_c', 'Cross-Platform Multiplayer_c',
                 'Full controller support_c', 'In-App Purchases_c', 'Includes Source SDK_c', 'Includes level editor_c',
                 'Local Co-op_c', 'Local Multi-Player_c', 'MMO_c', 'Multi-player_c', 'Online Co-op_c',
                 'Online Multi-Player_c', 'Partial Controller Support_c', 'Remote Play on Phone_c',
                 'Remote Play on TV_c', 'Remote Play on Tablet_c', 'Shared/Split Screen_c', 'Single-player_c',
                 'Stats_c', 'Steam Achievements_c', 'Steam Cloud_c', 'Steam Leaderboards_c', 'Steam Trading Cards_c',
                 'Steam Workshop_c', 'SteamVR Collectibles_c', 'VR Support_c', 'Valve Anti-Cheat enabled_c',
                 # tags
                 '1980s_t', "1990's_t", '2.5D_t', '2D_t', '3D_t', '3D Platformer_t', '3D Vision_t', '4 Player Local_t',
                 '4X_t', 'ATV_t', 'Action_t', 'Action RPG_t', 'Action-Adventure_t', 'Addictive_t', 'Adventure_t',
                 'Aliens_t', 'Alternate History_t', 'America_t', 'Animation & Modeling_t', 'Anime_t', 'Arcade_t',
                 'Arena Shooter_t', 'Artificial Intelligence_t', 'Assassin_t', 'Atmospheric_t', 'Audio Production_t',
                 'Automation_t', 'Base Building_t', 'Based On A Novel_t', 'Batman_t', 'Battle Royale_t',
                 "Beat 'em up_t", 'Beautiful_t', 'Benchmark_t', 'Bikes_t', 'Blood_t', 'Board Game_t', 'Building_t',
                 'Bullet Hell_t', 'Bullet Time_t', 'CRPG_t', 'Capitalism_t', 'Card Game_t', 'Cartoon_t', 'Cartoony_t',
                 'Casual_t', 'Cats_t', 'Character Action Game_t', 'Character Customization_t', 'Chess_t',
                 'Choices Matter_t', 'Choose Your Own Adventure_t', 'Cinematic_t', 'City Builder_t', 'Classic_t',
                 'Clicker_t', 'Co-op_t', 'Co-op Campaign_t', 'Colorful_t', 'Comedy_t', 'Comic Book_t',
                 'Competitive_t', 'Conspiracy_t', 'Controller_t', 'Crafting_t', 'Crime_t', 'Crowdfunded_t',
                 'Cult Classic_t', 'Cute_t', 'Cyberpunk_t', 'Dark_t', 'Dark Comedy_t', 'Dark Fantasy_t', 'Dark Humor_t',
                 'Dating Sim_t', 'Demons_t', 'Design & Illustration_t', 'Destruction_t', 'Detective_t', 'Difficult_t',
                 'Dinosaurs_t', 'Diplomacy_t', 'Documentary_t', 'Dragons_t', 'Drama_t', 'Driving_t',
                 'Dungeon Crawler_t', 'Dungeons & Dragons_t', 'Dynamic Narration_t', 'Dystopian_t', 'Early Access_t',
                 'Economy_t', 'Education_t', 'Emotional_t', 'Epic_t', 'Episodic_t', 'Experience_t', 'Experimental_t',
                 'Exploration_t', 'FMV_t', 'FPS_t', 'Family Friendly_t', 'Fantasy_t', 'Fast-Paced_t',
                 'Female Protagonist_t', 'Fighting_t', 'First-Person_t', 'Flight_t', 'Free to Play_t', 'Funny_t',
                 'Futuristic_t', 'Game Development_t', 'Games Workshop_t', 'God Game_t', 'Gore_t',
                 'Gothic_t', 'Grand Strategy_t', 'Great Soundtrack_t', 'Gun Customization_t', 'Hack and Slash_t',
                 'Hacking_t', 'Hand-drawn_t', 'Heist_t', 'Hex Grid_t', 'Hidden Object_t', 'Historical_t', 'Horror_t',
                 'Horses_t', 'Hunting_t', 'Illuminati_t', 'Immersive Sim_t', 'Indie_t',
                 'Intentionally Awkward Controls_t', 'Interactive Fiction_t', 'Inventory Management_t',
                 'Investigation_t', 'Isometric_t', 'JRPG_t', 'Kickstarter_t', 'LGBTQ+_t', 'Lara Croft_t',
                 'Level Editor_t', 'Linear_t', 'Local Co-Op_t', 'Local Multiplayer_t', 'Logic_t', 'Loot_t',
                 'Lovecraftian_t', 'MMORPG_t', 'MOBA_t', 'Magic_t', 'Management_t', 'Mars_t',
                 'Martial Arts_t', 'Massively Multiplayer_t', 'Masterpiece_t', 'Mature_t', 'Mechs_t', 'Medieval_t',
                 'Memes_t', 'Metroidvania_t', 'Military_t', 'Minigames_t', 'Minimalist_t', 'Mod_t', 'Moddable_t',
                 'Motocross_t', 'Motorbike_t', 'Mouse only_t', 'Movie_t', 'Multiplayer_t',
                 'Multiple Endings_t', 'Music_t', 'Music-Based Procedural Generation_t', 'Mystery_t', 'Mythology_t',
                 'Narration_t', 'Nature_t', 'Naval_t', 'Ninja_t', 'Noir_t', 'Nonlinear_t',
                 'Nudity_t', 'Offroad_t', 'Old School_t', 'Online Co-Op_t', 'Open World_t', 'Parkour_t', 'Parody_t',
                 'Party-Based RPG_t', 'Perma Death_t', 'Philisophical_t', 'Physics_t', 'Pirates_t', 'Pixel Graphics_t',
                 'Platformer_t', 'Point & Click_t', 'Political_t', 'Politics_t', 'Post-apocalyptic_t',
                 'Procedural Generation_t', 'Programming_t', 'Psychedelic_t', 'Psychological_t',
                 'Psychological Horror_t', 'Puzzle_t', 'Puzzle-Platformer_t', 'PvE_t', 'PvP_t', 'Quick-Time Events_t',
                 'RPG_t', 'RPGMaker_t', 'RTS_t', 'Racing_t', 'Real Time Tactics_t', 'Real-Time_t',
                 'Real-Time with Pause_t', 'Realistic_t', 'Relaxing_t', 'Remake_t', 'Replay Value_t',
                 'Resource Management_t', 'Retro_t', 'Rhythm_t', 'Robots_t', 'Rogue-like_t', 'Rogue-lite_t',
                 'Romance_t', 'Rome_t', 'Sailing_t', 'Sandbox_t', 'Satire_t', 'Sci-fi_t', 'Science_t', 'Score Attack_t',
                 'Sequel_t', 'Sexual Content_t', "Shoot 'Em Up_t", 'Shooter_t', 'Short_t', 'Side Scroller_t',
                 'Silent Protagonist_t', 'Simulation_t', 'Singleplayer_t', 'Sniper_t', 'Snowboarding_t', 'Software_t',
                 'Souls-like_t', 'Soundtrack_t', 'Space_t', 'Space Sim_t', 'Spectacle fighter_t', 'Split Screen_t',
                 'Sports_t', 'Star Wars_t', 'Stealth_t', 'Steampunk_t', 'Story Rich_t', 'Strategy_t', 'Strategy RPG_t',
                 'Stylized_t', 'Submarine_t', 'Superhero_t', 'Supernatural_t', 'Surreal_t', 'Survival_t',
                 'Survival Horror_t', 'Swordplay_t', 'Tactical_t', 'Tactical RPG_t', 'Tanks_t', 'Team-Based_t',
                 'Tennis_t', 'Text-Based_t', 'Third Person_t', 'Third-Person Shooter_t', 'Thriller_t',
                 'Time Manipulation_t', 'Time Travel_t', 'Top-Down_t', 'Top-Down Shooter_t', 'Touch-Friendly_t',
                 'Tower Defense_t', 'Trading_t', 'Trading Card Game_t', 'Trains_t', 'Transhumanism_t',
                 'Transportation_t', 'Turn-Based_t', 'Turn-Based Combat_t', 'Turn-Based Strategy_t',
                 'Turn-Based Tactics_t', 'Twin Stick Shooter_t', 'Typing_t', 'Underground_t', 'Underwater_t',
                 'Unforgiving_t', 'Utilities_t', 'VR_t', 'Villain Protagonist_t', 'Violent_t', 'Visual Novel_t',
                 'Voxel_t', 'Walking Simulator_t', 'War_t', 'Wargame_t', 'Warhammer 40K_t', 'Western_t',
                 'World War I_t', 'World War II_t', 'Zombies_t', 'eSports_t']
        # perform one-hot encoding
        genres_dummies = data['genres'].str.get_dummies(',').rename(lambda x: x + '_g', axis='columns')
        categories_dummies = data['categories'].str.get_dummies(',').rename(lambda x: x + '_c', axis='columns')
        tags_dummies = data['tags'].str.get_dummies(',').rename(lambda x: x + '_t', axis='columns')

        # replace 'genres' by 'genres_dummies', as well as replace 'categories' with 'categories_dummies'
        data = pd.concat([data, genres_dummies, categories_dummies, tags_dummies], axis=1)

        # convert a date into a datetime
        def date2time(schema='', date=''):
            # define a mapping from english months to numbers
            month2num = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
                         'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
            # convert english months into numbers
            for month, number in month2num.items():
                date = date.replace(month, number)
            # return
            return datetime.datetime.strptime(date, schema)

        # replace 'purchase_date'
        data['purchase_date'] = data['purchase_date'].apply(lambda d: date2time(schema='%m %d, %Y', date=d))
        # replace 'release_date'
        data['release_date'] = data['release_date'].apply(lambda d: date2time(schema='%d %m, %Y', date=d))

        # decompose 'purchase_date'
        data['purchase_year'] = data['purchase_date'].apply(lambda d: d.year)
        data['purchase_month'] = data['purchase_date'].apply(lambda d: d.month)
        data['purchase_day'] = data['purchase_date'].apply(lambda d: d.day)
        # decompose 'release_date'
        data['release_year'] = data['release_date'].apply(lambda d: d.year)
        data['release_month'] = data['release_date'].apply(lambda d: d.month)
        data['release_day'] = data['release_date'].apply(lambda d: d.day)

        # fill NANs
        data['total_positive_reviews'] = data['total_positive_reviews'].fillna(value=0)
        data['total_negative_reviews'] = data['total_negative_reviews'].fillna(value=0)

        # deal with the missing key(s)
        for k in keys:
            if data.get(k) is None:
                data[k] = 0
        # sort columns by keys
        data = data[keys]

        # perform pca
        if use_pca is not None:
            try:
                # attempt to load a PCA model from disk
                with open('save/pca.pickle', 'rb') as f:
                    pca = pickle.load(f)
            except FileNotFoundError:
                # generate a new PCA model
                pca = PCA(n_components=use_pca)
                # fit the model with 'data'
                pca.fit(data)
                # save the model
                with open('save/pca.pickle', 'wb') as f:
                    pickle.dump(pca, f)
            # reduce the dimensionality of data
            data = pca.transform(data)

        # perform standardization
        if to_standardize:
            try:
                # attempt to load a scaler model from disk
                with open('save/scaler.pickle', 'rb') as f:
                    scaler = pickle.load(f)
            except FileNotFoundError:
                # generate a new scaler
                scaler = StandardScaler()
                # fit data
                scaler.fit(data)
                # save the scaler
                with open('save/scaler.pickle', 'wb') as f:
                    pickle.dump(scaler, f)
            # apply gaussian standardization
            data = scaler.transform(data)

        # return
        return data

    def model(self):
        with tf.variable_scope(name_or_scope='game_rating'):
            # pre-processing layers
            fc_0 = fc_layer(self._features, 128, training=self.training, name='fc_0')
            # bi-linear layers
            bi_0 = bilinear_layer(fc_0, 128, training=self.training, name='bi_0')
            # output layer
            playtime_output = fc_layer(bi_0, 1, training=self.training, name='playtime_output')
            # return
            return playtime_output

    def train(self, to_standardize=False, use_pca=None):
        # load data
        train = pd.read_csv('train.csv')
        train, validation = train[:285], train[285:]
        test = pd.read_csv('test.csv')

        # extract features
        features = self.pre_process(train, to_standardize, use_pca)
        playtime = np.array(train.get('playtime_forever')).reshape((-1, 1))

        fv = self.pre_process(validation, to_standardize, use_pca)
        pv = np.array(validation.get('playtime_forever')).reshape((-1, 1))

        ft = self.pre_process(test, to_standardize, use_pca)

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
            queue = tf.train.slice_input_producer([features, playtime],
                                                  num_epochs=self.num_epochs, shuffle=True)
            x_batch, y_batch = tf.train.batch(queue, batch_size=self.batch_size, num_threads=1,
                                              allow_smaller_final_batch=False)

            # enable coordinator
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)

            try:
                for i in range(self.iterations):
                    x, y = sess.run([x_batch, y_batch])

                    _, loss = sess.run([train_ops, total_loss], feed_dict={self.training: True,
                                                                           self._features: x, self._playtime: y})

                    if i % 100 == 0:
                        print('iteration %d: loss = %f' % (i, loss))
                    if i % 500 == 0:
                        result, loss = sess.run([output, total_loss], feed_dict={self.training: False,
                                                                                 self._features: fv,
                                                                                 self._playtime: pv})
                        print('validation loss = %f' % loss)
                        if loss < optimal or i % 10000 == 0:
                            result = sess.run(output, feed_dict={self.training: False, self._features: ft})
                            result = np.reshape(result, newshape=(-1,))

                            # dump csv file
                            test['playtime_forever'] = result
                            test.to_csv('save/submission_%d_%f.csv' % (i, loss))

                            if loss < optimal:
                                optimal = loss

            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')

            finally:
                coord.request_stop()

            coord.join(threads)


if __name__ == '__main__':
    model = GameRating(batch_size=16, iterations=1e5, learning_rate=1e-4)

    model.train(to_standardize=True, use_pca=None)
