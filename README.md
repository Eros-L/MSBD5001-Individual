# MSBD5001-Individual

## Programming Language

Python

## Required Packages

Pandas, Numpy, Sklearn

## How to run it to reproduce my result

Considering that we are using RandomForestRegressor, the results of training could vary. However, we still have a naive method to eliminate results having large loss on the the public leaderboard. Based on observation, we can notice that the predicted values of some entries in testing set should be large. We can only preserve model having large predicted values on the 31st, the 73rd and the 75th entries in testing set to obtain better performance. 

---

## Solution

All of the related scripts could be found on [here](https://github.com/Eros-L/MSBD5001-Individual/tree/master/attachment). 

### Data Exploration

There are 11 columns in the data set. These columns are 'id', 'playtime_forever', 'is_free', 'price', 'genres', 'categories', 'tags', 'purchase_date', 'release_date', 'total_positive_reviews', 'total_negative_reviews' respectively. For our project, 'id' is trivial and 'playtime_forever' is the ground truth. Therefore, we should focus on the remaining columns. 

Firstly, we should interpret the correlation table. In this step, we ignore 'genres', 'categories' and 'tags', then convert 'purchase_date' and 'release_date' into timestamps. 

|key|playtime_forever|is_free|price|total_positive_reviews|total_negative_reviews|purchase_date|release_date|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|playtime_forever|1.000|-0.021|0.024|0.395|0.393|-0.140|0.106|
|is_free|-0.021|1.000|-0.024|-0.064|-0.013|-0.146|0.083|
|price|0.024|-0.024|1.000|0.142|0.003|-0.047186|-0.087|
|total_positive_reviews|0.395|-0.064|0.142|1.000|0.693|-0.097|-0.088|
|total_negative_reviews|0.393|-0.013|0.003|0.693|1.000|-0.038|0.059|
|purchase_date|-0.140|-0.146|-0.047|-0.097|-0.038|1.000|0.300|
|release_date|0.106|0.083|-0.087|-0.088|0.059|0.300|1.000|

We can notice that 'playtime_forever' is highly correlated with 'total_positive_reviews' and 'total_negative_reviews'. Additionally, 'purchase_date' and 'release_date' also matter a lot. However, 'is_free' and 'price' are relatively not so important. Considering that 'price' can indicate whether a game is free, we could drop 'is_free' in our project. 

### Feature Engineering

In the feature engineering part, we need to convert all columns with type of string to numbers. Firstly, we can convert 'purchase_date' and 'release_date' into a tuple of year, month and day. After that, we need to process 'genres', 'categories' and 'tags'. 

Considering 'genres' only has 20 unique values and 'categories' only has 29 unique values, we could simply perform one-hot encoding on these two columns. However, 'tags' has 312 diffent values, making it unsuitable to adopt one-hot encoding only. In this case, we first perform one-hot encoding on 'tags' and then use PCA to enrich information of 'tags'. 

### GBDT Model

Gradient Boosted Decision Trees (GBDT) is a machine learning algorithm that iteratively constructs an ensemble of weak decision tree learners through boosting. 

Considering the GBDT Model, we first apply the aforementioned feature engineering. Then, we perform min-max normalization on our data set. In addition, we also use a 5-fold CV to select hyperparameters. For implementation, we use LightGBM, which is a gradient boosting framework that uses tree based learning algorithms. 

### Random Forest Model

Random Forest is also popular in Kaggle competition. We are able to examinate our feature engineering quickly with the use of Random Forest. 

Roughly, we do the same thing as we do with the GBDT Model, only except that we use scikit-learn for programming instead. 

### Residual Network

Residual Neural Network is widely adopted in deep learning. For our project, we can simply build a network with the following architecture. 

![](https://raw.githubusercontent.com/Eros-L/MSBD5001-Individual/attachment/residual_model.png)

Batch Normalization and Dropout are used. 

### Transfer Learning

In this part, we have tried a two-branch residual network with NLP feature extracting. Specifically, we do not perform one-hot encoding on 'tags'. Instead, we use a feature extracting network to convert it into vectors. 

For feature extracting, we use a pretrained network called BERT. BERT is a method of pre-training language representations, meaning that we train a general-purpose "language understanding" model on a large text corpus (like Wikipedia), and then use that model for downstream NLP tasks that we care about (like question answering). BERT outperforms previous methods because it is the first unsupervised, deeply bidirectional system for pre-training NLP. More details could be found on [Bert's Github](https://github.com/google-research/bert). 

After that, we build a network with the following architecture. 

![](https://raw.githubusercontent.com/Eros-L/MSBD5001-Individual/attachment/transfer_learning_model.png)

Batch Normalization and Dropout are used. 
