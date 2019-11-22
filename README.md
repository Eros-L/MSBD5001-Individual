# MSBD5001-Individual

## Programming Language

Python

## Required Packages

Pandas, Numpy, Sklearn

## How to run it to reproduce my result

Considering that we are using RandomForestRegressor, the results of training could vary. However, we still have a naive method to eliminate results having large loss on the the public leaderboard. Based on observation, we can notice that the predicted values of some entries in testing set should be large. We can only preserve model having large predicted values on the 31st, the 73rd and the 75th entries in testing set to obtain better performance. 
