# Notebooks Description

## finetuning_xgboost_lgbm.ipynb
GridSearchCV to find the optimal hyperparameters of our models

## merge_date.ipynb
Merge all data (weather, strike intensity, champselyseesdata, conventiondata...) into one big dataframe

## prophet_baseline.ipynb
Result obtained with Fbprophet & NeuralProphet

## WaveNet.ipynb
Inspired from this notebook : https://github.com/JEddy92/TimeSeries_Seq2Seq/blob/master/notebooks/TS_Seq2Seq_Conv_Full_Exog.ipynb
Convolutional sequence-to-sequence neural network modeled after WaveNet with exogenous features

## Pipeline.ipynb
Workflow's Structure :
- Fill missing data
- Features Engineering
- Stacking models strategy and comparison with the baseline

## train_and_pred.ipynb
If you run this notebook, you will make the final training of our models and you will also make the prediction
and generate the submissions csv file.