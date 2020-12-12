# bcg_gamma_datathon

We are going to detail our approach.
We decided to use LGBMRegressor & XGBRegressor and stack them using the appropriate stacking technique.

# Features Extraction
Features collected: (from 2014 to 2020)
- Weather (description, rain, snow, temperature, temperaturelow, thunder, humidity, wind, pression)
- Quarantine Dates (first one (2020-03-17 / 2020-05-11) , second one (2020-10-30 / 2020-11-28) and third one
(2020-11-28 / 2020-12-15))
- Curfew
- RATP/SNCF strike intensity
- Time feature
- Holidays

# Features engineering
## Weather
We one hot encoded the weather description and we only kept description's names appearing more than 40 times
We have both wind speed & wind direction, we decided to make two relevant features : The horizontal component
of the wind (Wx) and the vertical component of the wind (Wy)
For the rest of the weather features, we left them as they were.

## Quarantine Dates
We decided to label encode this data. Why? Because there is an orderly relationship between them.
The first lockdown was more strict than the second one, and this latter was more strict than the third one.
first lockdown : -3
second lockdown : -2
third lockdown : -1
no lockdown : 0

## Curfew
A simple one hot encoding (1 for curfew and 0 for no curfew)

## Strike Intensity
We haven't modified them

## Holidays
Two features:
Public Holiday : one hot encoding (1 vs 0)
School Vacation : one hot encoding (1 vs 0)

## Time features
### Hours
We did a series of fourier (because of the periodicity) and we obtain two components:
Cosinus Hours & Sinus Hours
We choose to do that because the one hot encoded can let us think that for instance 10 pm & 11pm are not close.

### Dayofweek
One Hot encoding

### Quarter
One Hot encoding

### Month
One Hot Enconding

### DayOfYear
Same as hours, we did a series of fourier.

### WeekOfYear
series of fourier.

### Dayofmonth
One hot encoding

# Models & Finetuning
To validate our models & finetuning, we use the classical time series split.
Validation 1 : 2020-11-29 => 2020-12-04
Validation 2 : 2020-12-04 => 2020-12-10

## Models
We compared many machine learning models (thank to the Pycaret Library), one deep learning model (WaveNet) &
fbprophet/Sarimax. We choose the one that perform best on our validation set, and the winners are :
LGBMRegressor & XGBRegressor (which honestly doesn't surprise us as they are widely used during this kind
of challenges)

## Baseline
As baseline, we choose to do a naive prediction which consist of predicting the value of the previous week.
We can notice that some models didn't manage to beat this baseline which shows that sometimes we don't need
to use a Bazooka to tackle data science problems.

## Starting date for Training
For each street and each target (hourly flow or occupancy rate), we made some experiments on how the starting date
for training can influence the performance of the models. 
It appeared to us that training the models on data dating before 2020 only deteriorated the performance.
Our thought is that because of all what happenedd during this year (2020), we can't rely on past year.

## Adding last week value as features
For some "Arc" such as Convention & SaintPeres, adding the past value (rolling on one week) improves the performance.
Nevertheless, it doesn't seem to work on Champs Elysées.

## Hyperparameters finetuning
We run a classic grid search with time series cross validation in order to find the best hyperparameters for our both models (LGBMRegressor/XGBRegressor).

## Stacking Models
We choose to run some experiments on how are we going to stack these models. We came up with two strategy regarding
the predictions of each of the models :
- Min/Max Strategy : During the night, take the minimum predicted value / During the day, take the maximum predicted
value
- Average Strategy : Take the average prediction.
For each street & target features, we choose one of these two strategies. 

# Models & Finetuning : RECAP
**Summary of the finetuning (hyperparameters, choose the best date where to start training, which features to add ..)**

**Champs Elysées**

*For q :*
- Use data from October 2020 and take only features that we extract in the previous section
- Xgboost : {learning_rate = 0.15, subsample=0.8, n_estimators=300}
- LGBM : {learning_rate = 0.15, subsample=0.8, n_estimators=300} 

*Winning Strategy* : **AVERAGE STRATEGY**

*For k :*
- Use date from August 2020 and take only features that we extract in the previous section
- Xgboost : {n_estimators=300, subsample = 0.6, min_child_weight=5,  max_depth=4, random_state=27}
- LGBM : {n_estimators=300, subsample=0.8, num_leaves=25, learning_rate=0.15, random_state=27} 

*Winning Strategy* : **MIN/MAX STRATEGY**

**Convention**

*For q :*
- Use date from February 2020 and take only features that we extract in the previous section and add last week q value as feature
- Xgboost : {n_estimators=300, max_depth=4, min_child_weight=5, subsample=0.6, random_state=27}
- LGBM : {n_estimators=300, subsample=0.8, max_depth=4, colsample_bytree=0.8, subsample_freq=2, num_leaves=15,   random_state=27} 

*Winning Strategy* : **AVERAGE STRATEGY**

*For k :*
- Use date from February 2020 and take only features that we extract in the previous section and add last week k value as feature
- Xgboost : {random_state = 27, colsample_bytree=0.7, max_depth=4, min_child_weight=5, subsample=0.8,n_estimators=300}
- LGBM : {subsample=0.8, subsample_freq=2, colsample_bytree=0.8, num_leaves=15, n_estimators=300, random_state=27} 

*Winning Strategy* : **AVERAGE STRATEGY**

**Saint Peres**

*For q :*
- Use date from January 2020 and take only features that we extract in the previous section and add last week q value as feature
- Xgboost : {random_state = 27, max_depth=8, min_child_weight=5, n_estimators=300}
- LGBM : {colsample_bytree=0.8, subsample=0.8, num_leaves=25, n_estimators=300, subsample_freq=1, random_state=27} 

*Winning Strategy* : **MIN/MAX STRATEGY**

*For k :*
- Use date from January 2020 and take only features that we extract in the previous section and add both last week k value and q value as features
- Xgboost : {random_state = 27, max_depth=6, subsample=0.6, n_estimators=300}
- LGBM : {colsample_bytree=0.8, subsample=0.8, subsample_freq=1, n_estimators=300, random_state=27} 

*Winning Strategy* : **MIN/MAX STRATEGY**

# Datathon Results
The prediction must be made over 6 days, from December 11th to December 16th.
Unfortunately we notice that the prediction for the 16th december doesn't seem good, we realise at the end that
it was because of the missing data during the 9th december (remember that for some streets we take into account
the last week value), then our models was trying to predict with one feature missing (which was very important)
We could have use the 2th december but it was too late ...

We will released the final results as soon as we have them.
(There is also a readme on the notebook folder with a brief description of each notebooks)
