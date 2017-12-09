import pandas as pd
import numpy as np

def mssubclass(train, test, cols=['MSSubClass']):
    for i in (train, test):
        for z in cols:
            i[z] = i[z].apply(lambda x: str(x))
    return train, test

def log(train, test, y):
    numeric_feats = train.dtypes[train.dtypes != "object"].index
    for i in (train, test):
        i[numeric_feats] = np.log1p(i[numeric_feats])
    y = np.log1p(y)
    return train, test, y

def impute_mean(train, test):
    for i in (train, test):
        for s in [k for k in i.dtypes[i.dtypes != "object"].index if sum(pd.isnull(i[k])>0)]:
            i[s] = i[s].fillna(i[s].mean())
    return train, test

def dummies(train, test):
    columns = [i for i in train.columns if type(train[i].iloc[1]) == str or type(train[i].iloc[1]) == float]
    for column in columns:
        train[column].fillna('NULL', inplace = True)
        good_cols = [column+'_'+i for i in train[column].unique()[1:] if i in test[column].unique()]
        train = pd.concat((train, pd.get_dummies(train[column], prefix = column)[good_cols]), axis = 1)
        test = pd.concat((test, pd.get_dummies(test[column], prefix = column)[good_cols]), axis = 1)
        del train[column]
        del test[column]
    return train, test

def lotfrontage(train, test):
    for i in (train, test):
        i['LotFrontage'] = i['LotFrontage'].fillna(train['LotFrontage'].mean())
    return train, test

def garageyrblt(train, test):
    for i in (train, test):
        i['GarageYrBlt'] = i['GarageYrBlt'].fillna(train['GarageYrBlt'].mean())
    return train, test
