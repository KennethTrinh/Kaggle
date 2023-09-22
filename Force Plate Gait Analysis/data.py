import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
import warnings
warnings.filterwarnings('ignore')


# pd.set_option('display.max_columns', None) 
# train = pd.read_csv('LF_train.csv')
# ['LF', 'RF', 'LH', 'RH'] --> LF and RF are very imbalanced

def getData(label, train_or_test):
    df = pd.read_csv(f'kaggle4780/{label}_{train_or_test}.csv')

    def checkFloat(x):
        try:
            return float(x)
        except:
            return np.nan
    # convert object to float
    df['speed'] = df['speed'].apply(checkFloat)
    df['Speed'] = df['Speed'].apply(checkFloat)
    return df
# now build the train data
def getTrainFrontData(LF_or_RF):
    rf = getData('RF', 'train')
    lf = getData('LF', 'train')
    combined = pd.DataFrame(columns=rf.columns)
    # na for now
    combined['LF'] = np.nan
    for id in (set(rf.id.unique()) | set(lf.id.unique())):
        if id in rf['id'].unique() and id in lf['id'].unique():
            r = rf[rf['id'] == id]
            l = lf[lf['id'] == id]
            r['LF'] = l['LF'].values[0]
            combined = combined.append(r)
        elif id in rf['id'].unique(): # if id is in rf, add the columns of rf onto ids
            combined = combined.append(rf[rf['id'] == id])
        else: # otherwiise add the columns of lf onto ids
            combined = combined.append(lf[lf['id'] == id])
    combined['RF'].fillna(combined['LF'], inplace=True)
    combined['LF'].fillna(combined['RF'], inplace=True)
    if LF_or_RF == 'LF':
        combined.drop('RF', inplace=True, axis=1)
    else:
        combined.drop('LF', inplace=True, axis=1)
    return combined


# train = pd.read_csv('RH_train.csv')
# train[ train['id'] == 169][LF]
# train = pd.read_csv('LF_train.csv')
# train[ train['id'] == 169][LF]

# get all the columns V1_LH, V2_LH... V30_LH
"""
>>> for col in train.columns:
    print(col)
"""
def organizeColumns(train):
    # 30*4 = 120
    LF = [col for col in train.columns if 'V' in col and 'LF' in col and not 'trot' in col and not 'STD'  in col]
    RF = [col for col in train.columns if 'V' in col and 'RF' in col and not 'trot' in col and not 'STD'  in col]
    LH = [col for col in train.columns if 'V' in col and 'LH' in col and not 'trot' in col and not 'STD'  in col]
    RH = [col for col in train.columns if 'V' in col and 'RH' in col and not 'trot' in col and not 'STD'  in col]

    # 30*4 = 120
    LF_trot = [col for col in train.columns if 'V' in col and 'LF' in col and 'trot' in col and not 'STD' in col]
    RF_trot = [col for col in train.columns if 'V' in col and 'RF' in col and 'trot' in col and not 'STD' in col]
    LH_trot = [col for col in train.columns if 'V' in col and 'LH' in col and 'trot' in col and not 'STD' in col]
    RH_trot = [col for col in train.columns if 'V' in col and 'RH' in col and 'trot' in col and not 'STD' in col]

    # 15*4 = 60
    LF_STD = [col for col in train.columns if 'V' in col and 'LF' in col and not 'trot' in col and 'STD'  in col]
    RF_STD = [col for col in train.columns if 'V' in col and 'RF' in col and not 'trot' in col and 'STD'  in col]
    LH_STD = [col for col in train.columns if 'V' in col and 'LH' in col and not 'trot' in col and 'STD'  in col]
    RH_STD = [col for col in train.columns if 'V' in col and 'RH' in col and not 'trot' in col and 'STD'  in col]

    # 15*4 = 60
    LF_trot_STD = [col for col in train.columns if 'V' in col and 'LF' in col and 'trot' in col and 'STD' in col]
    RF_trot_STD = [col for col in train.columns if 'V' in col and 'RF' in col and 'trot' in col and 'STD' in col]
    LH_trot_STD = [col for col in train.columns if 'V' in col and 'LH' in col and 'trot' in col and 'STD' in col]
    RH_trot_STD = [col for col in train.columns if 'V' in col and 'RH' in col and 'trot' in col and 'STD' in col]

    # 2 * 2 = 4
    walk = ['gait', 'speed']
    trot = ['Gait', 'Speed']

    # 6
    misc = ['id', 'dob', 'gender', 'weight', 'forceplate_date', 'age', ]#LABEL]
    # total 

    for i in (LF, RF, LH, RH, LF_trot, RF_trot, LH_trot, RH_trot):
        assert len(i) == 30
    for i in (LF_STD, RF_STD, LH_STD, RH_STD, LF_trot_STD, RF_trot_STD, LH_trot_STD, RH_trot_STD):
        assert len(i) == 15
    assert train.shape[1] == (120 + 120 + 60 + 60 + 4 + 7)

    return (
        LF, RF, LH, RH,
        LF_trot, RF_trot, LH_trot, RH_trot,
        LF_STD, RF_STD, LH_STD, RH_STD,
        LF_trot_STD, RF_trot_STD, LH_trot_STD, RH_trot_STD,
    )


# train['gait'][train['gait'] != 'Walk']
# train['Gait'][train['Gait'] != 'Trot']


def plotSeveral(cols, batches=20):
    fig, axes = plt.subplots(10, 2, figsize=(8, 8))
    axes = axes.flatten()
    for i in range(batches):
        axes[i].plot(range(1,31), train[cols].iloc[i])
        axes[i].set_title(f'ID: {train.id.iloc[i]}')
    plt.show()
def plotOne(col, batch=0):
    plt.bar(range(1,31), train[col].iloc[batch])
    plt.title(f'ID: {train.id.iloc[batch]}')
    plt.show()

def plotNormalized(col, batch=0):
    plt.plot(range(15, 31), train[col].iloc[batch, 14:])
    plt.title(f'ID: {train.id.iloc[batch]}')
    plt.show()


def analyze(X):
    correlations= X.corr()
    correlations = correlations.where(np.tril(np.ones(correlations.shape)).astype(bool))
    correlations = correlations.stack()
    correlations = correlations[correlations > 0.85]
    correlations = correlations[correlations.index.get_level_values(0) != correlations.index.get_level_values(1)]
    for i in range(0, len(correlations)):
        print(f'{correlations.index[i][0]} and {correlations.index[i][1]} have a correlation of {correlations[i]}')


# HELPER FUNCTIONS
getRowById = lambda id, df: df[df.id == id]
checkAnyNan = lambda df: df.isnull().values.any()

# iterate over the rows, and see which columns have NaNs
def getMissingHashMap(train):
    """
    finds missing columns for each training example (row)
    Maps id to columns that have NaNs
    This makes it so that:
    getRowById(id, train)[ missnigHashMap[id] ].shape[1] == len(missingHashMap[id])
    """
    missingHashMap = {}
    for row in train[train.isnull().any(axis=1)].index:
        # print(f'*** ID: {train.id.iloc[row]} ***')
        # print([col for i, col in enumerate(train.columns) if train[col].isnull().iloc[row]])
        missingHashMap[train.id.iloc[row]] = \
            [col for i, col in enumerate(train.columns) if train[col].isnull().iloc[row]]
    return missingHashMap


# drop columns of missingHashMap[id]
def createTrainModifiedForId(id, train, missingHashMap):
    """
    Will eventually pass in testMissingHashMap, but still use train training data.
    Returns modified training data for a specific id such that all columns for with Nans 
    for that id are dropped.  Then, all rows (training examples) with NaNs are dropped.
    """
    missinigColumns = []
    if id in missingHashMap:
        missinigColumns = missingHashMap[id]
    #
    trainModifiedForId = train.drop(missinigColumns, axis=1, inplace=False)
    indicesNA = trainModifiedForId[trainModifiedForId.isnull().any(axis=1)].index
    trainModifiedForId = trainModifiedForId.drop(indicesNA)
    #
    if checkAnyNan(trainModifiedForId):
        raise Exception('There are still NaNs in the data')
    return trainModifiedForId

def Normalize(df, columns, columnWise=True):
    df = df.copy()
    scaler = Normalizer()
    if columns:
        df[columns] = scaler.fit_transform(df[columns].T).T if columnWise \
                                    else scaler.fit_transform(df[columns])
    return df

def Standardize(df, columns, columnWise=True):
    df = df.copy()
    scaler = StandardScaler()
    if columns:
        df[columns] = scaler.fit_transform(df[columns]) if columnWise \
                                    else scaler.fit_transform(df[columns].T).T
    return df

def normalizeTrain(train):
    """
    this doesn't seem to work
    """
    arr = [
        zip(LH[:len(LH_STD)], LH_STD),
        zip(RH[:len(RH_STD)], RH_STD),
        zip(LF[:len(LF_STD)], LF_STD),
        zip(RF[:len(RF_STD)], RF_STD),
        zip(LH_trot[:len(LH_trot_STD)], LH_trot_STD),
        zip(RH_trot[:len(RH_trot_STD)], RH_trot_STD),
        zip(LF_trot[:len(LF_trot_STD)], LF_trot_STD),
        zip(RF_trot[:len(RF_trot_STD)], RF_trot_STD),
    ]
    for batch in arr:
        for value, std in batch:
            mean = train[value].mean()
            train[value] = (train[value] - mean) / train[std]

# LABEL='RH'
# train = getData(LABEL, 'train')
# ID = train['id'].unique()[1]
# LF, RF, LH, RH, LF_trot, RF_trot, LH_trot, RH_trot, LF_STD, RF_STD, LH_STD, RH_STD, LF_trot_STD, RF_trot_STD, LH_trot_STD, RH_trot_STD = organizeColumns(train)

# cols = RH_trot
# TODO: maybe add std later, try fft too
# columns_to_scale = set( LF + RF + LH + RH + \
#                         LF_trot + RF_trot + LH_trot + RH_trot + \
#                         ['speed', 'Speed', 'age', 'weight']
# )

# what to do with: ['speed', 'Speed', 'age', 'weight'] ?

# columns_to_drop = set( LF_STD + RF_STD + LH_STD + RH_STD + \
#                     LF_trot_STD + RF_trot_STD + LH_trot_STD + RH_trot_STD 
# )
# missingHashMap = getMissingHashMap(train)
# all should be greater than or equal to 75
# for id in train.id.unique():
#     trainModifiedForId = createTrainModifiedForId(id, train, missingHashMap)
#     print(f'ID: {id}, shape: {trainModifiedForId.shape[0]}')

def getDataForId(id, missingHashMap, df, label, mode,
                         columns_to_drop, columns_to_scale,
                         standardization_func=lambda df, columns: Standardize(df, columns, columnWise=True),
                         normalization_func=lambda df, columns: df
                         ):
    dataModifiedForId = createTrainModifiedForId(id, df, missingHashMap)
    dataModifiedForId = dataModifiedForId.reset_index(drop=True)

    Xid = dataModifiedForId.id

    # columns to drop
    columns = [col for col in dataModifiedForId.columns if col in columns_to_drop]
    dataModifiedForId.drop(columns, axis=1, inplace=True)

    dataModifiedForId.drop([col for col in ('id', 'dob', 'gait', 'Gait', 'forceplate_date') \
                            if col in dataModifiedForId.columns], 
                            axis=1, inplace=True)
    # columns to scale
    columns = [col for col in dataModifiedForId.columns if col in columns_to_scale]
    dataModifiedForId = standardization_func(dataModifiedForId, columns)
    dataModifiedForId = normalization_func(dataModifiedForId, columns)

    # indicesNA = train[train.isnull().any(axis=1)].index
    # >>> train.drop(indicesNA).shape # dropping all rows that have at least one NaN
    # (75, 371)
    # >>> dataModifiedForId.shape # dropping all columns with Nan for id
    # (88, 189)
    if mode == 'train':
        X = dataModifiedForId.drop([label], axis=1, inplace=False)
        Y = dataModifiedForId[label]
        return X, Y, Xid
    
    X = dataModifiedForId
    return X, Xid



def balanceData(X, Y, under_multiplier=3, over_divisor=2):
    # over = SMOTE(sampling_strategy=over_percentage)
    # under = RandomUnderSampler(sampling_strategy=under_percentage)
    # pipeline = Pipeline(steps=[('o', over), ('u', under)])
    # under = RandomUnderSampler(sampling_strategy=under_percentage)
    # return pipeline.fit_resample(X, Y)
    count_0, count_1 = Y.value_counts() # 77, 13
    # under sample
    df_class_0 = X[Y == 0]
    df_class_1 = X[Y == 1]
    df_class_0_under = df_class_0.sample(count_1 * under_multiplier, random_state=0)
    df_class_1_over = df_class_1.sample(count_1 // over_divisor, random_state=0)
    df_class_1 = pd.concat([df_class_1, df_class_1_over], axis=0)
    X = pd.concat([df_class_0_under, df_class_1], axis=0)
    Y = pd.concat([pd.Series([0] * df_class_0_under.shape[0]), pd.Series([1] * df_class_1.shape[0])], axis=0)
    return X, Y

from sklearn.decomposition import PCA
def PCA_selection(X):
    pca = PCA(n_components=min(X.shape[0], X.shape[1]))
    pca.fit(X)
    X_pca = pca.transform(X)
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(X_pca.shape[1])]
    most_important_names = [X.columns[most_important[i]] for i in range(X_pca.shape[1])]
    return list(set(most_important_names))

from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                                GradientBoostingClassifier, AdaBoostClassifier)

def Trees_selection(X_train, Y_train, tree=ExtraTreesClassifier):
    TOP_FEATURES = X_train.shape[1]//2

    forest = tree(n_estimators=250, max_depth=5, random_state=1) if tree != AdaBoostClassifier else tree(n_estimators=250, random_state=1)
    forest.fit(X_train, Y_train)

    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    indices = indices[:TOP_FEATURES]

    mostImportantFeatures = X_train.columns[indices].to_list()
    return mostImportantFeatures


from xgboost import XGBClassifier
from sklearn.feature_selection import RFE
def RFE_selection(X_train, Y_train):
    rfe = RFE(estimator=XGBClassifier(n_jobs=-1, 
                                    n_estimators=250, 
                                    max_depth=5, 
                                    random_state=1))
    rfe.fit(X_train, Y_train)
    mostImportantFeatures = X_train.columns[rfe.support_].to_list()
    return mostImportantFeatures



from functools import reduce
def feature_selection(X_train, Y_train):
    features = [
        Trees_selection(X_train, Y_train, tree=t) for t in (ExtraTreesClassifier, RandomForestClassifier,
                                GradientBoostingClassifier, AdaBoostClassifier)
    ] + [RFE_selection(X_train, Y_train)] 
    # + [PCA_selection(X_train)]
    result = reduce(lambda x, y: set.intersection(set(x), set(y)), features)
    return sorted(list(result)) # no ordering in set

"""
rh = getData('RH', 'train')
lh = getData('LH', 'train')
rf = getData('RF', 'train')
lf = getData('LF', 'train')
# make df of ids, RH, LH
rh = rh[ ['id','RH']]
lh = lh[ ['id','LH']]
rf = rf[ ['id','RF']]
lf = lf[ ['id','LF']]
# ids = pd.merge(rh, lh, on='id', how='outer')
# ids = pd.merge(ids, rf, on='id', how='outer')
# ids = pd.merge(ids, lf, on='id', how='outer')
ids = pd.merge(rf, lf, on='id', how='outer')
# fiind ids with NA
idsNA = ids[ids.isnull().any(axis=1)].index
ids.drop(idsNA, inplace=False).corr()
# turns out that RF and LF are perfectly correlated, so we 
# fill the NAs for RF with LF, and vice versa
ids['RF'].fillna(ids['LF'], inplace=True)
ids['LF'].fillna(ids['RF'], inplace=True)
"""
