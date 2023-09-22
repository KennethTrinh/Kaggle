import pandas as pd
import numpy as np

from data import (getData, organizeColumns,
                  getMissingHashMap, 
                  getRowById,
                  Normalize, Standardize,
                  getDataForId,
                  balanceData, feature_selection, PCA_selection, 
                  getTrainFrontData
)
from train import (getAllModels, MLA_selection, ensemblePredict, ensembleVote, ensembleModel)


def prepareTrainingItems(label):
    train = getData(label, 'train') if label in ('LH', 'RH') else getTrainFrontData(label)
    test = getData(label, 'test') # train for now, change later
    (LF, RF, LH, RH, 
    LF_trot, RF_trot, LH_trot, RH_trot, 
    LF_STD, RF_STD, LH_STD, RH_STD, 
    LF_trot_STD, RF_trot_STD, LH_trot_STD, RH_trot_STD) = organizeColumns(train)

    COLUMNS_TO_SCALE = set( LF + RF + LH + RH + \
                            LF_trot + RF_trot + LH_trot + RH_trot + \
                            ['speed', 'Speed', 'age', 'weight'] + \
                            LF_STD + RF_STD + LH_STD + RH_STD + \
                            LF_trot_STD + RF_trot_STD + LH_trot_STD + RH_trot_STD
    )
    COLUMNS_TO_DROP = set(  
    )

    testMissingHashMap = getMissingHashMap(test)
    return train, test, testMissingHashMap, COLUMNS_TO_DROP, COLUMNS_TO_SCALE

def synthesizeModelAndFeatures(id, label, train, test, testMissingHashMap, 
                                COLUMNS_TO_DROP, COLUMNS_TO_SCALE, MLA):
    X_train, Y_train, _ = getDataForId(id, testMissingHashMap, train,
                                    label=label, mode = 'train',
                                    columns_to_drop=COLUMNS_TO_DROP,
                                    columns_to_scale=COLUMNS_TO_SCALE,
                                    standardization_func=lambda df, columns: Standardize(df, columns, columnWise=True),
                                    normalization_func=lambda df, columns: df
    )
    if label in ('LF', 'RF'):
        X_train, Y_train = balanceData(X_train, Y_train)
    # print('Shape of X_train after balancing:', X_train.shape)
    features = feature_selection(X_train, Y_train)
    print('Feat selection:', X_train.shape, '-->', X_train[features].shape)
    X_train = X_train[features]
    model = ensembleModel(X_train, Y_train, MLA)
    return model, features, X_train, Y_train

def generatePredictionForLabel(id, label, train, test, testMissingHashMap, 
                                COLUMNS_TO_DROP, COLUMNS_TO_SCALE, MLA):
    if id in train.id.unique():
        print(f'** id {id} is in training set for label {label} **')
        return train[train.id == id][label].values[0]
    model, features, *_ = synthesizeModelAndFeatures(id, label, train, test, testMissingHashMap,
                                        COLUMNS_TO_DROP, COLUMNS_TO_SCALE, MLA)
    # mlaRankings, algs = MLA_selection(X_train, Y_train, MLA)
    # print('mean accuracy:',  mlaRankings[ 'MLA Test Accuracy Mean'].mean() )

    X_test, Xid = getDataForId(id, testMissingHashMap, test,
                                label=label, mode = 'test',
                                columns_to_drop=COLUMNS_TO_DROP,
                                columns_to_scale=COLUMNS_TO_SCALE,
                                standardization_func=lambda df, columns: Standardize(df, columns, columnWise=True),
                                normalization_func=lambda df, columns: df
    )
    X_test = X_test[features]
    return model.predict(X_test[Xid == id])[0]

def generatePredictionsForLabelRemainder(ids, label, train, test, testMissingHashMap, 
                                COLUMNS_TO_DROP, COLUMNS_TO_SCALE, MLA):
    """
    So we don't have to retrain the model for the remaining ids since they are not missing any values
    ids[0] is the same as all the other ids since not in testMissingHashMap
    """
    model, features, *_ = synthesizeModelAndFeatures(ids[0], label, train, test, testMissingHashMap,
                                        COLUMNS_TO_DROP, COLUMNS_TO_SCALE, MLA)
    
    predictions = pd.DataFrame(columns=['id', label])
    for i, id in enumerate(ids):
        if id in train.id.unique():
            print(f'** id {id} is in training set for label {label} **')
            prediction = train[train.id == id][label].values[0]
            predictions = predictions.append({'id': id, label: prediction}, ignore_index=True)
            continue
        print(f'** Predicting for id {id} ({i+1}/{len(ids)}) **')
        X_test, Xid = getDataForId(id, testMissingHashMap, test,
                                    label=label, mode = 'test',
                                    columns_to_drop=COLUMNS_TO_DROP,
                                    columns_to_scale=COLUMNS_TO_SCALE,
                                    standardization_func=lambda df, columns: Standardize(df, columns, columnWise=True),
                                    normalization_func=lambda df, columns: df
        )
        X_test = X_test[features]
        prediction = model.predict(X_test[Xid == id])[0]
        predictions = predictions.append({'id': id, label: prediction}, ignore_index=True)
    return predictions



def generatePredictionsForLabel(label, train, test, testMissingHashMap, 
                                COLUMNS_TO_DROP, COLUMNS_TO_SCALE, MLA):    
    predictions = pd.DataFrame(columns=['id', label])
    for i, id in enumerate(testMissingHashMap.keys()):
        print(f'** Predicting for id {id} ({i+1}/{len(testMissingHashMap.keys())}) **')
        prediction = generatePredictionForLabel(id, label, train, test, testMissingHashMap,
                                                COLUMNS_TO_DROP, COLUMNS_TO_SCALE, MLA)
        
        predictions = predictions.append({'id': id, label: prediction}, ignore_index=True)

    remainingIds = list(set(test.id.unique()).difference(set(testMissingHashMap.keys())))
    remainingPredictions = generatePredictionsForLabelRemainder(remainingIds, label, train, test, testMissingHashMap,
                                                COLUMNS_TO_DROP, COLUMNS_TO_SCALE, MLA)
    predictions = predictions.append(remainingPredictions, ignore_index=True)
    return predictions

# predictions = generatePredictionForLabel(67, label, train, test, testMissingHashMap,
#                                             COLUMNS_TO_DROP, COLUMNS_TO_SCALE, MLA)
if __name__ == '__main__':
    import time
    start = time.time()
    LABELS = ['LF', 'RF', 'LH', 'RH']
    MLA = getAllModels()

    for label in LABELS:
        print(f'** Predicting for {label} **')
        train, test, testMissingHashMap, COLUMNS_TO_DROP, COLUMNS_TO_SCALE = prepareTrainingItems(label)
        predictions = generatePredictionsForLabel(label, train, test, testMissingHashMap,
                                                    COLUMNS_TO_DROP, COLUMNS_TO_SCALE, MLA)
        predictions.id = predictions.id.astype(int)
        predictions[label] = predictions[label].astype(int)
        print(predictions)
        predictions.to_csv(f'{label}_test_labels.csv', index=False)

    print(f'Finished in {time.time() - start} seconds')


# MLA = getAllModels()
# label = 'LF'
# id = 162#list(set(test.id.unique()).difference(set(testMissingHashMap.keys())))
# train, test, testMissingHashMap, COLUMNS_TO_DROP, COLUMNS_TO_SCALE = prepareTrainingItems(label)
# print('Inference for id:', id, 'label:', label)
# model, features, X_train, Y_train = synthesizeModelAndFeatures(
#                                     id, label, train, test, testMissingHashMap,
#                                     COLUMNS_TO_DROP, COLUMNS_TO_SCALE, MLA)

# X_train, Y_train, _ = getDataForId(id, testMissingHashMap, train,
#                                 label=label, mode = 'train',
#                                 columns_to_drop=COLUMNS_TO_DROP,
#                                 columns_to_scale=COLUMNS_TO_SCALE,
#                                 standardization_func=lambda df, columns: Standardize(df, columns, columnWise=True),
#                                 normalization_func=lambda df, columns: df
# )
# if label in ('LF', 'RF'):
#     X_train, Y_train = balanceData(X_train, Y_train)
# features = feature_selection(X_train, Y_train)
# print('Feat selection:', X_train.shape, '-->', X_train[features].shape)
# X_train = X_train[features]
# print(X_train.columns)
# print('mean accuracy:',  ensembleVote(X_train, Y_train))
                                                           
# predictions = pd.DataFrame(columns=['id', label])
# i, id = 0, ids[0]
# print(f'** Predicting for id {id} ({i+1}/{len(ids)}) **')
# X_test, Xid = getDataForId(id, testMissingHashMap, test,
#                             label=label, mode = 'test',
#                             columns_to_drop=COLUMNS_TO_DROP,
#                             columns_to_scale=COLUMNS_TO_SCALE,#train.columns.drop(label),
#                             standardization_func=lambda df, columns: Standardize(df, columns, columnWise=True),
#                             normalization_func=lambda df, columns: df
# )
# X_test = X_test[features]
# prediction = model.predict(X_test[Xid == id])[0]


# features = feature_selection(X_train, Y_train)
# from sklearn.ensemble import IsolationForest
# isf = IsolationForest(random_state=0)
# isf.fit(X_train, Y_train)
# outliers = isf.predict(X_train)
# X_train = X_train[outliers == -1]



# features = PCA_selection(X_train)
# X_train = X_train[features]

# def measure():
#     MLA = getAllModels()
#     LABELS = ['LF', 'RF', 'LH', 'RH']

#     def meanAccForID(id, label, train, test, testMissingHashMap, 
#                                     COLUMNS_TO_DROP, COLUMNS_TO_SCALE, MLA):
#         *_, X_train, Y_train = synthesizeModelAndFeatures(
#                                             id, label, train, test, testMissingHashMap,
#                                             COLUMNS_TO_DROP, COLUMNS_TO_SCALE, MLA)
#         return ensembleVote(X_train, Y_train), X_train.columns
#     def run(label):
#         train, test, testMissingHashMap, COLUMNS_TO_DROP, COLUMNS_TO_SCALE = prepareTrainingItems(label)
#         SAMPLES = 10
#         total = 0
#         for id in list(testMissingHashMap.keys())[:SAMPLES]:
#             print(f'** Label: {label}  id: {id} **')
#             x, features = meanAccForID(id, label, train, test, testMissingHashMap,
#                                         COLUMNS_TO_DROP, COLUMNS_TO_SCALE, MLA)
#             print('mean accuracy:', x)
#             total += x
#         print(f'Average accuracy for {label} over {SAMPLES} samples: {total/SAMPLES}')
#         ids = list(set(test.id.unique()).difference(set(testMissingHashMap.keys())))
#         print(f'Mean accuracy for remainider {label}:', meanAccForID(ids[0], label, train, test, testMissingHashMap,
#                                         COLUMNS_TO_DROP, COLUMNS_TO_SCALE, MLA))

#     run('LF')
#     run('LH')

# measure()

# id = 81
# X_train, Y_train, _ = getDataForId(id, testMissingHashMap, train,
#                                 label=label, mode = 'train',
#                                 columns_to_drop=COLUMNS_TO_DROP,
#                                 columns_to_scale=COLUMNS_TO_SCALE,
#                                 standardization_func=lambda df, columns: Standardize(df, columns, columnWise=True),
#                                 normalization_func=lambda df, columns: df
# )
# features = feature_selection(X_train, Y_train)#PCA_selection(X_train)
# X_train = X_train[features]
# from sklearn import ensemble, model_selection
# vote_est = [ (alg.__class__.__name__, alg) for alg in getAllModels() ]
# vote_hard = ensemble.VotingClassifier(estimators=vote_est , voting='hard')
# cv_split = model_selection.KFold(n_splits=10, shuffle=True, random_state=0)
# vote_hard_cv = model_selection.cross_validate(vote_hard, X_train, Y_train, cv=cv_split , return_train_score=True)
# vote_hard.fit(X_train, Y_train)
# print(vote_hard_cv['test_score'][~np.isnan(vote_hard_cv['test_score'])].mean())

# loop through the cv_split and print out the Y.value_counts() for each iteration
# for i, (train_index, test_index) in enumerate(cv_split.split(X_train, Y_train)):
#     print(f'Fold {i+1}')
#     print(f"TRAIN:\n{Y_train.iloc[train_index].value_counts()}")
#     print(f"TEST:\n{Y_train.iloc[test_index].value_counts()}")



# features = SelectKBest_selection(X_train, Y_train)
# features = SelectPercentile_selection(X_train, Y_train)



