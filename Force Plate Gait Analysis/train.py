import numpy as np
import pandas as pd

from data import getDataForId, balanceData 

#Common Model Algorithms
from sklearn import (svm, tree, linear_model, neighbors, naive_bayes, 
                     ensemble, discriminant_analysis, gaussian_process)
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from sklearn.base import clone
from sklearn.utils import all_estimators
from xgboost import XGBClassifier

import time
# https://www.kaggle.com/code/grigol1/all-classification-models-40-sklearn-models
def getAllModels():
    estimators = [ (
            str(class_).split("'")[1].split(".")[1],
            name, 
            class_
        ) for name, class_ in all_estimators(type_filter='classifier') ]
    # from itertools import groupby
    # for key, group in groupby(estimators, lambda x: x[0]):
    #     print('from sklearn.' + key + ' import ' + ', '.join([x[1] for x in group]))
    MLA = []
    for module, name, class_ in estimators:
        if module in ('multiclass', 'multioutput', 'naive_bayes', 
                      'neighbors', 'gaussian_process', 'dummy',
                      'discriminant_analysis', 'calibration',
                      'semi_supervised'):
            continue
        if name in ('StackingClassifier', 'VotingClassifier', 'NuSVC'):
            continue
        # TODO: add more classifiers for ensemble
        if name in ('AdaBoostClassifier', 'BaggingClassifier', 
                    'ExtraTreesClassifier', 'GradientBoostingClassifier', 
                    'RandomForestClassifier'):
            MLA.append(class_(n_estimators=200, random_state=0))
            continue
        if name in ('SVC', ):
            MLA.append(class_(probability=True, random_state=0))
            continue
        MLA.append(class_(random_state=0) if 'random_state' in class_().get_params() else class_())
    return MLA + [XGBClassifier(n_estimators=200, random_state=0)]


def MLA_selection(X, Y, MLA):
    cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 )

    MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean','MLA Time']
    MLA_compare = pd.DataFrame(columns = MLA_columns)

    MLA_model = Y.copy()

    row_index = 0
    for alg in MLA[:]:
        # set name and parameters
        MLA_name = alg.__class__.__name__
        MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
        MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
        
        # score model with cross validation
        cv_results = model_selection.cross_validate(alg, X, Y, cv=cv_split, return_train_score=True)

        MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
        MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
        MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
        # save MLA predictions - see section 6 for usage
        try:
            alg.fit(X, Y)
            MLA_model[MLA_name] = alg # TODO: maybe don't pass in the fitted alg
        except Exception as e:
            MLA_compare.loc[row_index, 'MLA Time'] = np.nan
            MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = np.nan
            MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = np.nan
        
        row_index+=1

    MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
    return  MLA_compare, MLA_model 


def ensembleVote(X, Y):
    """
    This is deterministic!
    """
    vote_est = [
        # #Ensemble Methods: http://scikit-learn.org/stable/modules/ensemble.html
        # ('ada', ensemble.AdaBoostClassifier(n_estimators=200)),
        # ('bc', ensemble.BaggingClassifier(n_estimators=200)),
        # ('etc',ensemble.ExtraTreesClassifier(n_estimators=200)),
        # ('gbc', ensemble.GradientBoostingClassifier(n_estimators=200)),
        # ('rfc', ensemble.RandomForestClassifier(n_estimators=200)),

        # #Gaussian Processes: http://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-classification-gpc
        # ('gpc', gaussian_process.GaussianProcessClassifier()),
        
        # #GLM: http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
        # ('lr', linear_model.LogisticRegressionCV()),
        # ('lreg', linear_model.LogisticRegression()),
        
        # # #Navies Bayes: http://scikit-learn.org/stable/modules/naive_bayes.html
        # # ('bnb', naive_bayes.BernoulliNB()),
        # # ('gnb', naive_bayes.GaussianNB()),
        
        # #Nearest Neighbor: http://scikit-learn.org/stable/modules/neighbors.html
        # ('knn', neighbors.KNeighborsClassifier()),
        
        # #SVM: http://scikit-learn.org/stable/modules/svm.html
        # ('svc', svm.SVC(probability=True)),
        
        # #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
        # ('xgb', XGBClassifier(n_estimators=200))
        (alg.__class__.__name__, alg) for alg in getAllModels()
    ]
    # for model in vote_est:
    #     if model[0] in ('NuSVC', 'XGBClassifier'):
    #         vote_est.remove(model)
            # try:
            #     model[1].fit(X, Y)
            # except Exception as e:
            #     vote_est.remove(model)
    vote_hard = ensemble.VotingClassifier(estimators=vote_est , voting='hard')
    # cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 )
    cv_split = model_selection.KFold(n_splits=10, shuffle=True, random_state=0)
    vote_hard_cv = model_selection.cross_validate(vote_hard, X, Y, cv=cv_split , return_train_score=True)
    vote_hard.fit(X, Y)
    # vote_hard_cv['test_score'] is an array of size 10
    return vote_hard_cv['test_score'][~np.isnan(vote_hard_cv['test_score'])].mean()

def ensembleModel(X,Y, MLA):
    vote_est = [ (alg.__class__.__name__, alg) for alg in MLA]
    vote_hard = ensemble.VotingClassifier(estimators=vote_est , voting='hard')
    cv_split = model_selection.KFold(n_splits=10, shuffle=True, random_state=0)
    vote_hard_cv = model_selection.cross_validate(vote_hard, X, Y, cv=cv_split , return_train_score=True)
    vote_hard.fit(X, Y)
    return vote_hard



# def ensemblePredict(X, X_train, Y_train, algorithms):
    # vote_est = [ (x.__class__.__name__, x) for x in algorithms]
    # vote_hard = ensemble.VotingClassifier(estimators=vote_est , voting='hard')
    # cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 )
    # vote_hard_cv = model_selection.cross_validate(vote_hard, X_train, Y_train, cv=cv_split)
    # vote_hard.fit(X_train, Y_train)
    # return vote_hard.predict(X)

def ensemblePredict(X, X_train, Y_train, algorithms):
    result = [alg.predict(X)[0] for alg in algorithms]
    # get the most common result
    return max(set(result), key=result.count)


# mla, predict = MLA_selection(X, Y, MLA) 
# print( mla )
def getBestParams(X, Y):
    grid_n_estimator = [10, 50, 100, 300]
    grid_ratio = [.1, .25, .5, .75, 1.0]
    grid_learn = [.01, .03, .05, .1, .25]
    grid_max_depth = [2, 4, 6, 8, 10, None]
    grid_min_samples = [5, 10, .03, .05, .10]
    grid_criterion = ['gini', 'entropy']
    grid_bool = [True, False]
    grid_seed = [0]
    grid_penalty = ['l1', 'l2', 'elasticnet', None]

    clf_param_map = {
        'AdaBoostClassifier': (ensemble.AdaBoostClassifier(), { 
                                'n_estimators': grid_n_estimator, 
                                'learning_rate': grid_learn, 
                                'random_state': grid_seed 
                                }),
        'RandomForestClassifier': (ensemble.RandomForestClassifier(n_jobs=-1), {
                                'n_estimators': grid_n_estimator,
                                'criterion': grid_criterion,
                                'max_depth': grid_max_depth,
                                'max_features': ['sqrt', 'log2', None],
                                'oob_score': [True],
                                'random_state': grid_seed
                                }),
        'LogisticRegression': (linear_model.LogisticRegression(), {
                                'penalty': grid_penalty,
                                'C': np.logspace(0, 4, 10),
                                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], #default: lbfgs
                                'max_iter': [100, 1000, 10000],
                                'random_state': grid_seed
                                }),
        'XGBClassifier': (XGBClassifier(), {
                                'learning_rate': grid_learn,
                                'max_depth': [1,2,4,6,8,10],
                                'n_estimators': grid_n_estimator,
                                'seed': grid_seed
                                }),
        'PassiveAggressiveClassifier': (linear_model.PassiveAggressiveClassifier(), {
                                'C': np.logspace(0, 4, 10),
                                'fit_intercept': grid_bool,
                                'max_iter': [100, 1000, 10000],
                                'tol': [1e-3, 1e-4, 1e-5],
                                'random_state': grid_seed
                                }),
        'LinearDiscriminantAnalysis': (discriminant_analysis.LinearDiscriminantAnalysis(), {
                                'solver': ['svd', 'lsqr', 'eigen'],
                                'tol': [1e-3, 1e-4, 1e-5],
                                'shrinkage': [None, 'auto', 0.1, 0.5, 1.0],
                                'store_covariance': grid_bool
                                }),
    }
    start_total = time.perf_counter() #https://docs.python.org/3/library/time.html#time.perf_counter
    cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 )
    best = {}
    for clf_name, (clf, param) in clf_param_map.items():
        start = time.perf_counter()
        best_clf = model_selection.GridSearchCV(estimator=clf, param_grid=param, cv=cv_split, scoring='roc_auc')
        best_clf.fit(X, Y)
        run = time.perf_counter() - start
        best_param = best_clf.best_params_
        print(f'BEST PARAM: {best_param} - {clf_name} - {run:.2f}s')
        best[clf_name] = best_clf.best_estimator_
        print(f'Best Score: {best_clf.best_score_:.2f}')



"""
BEST PARAM: {'learning_rate': 0.01, 'n_estimators': 300, 'random_state': 0} - AdaBoostClassifier - 72.57s
Best Score: 0.87
BEST PARAM: {'criterion': 'entropy', 'max_depth': 6, 'max_features': 'log2', 'n_estimators': 300, 'oob_score': True, 'random_state': 0} - RandomForestClassifier - 456.10s
Best Score: 0.90
BEST PARAM: {'C': 2.7825594022071245, 'max_iter': 100, 'penalty': 'l1', 'random_state': 0, 'solver': 'saga'} - LogisticRegression - 276.99s
Best Score: 0.90
BEST PARAM: {'learning_rate': 0.1, 'max_depth': 1, 'n_estimators': 300, 'seed': 0} - XGBClassifier - 241.58s
Best Score: 0.87
BEST PARAM: {'C': 1.0, 'fit_intercept': True, 'max_iter': 100, 'random_state': 0, 'tol': 0.001} - PassiveAggressiveClassifier - 18.63s
Best Score: 0.87
BEST PARAM: {'shrinkage': 0.5, 'solver': 'lsqr', 'store_covariance': True, 'tol': 0.001} - LinearDiscriminantAnalysis - 31.88s
Best Score: 0.88
"""

if __name__ == "__main__":
    MLA = getAllModels()

    X, Y, Xid = getDataForId(ID, missingHashMap, train) 
    # X, Y = balanceData(X, Y)
    
    mla, predict = MLA_selection(X, Y, MLA)
    # mla, predict = MLA_selection(X, Y, MLA)  
    # get top 5 models
    print( mla[ ['MLA Name' , 'MLA Test Accuracy Mean'] ] )


# https://www.kaggle.com/code/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy
# https://www.kaggle.com/code/durgancegaur/a-guide-to-any-classification-problem
# https://www.kaggle.com/code/amerwafiy/titanic-competition-journey-to-100-accuracy#
# https://www.kaggle.com/getting-started/159643
# https://www.kaggle.com/code/jasonchong914/feature-selection-dimensionality-reduction
# https://towardsdatascience.com/lazy-predict-fit-and-evaluate-all-the-models-from-scikit-learn-with-a-single-line-of-code-7fe510c7281


def CNN():
    # train[[*LF, *RF, *LH, *RH]].iloc[0]
    X = np.stack((
        train[LF].to_numpy(),
        train[RF].to_numpy(),
        train[LH].to_numpy(),
        train[RH].to_numpy(),
    ), axis=2)
    Y = train['RH'] # 0 or 1

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization

    # define the model
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(30, 4)),
        Conv1D(filters=32, kernel_size=3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Conv1D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        # Conv1D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        # Conv1D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        # BatchNormalization(),
        # MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    LR = 0.0001
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    # compile the model
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # fit the model
    model.fit(X_train, y_train, epochs=200, batch_size=4, validation_data=(X_test, y_test))

    # evaluate the model
    model.evaluate(X_test, y_test)

