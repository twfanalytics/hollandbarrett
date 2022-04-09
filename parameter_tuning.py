import warnings
import pdb
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb


from time import time
from datetime import datetime
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import randint, uniform, loguniform
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization

warnings.filterwarnings('ignore')


def my_custom_loss_func(y_true, y_pred):
    # False negative rate
    cm = metrics.confusion_matrix(y_true, y_pred)

    # tn = cm[0][0]
    fn = cm[1][0]
    tp = cm[1][1]
    fp = cm[0][1]
    fnr = fn / (fn + tp)
    fpr = fp / (fp + tp)

    return fnr + fpr


# Run a random search over a range of parameters for a number of iterations
def run_random_search(x, y, clf, param_dist, n_iter_search):
    """Run a random search for best Decision Tree parameters.

    Args
    ----
    X -- features
    y -- targets (classes)
    cf -- scikit-learn Decision Tree
    param_dist -- [dict] list, distributions of parameters
                  to sample
    cv -- fold of cross-validation, default 5
    n_iter_search -- number of random parameter sets to try,
                     default 20.

    Returns
    -------
    top_params -- [dict] from report()
    """
    print("Start random search")

    score = metrics.make_scorer(my_custom_loss_func, greater_is_better=False) ## maybe standard scorer?

    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search,
                                       # scoring=score,
                                       verbose=2)

    start = time()
    random_search.fit(x, y)

    print("RandomizedSearchCV took {:.2f} seconds for {:d} candidates parameter settings.".format(
        (time() - start), n_iter_search))

    # top_params = report(random_search.cv_results_, 3)
    print(random_search.best_params_)

    return random_search.cv_results_, random_search.best_params_


def bayes_opt_xgb(df):
    print('Start bayes opt xgb')

    # Creating x and y
    x = df.drop('Y', axis=1)
    y = df['Y']

    # Converting the dataframe into XGBoost’s Dmatrix object
    dtrain = xgb.DMatrix(x, label=y)

    # Bayesian Optimization function for xgboost specify the parameters you want to tune as keyword arguments
    def bo_tune_xgb(max_depth, gamma, n_estimators, learning_rate):
        params = {'max_depth': int(max_depth),
                  'gamma': gamma,
                  'n_estimators': int(n_estimators),
                  'learning_rate': learning_rate,
                  'subsample': 0.8,
                  'eta': 0.1,
                  'eval_metric': 'rmse'}

        # Cross validating with the specified parameters in 5 folds and 70 iterations
        cv_result = xgb.cv(params, dtrain, num_boost_round=70, nfold=5)

        # Return the negative RMSE
        return -1.0 * cv_result['test-rmse-mean'].iloc[-1]

    # Invoking the Bayesian Optimizer with the specified parameters to tune
    xgb_bo = BayesianOptimization(bo_tune_xgb, {'max_depth': (3, 10),
                                                'gamma': (0, 1),
                                                'learning_rate': (0, 1),
                                                'n_estimators': (100, 120)})

    # Performing Bayesian optimization for 5 iterations with 8 steps of random exploration with an
    # acquisition function of expected improvement
    xgb_bo.maximize(n_iter=10, init_points=8, acq='ei')

    # Extracting the best parameters
    params = xgb_bo.max['params']

    # Converting the max_depth and n_estimator values from float to int
    params['max_depth'] = int(params['max_depth'])
    params['n_estimators'] = int(params['n_estimators'])

    print(params)

    return params
