import pandas as pd
import numpy as np
import seaborn as sns
from functools import partial
from bayes_opt import BayesianOptimization
import gc
from sklearn.metrics import accuracy_score , mean_squared_error , f1_score , average_precision_score , roc_auc_score , confusion_matrix , precision_recall_curve , roc_auc_score
from sklearn.model_selection import StratifiedKFold , train_test_split , KFold

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from time import time
import catboost as cat

global_regressor  = cat.CatBoostRegressor
global_classifier = cat.CatBoostClassifier

# mode = regression , classification
def get_max_f1_cutoff(model , xs_val , ys_val ):
    pred_train_val = model.predict_proba(xs_val)[:, 1]

    if np.isnan(pred_train_val).sum() > 0:
        return 0

    pr  ,re ,th= precision_recall_curve(ys_val , pred_train_val)
    th = np.concatenate([th, np.array([1])])
    df_prre = pd.DataFrame( np.array([pr , re , th]).T , columns = ['pr' , 're' , 'th'])
    df_prre['f1'] = 2*(df_prre['pr']*df_prre['re'])/(df_prre['pr']+df_prre['re'])
    df_prre = df_prre.sort_values(by = 'f1' , ascending = False)
    return df_prre.iloc[0,:]

def objective_fix(params, xs , ys , xtest , ytest , mode = 'regression',usef1 = False):
    if mode == 'regression':
        params['objective'] = 'RMSE'
        params['eval_metric'] = 'RMSE'

        model = global_regressor(**params)
        model.fit(xs,ys,eval_set=[(xtest,ytest)] , verbose = False)
        pred = model.predict(xtest)

        #evaluation
        result = -1 * mean_squared_error(ytest, pred)

    elif mode == 'classification':
        params['objective'] = 'MultiClass'
        params['eval_metric'] = 'MultiClass'
        params['num_class'] = len(np.unique(ys))

        model = global_classifier(**params)
        model.fit(xs,ys,eval_set=[(xtest,ytest)] , verbose = False)
        pred = model.predict(xtest)

        #evaluation
        result = None
        if usef1:
            result = f1_score(ytest, pred , average = 'micro')
        else:
            result = accuracy_score(ytest, pred)

    elif mode == 'binary':
        params['objective'] = 'Logloss'
        params['eval_metric'] = 'AUC'

        model = global_classifier(**params)
        model.fit(xs,ys,eval_set=[(xtest,ytest)] , verbose = False)
        pred = model.predict(xtest)

        #evaluation
        result = None
        if usef1:
            result = get_max_f1_cutoff(model, xtest, ytest)[-1]  # f1 value
        else:
            result = roc_auc_score(ytest, model.predict_proba(xtest)[:, 1])  # auc value
        
    return result


def objective_fold(params,xs,ys,n_split = 5, mode = 'regression',usef1 = False):
    
    if mode == 'regression':
        kfold = KFold(n_splits=n_split, random_state=7, shuffle=True)
    elif (mode == 'classification') or (mode == 'binary'):
        kfold = StratifiedKFold(n_splits=n_split, random_state=7, shuffle=False)
    
    allscore = []
    for train_index, val_index  in kfold.split(xs,ys):
        xtrain, xtest = xs[train_index], xs[val_index]
        ytrain, ytest = ys[train_index], ys[val_index]
        
        if mode == 'regression':
            params['objective'] = 'RMSE'
            params['eval_metric'] = 'RMSE'


            model = global_regressor(**params)
            model.fit(xtrain,ytrain,eval_set=[(xtest,ytest)] , verbose = False)
            pred = model.predict(xtest)

            #evaluation
            result = -1 * mean_squared_error(ytest, pred)
            
            
        elif mode == 'classification':
            params['objective'] = 'MultiClass'
            params['eval_metric'] = 'MultiClass'
            params['num_class'] = len(np.unique(ys))
            
            model = global_classifier(**params)
            model.fit(xtrain,ytrain,eval_set=[(xtest,ytest)] , verbose = False)
            pred = model.predict(xtest)
            #pred_round = pred.round()
            result = None
            if usef1:
                result = f1_score(ytest, pred , average = 'micro')
            else:
                result = accuracy_score(ytest, pred)
                
        elif mode == 'binary':
            params['objective'] = 'Logloss'
            params['eval_metric'] = 'AUC'

            model = global_classifier(**params)
            model.fit(xtrain,ytrain,eval_set=[(xtest,ytest)] , verbose = False)
            pred = model.predict(xtest)

            #evaluation
            result = None
            if usef1:
                result = get_max_f1_cutoff(model, xtest, ytest)[-1]  # f1 value
            else:
                result = roc_auc_score(ytest, model.predict_proba(xtest)[:, 1])  # auc value
            
        allscore.append(result)
            
    return np.asarray(allscore).mean()


def create_bysop_eval(params , mode ,  xdata , ydata , xtest = None , ytest = None, n_split = 5 , use_f1 = False):
    def xgb_eval(iterations,
                  learning_rate,
                  depth,
                  l2_leaf_reg,
                  random_strength,
                  bagging_temperature,
                  ):
        params['iterations']           = int(iterations)
        params['learning_rate']        =  max(learning_rate, 0)
        params['depth']                = int(depth)
        params['l2_leaf_reg']        = int(l2_leaf_reg)
        params['random_strength']        = max(random_strength, 0)
        params['bagging_temperature']       = max(bagging_temperature, 0)
      

        params['verbose']=0
        
        
        # start scoring
        if (type(xtest) == type(None) ) and ( type(ytest) == type(None) ):
            cv_score = objective_fold(params, xdata, ydata, n_split=n_split, mode=mode, usef1=use_f1)  # 여기에 전역 데이터를 넣어야 한다.
        else:
            cv_score = objective_fix(params, xdata, ydata, xtest, ytest, mode=mode, usef1=use_f1)  # 여기에 전역 데이터를 넣어야 한다.
        gc.collect()
        return cv_score
    return xgb_eval
    

params = dict()
#params['task_type'] = 'CPU'
params['task_type'] = 'GPU'

params['grow_policy'] = 'SymmetricTree'
params['bootstrap_type'] = 'Bayesian' #Bernoulli , MVS ( CPU only) , Poisson ( GPU only) , No


# Data and run
# Data Load
# exsample
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_iris

df_regression = fetch_california_housing()
df_binary = load_breast_cancer()
df_multi = load_iris()

df_list = {}
df_list['regression'] = df_regression
df_list['binary'] = df_binary
df_list['classification'] = df_multi
starttime = time()
for k, v in df_list.items():
    if k == 'regression':
        X_train, X_test, y_train, y_test = train_test_split(v.data, v.target, test_size=0.20,random_state= 517)
    else:
        X_train, X_test, y_train, y_test = train_test_split(v.data, v.target, test_size=0.20, stratify=v.target, random_state= 517)

    y_train = y_train.flatten()
    y_test = y_test.flatten()

    cat_eval = create_bysop_eval(params ,k,X_train,y_train,X_test,y_test,5, use_f1 = True)
    clf_bo = BayesianOptimization(cat_eval, {'iterations': (2, 3000),
                                             'learning_rate': (0.003, 0.9),
                                             'depth': (2, 16),
                                             'l2_leaf_reg': (0.01, 6), 
                                             'random_strength': (0.1, 40),
                                             'bagging_temperature': (0.1, 40),
                                        
                                            })
    clf_bo.maximize(init_points=1, n_iter=2)
    print(clf_bo.max)
    print('-' * 100)

    cat_eval = create_bysop_eval(params ,k ,X_train,y_train,5, use_f1 = True)
    clf_bo = BayesianOptimization(cat_eval, {'iterations': (2, 3000),
                                             'learning_rate': (0.003, 0.9),
                                             'depth': (2, 16),
                                             'l2_leaf_reg': (0.01, 6), 
                                             'random_strength': (0.1, 40),
                                             'bagging_temperature': (0.1, 40),
                                            })
    clf_bo.maximize(init_points=1, n_iter=2)
    print(clf_bo.max)
    print('-' * 100)
print('all pass')
print('time : {}'.format(time()-starttime ))