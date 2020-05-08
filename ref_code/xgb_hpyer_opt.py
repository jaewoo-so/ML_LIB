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
import xgboost as xgb
global_regressor  = xgb.XGBRegressor
global_classifier = xgb.XGBClassifier

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
        params['objective'] = 'reg:squarederror'
        params['eval_metric'] = 'rmse'

        model = global_regressor(**params)
        model.fit(xs,ys,eval_set=[(xtest,ytest)] , verbose = False)
        pred = model.predict(xtest)

        #evaluation
        result = -1 * mean_squared_error(ytest, pred)

    elif mode == 'classification':
        params['objective'] = 'multi:softmax'
        params['eval_metric'] = 'mlogloss'
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
        params['objective'] = 'reg:logistic'
        params['eval_metric'] = 'error'

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
            params['objective'] = 'reg:squarederror'
            params['eval_metric'] = 'rmse'


            model = global_regressor(**params)
            model.fit(xtrain,ytrain,eval_set=[(xtest,ytest)] , verbose = False)
            pred = model.predict(xtest)

            #evaluation
            result = -1 * mean_squared_error(ytest, pred)
            
            
        elif mode == 'classification':
            params['objective'] = 'multi:softmax'
            params['eval_metric'] = 'mlogloss'
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
            params['objective'] = 'reg:logistic'
            params['eval_metric'] = 'error'

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
    def xgb_eval(colsample_bytree,
                  subsample,
                  n_estimators,
                  max_depth,
                  reg_alpha,
                  reg_lambda,
                  max_delta_step ,
                  min_child_weight):
        params['colsample_bytree'] = max(min(colsample_bytree, 1), 0)
        params['subsample']        = max(min(subsample, 1), 0)
        params['n_estimators']     = int(n_estimators)
        params['max_depth']        = int(max_depth)
        params['reg_alpha']        = max(reg_alpha, 0)
        params['reg_lambda']       = max(reg_lambda, 0)
        params['max_delta_step']  =max(max_delta_step, 0)
        params['min_child_weight'] = min_child_weight

        params['verbose']=0
        
        
        # start scoring
        if (type(xtest) == type(None) ) or ( type(ytest) == type(None) ):
            cv_score = objective_fold(params, xdata, ydata, n_split=n_split, mode=mode, usef1=use_f1)  # 여기에 전역 데이터를 넣어야 한다.
        else:
            cv_score = objective_fix(params, xdata, ydata, xtest, ytest, mode=mode, usef1=use_f1)  # 여기에 전역 데이터를 넣어야 한다.
        gc.collect()
        return cv_score
    return xgb_eval
    

params = dict()
params['tree_method'] = 'gpu_hist'
#params['tree_method'] = 'hist'
params['booster'] = 'gbtree' 
#params['booster'] = 'gblinear' 
params['objective'] = 'reg:logistic' ### 변경 요망 
#params['objective'] = 'reg:squarederror' # nan 오류난다. 
params['eval_metric'] = 'error'
params['feature_fraction'] = 1.0
params["learning_rate"] = 0.02 
params['n_jobs'] = -1


# Data and run
# Data Load
# exsample
from sklearn.datasets import load_boston, load_breast_cancer, load_iris

df_regression = load_boston()
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

    xgb_eval = create_bysop_eval(params ,k,X_train,y_train,X_test,y_test,5, use_f1 = True)
    clf_bo = BayesianOptimization(xgb_eval, {'max_depth': (2, 100),
                                          'colsample_bytree': (0.001, 1),
                                            'subsample': (0.001, 1),
                                            'n_estimators': (2, 1000), 
                                            'reg_alpha': (2, 20),
                                            'reg_lambda': (0, 10),
                                            'max_delta_step': (0, 10),
                                            'min_child_weight': (0, 45),
                                            })
    clf_bo.maximize(init_points=1, n_iter=2)
    print(clf_bo.max)
    print('-' * 100)
    
    xgb_eval = create_bysop_eval(params ,k ,X_train,y_train,5, use_f1 = True)
    clf_bo = BayesianOptimization(xgb_eval, {'max_depth': (2, 100),
                                          'colsample_bytree': (0.001, 1),
                                            'subsample': (0.001, 1),
                                            'n_estimators': (2, 1000),
                                            'reg_alpha': (2, 20),
                                            'reg_lambda': (0, 10),
                                            'max_delta_step': (0, 10),
                                            'min_child_weight': (0, 45),
                                            })
    clf_bo.maximize(init_points=1, n_iter=2)
    print(clf_bo.max)
    print('-' * 100)
print('all pass')
print('time : {}'.format(time() - starttime))


param_best = clf_bo.max['params']
#param_best ={'colsample_bytree': 0.3818480846747036,
# 'max_delta_step': 6.07835552618218,
# 'max_depth': 50.94531911709876,
# 'min_child_weight': 10.01593053996585,
# 'n_estimators': 787.0987580783351,
# 'reg_alpha': 13.008972248251542,
# 'reg_lambda': 0.4609240586904695,
# 'subsample': 0.9054453307650995}
param_best['n_estimators'] = int(param_best['n_estimators'])
param_best['max_depth'] = int(param_best['max_depth'])

param_best['tree_method'] = 'hist'
param_best['booster'] = 'gbtree' 
param_best['objective'] = 'reg:squarederror'
param_best['eval_metric'] = 'error'
param_best['feature_fraction'] = 1.0
param_best["learning_rate"] = 0.02 
param_best['n_jobs'] = -1


model = global_regressor(**param_best) ## regressor or classifier?
model.fit(X_train, y_train, verbose=False)



param_best ={'colsample_bytree': 0.3818480846747036,
 'max_delta_step': 6.07835552618218,
 'max_depth': 50.94531911709876,
 'min_child_weight': 10.01593053996585,
 'n_estimators': 787.0987580783351,
 'reg_alpha': 13.008972248251542,
 'reg_lambda': 0.4609240586904695,
 'subsample': 0.9054453307650995}
param_best['n_estimators'] = int(param_best['n_estimators'])
param_best['max_depth'] = int(param_best['max_depth'])

param_best['tree_method'] = 'hist'
param_best['booster'] = 'gbtree' 
param_best['objective'] = 'reg:squarederror'
param_best['eval_metric'] = 'error'
param_best['feature_fraction'] = 1.0
param_best["learning_rate"] = 0.02 
param_best['n_jobs'] = -1
