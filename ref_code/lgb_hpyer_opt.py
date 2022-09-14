import pandas as pd
import numpy as np
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
from functools import partial
from bayes_opt import BayesianOptimization
import gc
from sklearn.metrics import accuracy_score , mean_squared_error , f1_score , average_precision_score , roc_auc_score , confusion_matrix , precision_recall_curve , roc_auc_score
from sklearn.model_selection import StratifiedKFold , train_test_split , KFold
from time import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



import lightgbm as lgb
global_regressor  = lgb.LGBMRegressor
global_classifier = lgb.LGBMClassifier

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
        params['objective'] = 'regression'
        params['metric'] = 'l2'

        model = global_regressor(**params)
        model.fit(xs,ys,eval_set=[(xtest,ytest)] , )
        pred = model.predict(xtest)

        #evaluation
        result = -1 * mean_squared_error(ytest, pred)

    elif mode == 'classification':
        params['objective'] = 'multiclass'
        params['metric'] = 'multiclass'
        params['num_class'] = len(np.unique(ys))

        model = global_classifier(**params)
        model.fit(xs,ys,eval_set=[(xtest,ytest)] , )
        pred = model.predict(xtest)

        #evaluation
        result = None
        if usef1:
            result = f1_score(ytest, pred , average = 'micro')
        else:
            result = accuracy_score(ytest, pred)

    elif mode == 'binary':
        params['objective'] = 'binary'
        params['metric'] = 'auc'

        model = global_classifier(**params)
        model.fit(xs,ys,eval_set=[(xtest,ytest)] , )
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
            params['objective'] = 'regression' # 트레이닝에 사용
            params['metric'] = 'l2' # eval_set에 사용

            model = global_regressor(**params)
            model.fit(xtrain,ytrain,eval_set=[(xtest,ytest)] , )
            pred = model.predict(xtest)

            #evaluation
            result = -1 * mean_squared_error(ytest, pred)
            
            
        elif mode == 'classification':
            params['objective'] = 'multiclass'
            params['metric'] = 'multiclass'
            params['num_class'] = len(np.unique(ys))

            model = global_classifier(**params)
            model.fit(xtrain,ytrain,eval_set=[(xtest,ytest)] , )
            pred = model.predict(xtest)
            #pred_round = pred.round()
            result = None
            if usef1:
                result = f1_score(ytest, pred , average = 'micro')
            else:
                result = accuracy_score(ytest, pred)
                
        elif mode == 'binary':
            params['objective'] = 'binary'
            params['metric'] = 'auc'

            model = global_classifier(**params)
            model.fit(xtrain,ytrain,eval_set=[(xtest,ytest)] , )
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
    def lgbm_eval(num_iterations,
                  num_leaves,
                  colsample_bytree,
                  subsample,
                  max_depth,
                  reg_alpha,
                  reg_lambda,
                  min_split_gain,
                  min_child_weight,
                  min_data_in_leaf,
                  lambda_l1,
                  lambda_l2,
                  learning_rate,
                  bagging_fraction,
                  feature_fraction
                  ):
        params["num_iterations"] = int(num_iterations)          
        params["num_leaves"] = int(num_leaves)
        params['colsample_bytree'] = max(min(colsample_bytree, 1), 0)
        params['subsample'] = max(min(subsample, 1), 0)
        params['max_depth'] = int(max_depth)
        params['reg_alpha'] = max(reg_alpha, 0)
        params['reg_lambda'] = max(reg_lambda, 0)
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        params['min_data_in_leaf'] = int(min_data_in_leaf)
        params['lambda_l1'] = max(lambda_l1, 0)
        params['lambda_l2'] = max(lambda_l2, 0)
        params['learning_rate'] = max(learning_rate, 0.0001)
        params['bagging_fraction'] = min(bagging_fraction, 1.0)
        params['feature_fraction'] = min(feature_fraction, 1.0)
        params['verbose']= -1
        
        
         # start scoring
        if (type(xtest) == type(None) ) and ( type(ytest) == type(None) ):
            cv_score = objective_fold(params, xdata, ydata, n_split=n_split, mode=mode, usef1=use_f1)  # 여기에 전역 데이터를 넣어야 한다.
        else:
            cv_score = objective_fix(params, xdata, ydata, xtest, ytest, mode=mode, usef1=use_f1)  # 여기에 전역 데이터를 넣어야 한다.
        gc.collect()
        return cv_score
    return lgbm_eval
    

# multi calss classification

params = dict()
params['task'] = 'train'
params['device'] = 'gpu'
params['boosting'] = 'gbdt'

params['is_unbalance'] = True
#params["learning_rate"] = 0.02
params['seed']=326
params['bagging_seed']=326
#params['early_stopping_rounds'] = 100

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

    lgb_eval = create_bysop_eval(params ,k,X_train,y_train,X_test,y_test,5, use_f1 = True)
    clf_bo = BayesianOptimization(lgb_eval, {'num_iterations' : (2, 1000),
                                            'num_leaves': (42, 5000),
                                            'colsample_bytree': (0.001, 1),
                                            'subsample': (0.001, 1),
                                            'max_depth': (4, 15),
                                            'reg_alpha': (0, 20),
                                            'reg_lambda': (0, 10),
                                            'min_split_gain': (0, 1),
                                            'min_child_weight': (0, 45),
                                            'min_data_in_leaf': (1, 10),
                                           'lambda_l1': (0, 10),
                                           'lambda_l2': (0, 10),
                                           'learning_rate' : (0.6,0.001),
                                            'bagging_fraction' : (0.1,1.0),
                                            'feature_fraction' : (0.1,1.0)})
    clf_bo.maximize(init_points=1, n_iter=2)
    print(clf_bo.max)
    print('-' * 100)
    
    lgb_eval = create_bysop_eval(params ,k ,X_train,y_train,5, use_f1 = True)
    clf_bo = BayesianOptimization(lgb_eval, {'num_iterations' : ( 2 , 1000 ),
                                             'num_leaves': (42, 5000),
                                            'colsample_bytree': (0.001, 1),
                                            'subsample': (0.001, 1),
                                            'max_depth': (4, 15),
                                            'reg_alpha': (0, 20),
                                            'reg_lambda': (0, 10),
                                            'min_split_gain': (0, 1),
                                            'min_child_weight': (0, 45),
                                            'min_data_in_leaf': (0, 100),
                                            'lambda_l1': (0, 10),
                                            'lambda_l2': (0, 10),
                                            'learning_rate' : (0.6,0.001),
                                            'bagging_fraction' : (0.1,1.0),
                                            'feature_fraction' : (0.1,1.0)})
    clf_bo.maximize(init_points=1, n_iter=2)
    print(clf_bo.max)
    print('-' * 100)
print('all pass')
print('time : {}'.format(time() - starttime))



param_best = {'bagging_fraction': 0.21176541753913136,
 'colsample_bytree': 0.7140942216454612,
 'feature_fraction': 0.11946795064183889,
 'lambda_l1': 2.310135421593227,
 'lambda_l2': 7.468019368264845,
 'learning_rate': 0.22728585822684572,
 'max_depth': 9.240811767549072,
 'min_child_weight': 12.221204534872191,
 'min_data_in_leaf': 7.0330947275715765,
 'min_split_gain': 0.4964603965643156,
 'num_leaves': 4229.636789209139,
 'reg_alpha': 0.786784124586184,
 'reg_lambda': 7.287151182953648,
 'subsample': 0.024418195680870523}
param_best = clf_bo.max['params']
param_best['num_leaves'] = int(param_best['num_leaves'])
param_best['max_depth'] = int(param_best['max_depth'])
param_best['min_data_in_leaf'] = int(param_best['min_data_in_leaf'])
param_best['task'] = 'train'
param_best['device'] = 'cpu'
param_best['boosting'] = 'gbdt'
param_best['seed']=326
param_best['bagging_seed']=326
param_best['objective'] = 'regression' # 트레이닝에 사용
#param_best['metric'] = 'multi_logloss' # eval_set에 사용 