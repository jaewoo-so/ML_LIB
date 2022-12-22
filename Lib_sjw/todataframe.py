import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.metrics import accuracy_score


# 각 폴드의 validation에 대한 점수
def result2df_rowmodel_colfold_oof(res_all , ytest , score_func):
    '''
    res_all format
    fold_predict , fold_oof , fold_metric , fold_models = tr.training_fixedTest( )
    res_all[name] = [ fold_predict , fold_oof , fold_metric , fold_models ]
    '''
    model_name_first = list(res_all.items())[0][0]
    fold_names  = list(res_all[model_name_first][0].keys())
    result = pd.DataFrame( columns = ['model' , 'score'])
    model_name_list = []
    for model_name in res_all:
        pred_res = res_all[model_name][1]
        score = {}
        score['model'] = model_name
        score['score'] = score_func(ytest, pred_res)
        result = result.append(score , ignore_index=True)
    return result

# 각 폴드의 test에 대한 점수
def result2df_rowmodel_colfold(res_all , ytest , score_func):
    '''
    res_all format
    fold_predict , fold_oof , fold_metric , fold_models = tr.training_fixedTest( )
    res_all[name] = [ fold_predict , fold_oof , fold_metric , fold_models ]
    '''
    model_name_first = list(res_all.items())[0][0]
    fold_names  = list(res_all[model_name_first][0].keys())
    result = pd.DataFrame( columns = ['model'] + fold_names)
    model_name_list = []
    for model_name in res_all:
        pred_res = res_all[model_name][0]
        score = {}
        score['model'] = model_name
        for i ,nfold in enumerate(pred_res):    
            score[str(nfold)] = score_func(ytest, pred_res[nfold])
        result = result.append(score , ignore_index=True)
    return result

def result2df_rowsample_colmodel_train(res_all):
    '''
    res_all format in Test_Regression

    fold_predict , fold_oof , fold_metric , fold_models = tr.training_fixedTest( )
    res_all[name] = [ fold_predict , fold_oof , fold_metric , fold_models ]
    '''
    res = {}
    for model_name in res_all:
        pred_res = res_all[model_name][1]
        res[model_name] = pred_res
    df_res = pd.DataFrame(res)
    return df_res

def result2df_rowsample_colmodel_test(res_all):
    '''
    res_all format in Test_Regression

    fold_predict , fold_oof , fold_metric , fold_models = tr.training_fixedTest( )
    res_all[name] = [ fold_predict , fold_oof , fold_metric , fold_models ]
    '''
    res = {}
    for model_name in res_all:
        pred_res = res_all[model_name][0]
        for i ,nfold in enumerate(pred_res):
            res[model_name+str(i)] = pred_res[nfold]
    df_res = pd.DataFrame(res)
    return df_res

def result2df_rowsample_colmodel_test_nonfixtest(res_all):
    '''
    res_all format in Test_Regression

    fold_predict , fold_oof , fold_metric , fold_models = tr.training_fixedTest( )
    res_all[name] = [ fold_predict , fold_oof , fold_metric , fold_models ]
    '''
    res = {}
    for model_name in res_all:
        pred_res = res_all[model_name][0]
        for i ,nfold in enumerate(pred_res):
            res[model_name+str(i)] = pred_res[nfold]
    df_res = pd.DataFrame(res)
    return df_res

def result2df_rowsample_colmodel_testmean(res_all):
    '''
    res_all format in Test_Regression

    fold_predict , fold_oof , fold_metric , fold_models = tr.training_fixedTest( )
    res_all[name] = [ fold_predict , fold_oof , fold_metric , fold_models ]
    '''
    res = {}
    for model_name in res_all:
        pred_res = res_all[model_name][0]

        result = []
        for i ,nfold in enumerate(pred_res):
            result.append(pred_res[nfold])
        res[model_name] = np.mean(result , axis = 0)
    df_res = pd.DataFrame(res)
    return df_res

# --- dictionary each model ---
def result2dict_name_model(res_all):
    '''
    res_all format in Test_Regression

    fold_predict , fold_oof , fold_metric , fold_models = tr.training_fixedTest( )
    res_all[name] = [ fold_predict , fold_oof , fold_metric , fold_models ]
    '''
    res = OrderedDict()
    for model_name in res_all:
        pred_res = res_all[model_name][3]
        for i ,nfold in enumerate(pred_res):
            res[model_name + str(i)] = pred_res[nfold]
    return res


### --------------------  Classification ------------------- ###
'''
멀티클래스의 경우는 다르게 처리해야 한다. 

전체에 대한 평균 정확도, 클래스별 평균 정확도  

클래스별 / 전체 평균 / 단순 정확도 나눠서 코드 만들어야된다.

'''
def result2df_rowmodel_colfold_acc_multi(res_all , ytest ):
    '''
    res_all format
    fold_predict , fold_oof , fold_metric , fold_models = tr.training_fixedTest( )
    res_all[name] = [ fold_predict , fold_oof , fold_metric , fold_models ]
    ytest = int type , ndarray 
    '''
   
    model_name_first = list(res_all.items())[0][0]
    fold_names  = list(res_all[model_name_first][0].keys())
    result = pd.DataFrame( columns = ['model'] + fold_names)
    model_name_list = []

    # setting
    score_func = accuracy_score
    ytest_int =  ytest.astype(int)

    for model_name in res_all:
        pred_res = res_all[model_name][0]
        score = {}
        score['model'] = model_name
        for i ,nfold in enumerate(pred_res):    
            pred_int = np.argmax(pred_res[nfold] , axis = 1)
            score[str(nfold)] = score_func( ytest.values.astype(int) , pred_int )
        result = result.append(score, ignore_index=True)
    result['mean'] = result.mean(axis = 1)
    return result
    

'''
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from Lib_sjw import training_test_binary as ttb


if __name__ == "__main__":
    def toy_example():
        dataset = load_breast_cancer()
        x = dataset.data
        y = dataset.target
        
        xtrain,xtest,ytrain,ytest = train_test_split( x,y , test_size = 0.2 )

        res = ttb.Test_Binary(xtrain,ytrain,xtest)
        res2 = result2df_rowmodel_colfold_oof(res, ytrain, roc_auc_score)
        res3 = result2df_rowmodel_colfold(res, ytest, roc_auc_score)
        print(res2)
        print('toy example done')
    toy_example()

'''