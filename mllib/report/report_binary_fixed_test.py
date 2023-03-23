'''
This 

'''
import sys
sys.path.append('../')

import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.metrics import accuracy_score
from report import report_base
from data_class.data_fixed_test import FixedTestData, FixedTestResult

class ReportBinaryFixedTest(report_base.Report):
    def __init__(self) -> None:
        super().__init__()

    def oof_custom_score_each_model(self, result : FixedTestResult , ytrain , score_func) -> pd.DataFrame:
        '''
        result format
        fold_predict , fold_oof , fold_metric , fold_models = tr.training_fixedTest( )
        result[name] = [ fold_predict , fold_oof , fold_metric , fold_models ]
        '''
        model_name_first = list(result.result.keys())[0]
        fold_names  = list(result[model_name_first][0].keys())
        result = pd.DataFrame( columns = ['model' , 'score'])
        model_name_list = []
        for model_name in result:
            pred_res = result[model_name][1]
            score = {}
            score['model'] = model_name

            # there is no way to figure out input type of score_func, binary or probablity. so use try and except. 
            try:
                score['score'] = score_func(ytrain, pred_res)
            except:
                score['score'] = score_func( ytrain, np.round(pred_res))#
            result = result.append(score , ignore_index=True)
        return result

    def oof_score_each_model(self):
        pass

    def oof_pred_each_model(self, result) -> pd.DataFrame:
        '''
        result format
        fold_predict , fold_oof , fold_metric , fold_models = tr.training_fixedTest( )
        result[name] = [ fold_predict , fold_oof , fold_metric , fold_models ]
        '''
        res = {}
        for model_name in result:
            pred_res = result[model_name][1]
            res[model_name] = pred_res
        df_res = pd.DataFrame(res)
        return df_res


    def test_score_each_model(self) -> pd.DataFrame:
        pass

    def test_pred_each_model(self,result) -> pd.DataFrame:
        '''
        result format
        fold_predict , fold_oof , fold_metric , fold_models = tr.training_fixedTest( )
        result[name] = [ fold_predict , fold_oof , fold_metric , fold_models ]
        '''
        res = {}
        for model_name in result:
            pred_res = result[model_name][0]
            for i ,nfold in enumerate(pred_res):
                res[model_name+str(i)] = pred_res[nfold]
        df_res = pd.DataFrame(res)
        return df_res

    def train_pred_each_model(self,result) -> pd.DataFrame:
        '''
        res_all format in Test_Regression

        fold_predict , fold_oof , fold_metric , fold_models = tr.training_fixedTest( )
        res_all[name] = [ fold_predict , fold_oof , fold_metric , fold_models ]
        '''
        res = {}
        for model_name in result:
            pred_res = result[model_name][1]
            res[model_name] = pred_res
        df_res = pd.DataFrame(res)
        return df_res



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

        # there is no way to figure out input type of score_func, binary or probablity. so use try and except. 
        try:
            score['score'] = score_func(ytest, pred_res)
        except: 
            score['score'] = score_func( ytest, np.round(pred_res)) #
        result = result.append(score , ignore_index=True)
    return result

def oof_each_model(res_all , ytest , score_func):
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

        # there is no way to figure out input type of score_func, binary or probablity. so use try and except. 
        try:
            score['score'] = score_func(ytest, pred_res)
        except: 
            score['score'] = score_func( ytest, np.round(pred_res)) #
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
            try:
                score[str(nfold)] = score_func(ytest, pred_res[nfold])
            except: 
                score[str(nfold)] = score_func(np.round(ytest,0), pred_res[nfold])
        result = result.append(score , ignore_index=True)
    return result

def test_fold_each_model(res_all , ytest , score_func):
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
            try:
                score[str(nfold)] = score_func(ytest, pred_res[nfold])
            except: 
                score[str(nfold)] = score_func(np.round(ytest,0), pred_res[nfold])
        result = result.append(score , ignore_index=True)
    return result


# 각 테스트된 모델의 트레이닝세트에 대한 예측 값
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

def result2df_rowsample_colmodel_testmean_old(res_all):
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
def result2df_rowsample_colmodel_testmean(res_all):
    '''
    res_all format in Test_Regression

    fold_predict , fold_oof , fold_metric , fold_models = tr.training_fixedTest( )
    res_all[name] = [ fold_predict , fold_oof , fold_metric , fold_models ]
    '''
    res = {}
    for model_name in res_all:
        pred_res = res_all[model_name][0]
        res[model_name] = np.vstack(pred_res.values())
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