import pandas as pd
from collections import OrderedDict


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

import pickle
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from Lib_sjw import training_test_binary as ttb



if __name__ == "__main__":
    def real_example():
        # 추후 삭제 예정
        data_folder = r'E:/00_proj/genestart/inputs/'
        snp_list = ['ID', 'Sex', 'Age', 'FTO', 'MC4R', 'BDNF', 'ANGPTL3',
           'MLXIPL', 'TRIB1', 'SORT1', 'HMGCR', 'ABO', 'ABCA1', 'MYL2', 'LIPG',
           'CETP', 'CDKN2A_B', 'G6PC2', 'GCK', 'GCKR', 'GLIS3', 'MTNR1B', 'DGKB',
           'SLC30A8', 'NPR3', 'ATP2B1', 'NT5C2', 'CSK', 'HECTD4', 'GUCY1A3',
           'CYP17A1', 'FGF5', 'AHR', 'CYP1A2', 'SLC23A1', 'AGER', 'MMP1', 'OCA2_1',
           'OCA2_2', 'MC1R', 'EDAR', 'target', 'BMI_range']
        dfte = pd.read_csv(data_folder + 'testset_mean_v01_norm.BMI2binning_v05.csv')[snp_list]
        ytest = dfte.iloc[:,-2].values
        ytest_binary = np.where(ytest > 25, 1, 0)

        res_all = None
        with open(r'E:/00_proj/genestart/code_final/res_01_baseline_v02_onlySNP.pickle' , 'rb') as f:
            res_all = pickle.load(f)

        temp = result2df_rowmodel_colfold(res_all, ytest_binary, roc_auc_score)
        print(temp)
        print('test done')

    def toy_example():
        dataset = load_breast_cancer()
        x = dataset.data
        y = dataset.target
        
        xtrain,xtest,ytrain,ytest = train_test_split( x,y , test_size = 0.2 )

        res = ttb.Test_Binary(xtrain,ytrain,xtest)
        res2 = result2df_rowmodel_colfold_oof(res, ytrain, roc_auc_score)
        print(res2)
        print('toy example done')
    toy_example()

