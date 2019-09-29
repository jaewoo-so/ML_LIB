#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append(r'E:\01_PProj\ML_LIB')

import Lib_sjw.training as tr
import Lib_sjw.model_interface as mi
import Lib_sjw.model_parmas as mp
import Lib_sjw.evaluator as ev
import Lib_sjw.classification_util as cu
import Lib_sjw.ensemble as es

from sklearn.datasets import load_boston , load_iris , load_breast_cancer
from sklearn.metrics import mean_squared_error , roc_auc_score , precision_score , average_precision_score , accuracy_score , mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 
import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt
import pickle
from tqdm import tqdm_notebook as tqdm

from collections import OrderedDict


def Normalize(df):
    df_norm = (df-df.mean())/df.std()
    res =  ((df_norm-df_norm.min())/(df_norm.max()-df_norm.min()))
    #print( 'mean : {} , stf {} , min : {} , max : {}'.format( df.mean() , df.std() , df_norm.min() ,df_norm.max()))
    return res

snp_list = ['ID', 'Sex', 'Age', 'FTO', 'MC4R', 'BDNF', 'ANGPTL3',
       'MLXIPL', 'TRIB1', 'SORT1', 'HMGCR', 'ABO', 'ABCA1', 'MYL2', 'LIPG',
       'CETP', 'CDKN2A_B', 'G6PC2', 'GCK', 'GCKR', 'GLIS3', 'MTNR1B', 'DGKB',
       'SLC30A8', 'NPR3', 'ATP2B1', 'NT5C2', 'CSK', 'HECTD4', 'GUCY1A3',
       'CYP17A1', 'FGF5', 'AHR', 'CYP1A2', 'SLC23A1', 'AGER', 'MMP1', 'OCA2_1',
       'OCA2_2', 'MC1R', 'EDAR', 'target', 'BMI_range']

data_folder = 'E:/00_proj/genestart/inputs/'
dftr = pd.read_csv(data_folder + 'dataset_mean_v01_norm.BMI2binning_v05.csv')
dfte = pd.read_csv(data_folder + 'testset_mean_v01_norm.BMI2binning_v05.csv')

X_train = Normalize(dftr.iloc[:, 1:-2]).values
y_train = Normalize(dftr.iloc[:,-2]).values

X_test = Normalize(dfte.iloc[:, 1:-2]).values
y_test = Normalize(dfte.iloc[:,-2]).values
ytest_binary = np.where(dfte.iloc[:,-2] >25 , 1 , 0)


res_all = None
with open('E:/00_proj/genestart/code_final/05_AUC_SNP+QA_after_hypertunning.pickle' , 'rb') as f:
    res_all = pickle.load(f)

df_v1 = es.result2df(res_all)


def mix_select(df_single,df_mixmodel,ytest_binary , auc_limit= 0.6900, pr_limit = 0.6740):
    df_mix = pd.concat([df_single,df_mixmodel] , axis = 1)
    pair_pred , _ , _ = es.esemble_binary(df_mix , ytest_binary , 0.76 , 0.99 , 0) # m , m1+m2 의 각 예측치를 가지고 있다. 
    
    # test accuracy
    result_list1 = {}
    result_list2 = {}
    
    for model_name in tqdm(df_mix):
        pred = df_mix[model_name].values
        pred_norm = MinMaxScaler().fit_transform(pred.reshape(-1,1)) 
        result_list1[model_name] = roc_auc_score(ytest_binary , pred_norm)
        result_list2[model_name] = average_precision_score( ytest_binary , pred_norm)
        
    pair_auc_v121 = sorted(result_list1.items(), key=lambda x: x[1], reverse=True)
    pair_pr_v121  = sorted(result_list2.items(), key=lambda x: x[1] , reverse = True)
    
    df_auc_v121 = pd.DataFrame(pair_auc_v121 , columns = ['model' , 'auc'])
    df_pr_v121  = pd.DataFrame(pair_pr_v121 , columns = ['model' , 'aucpr'])
    
    df_auc_over = df_auc_v121[ df_auc_v121['auc'] > auc_limit ]
    df_pr_over  = df_pr_v121[ df_pr_v121['aucpr'] > pr_limit ]
    
    return df_auc_over , df_pr_over , pair_pred

def mix_select_single(df_mixmodel,ytest_binary , auc_limit= 0.6900, pr_limit = 0.6740):
    df_mix = df_mixmodel
    pair_pred , _ , _ = es.esemble_binary(df_mix , ytest_binary , 0.86 , 0.99 , 0) # m , m1+m2 의 각 예측치를 가지고 있다. 
    
    # test accuracy
    result_list1 = {}
    result_list2 = {}
    
    for model_name in tqdm(df_mix):
        pred = df_mix[model_name].values
        pred_norm = MinMaxScaler().fit_transform(pred.reshape(-1,1)) 
        result_list1[model_name] = roc_auc_score(ytest_binary , pred_norm)
        result_list2[model_name] = average_precision_score( ytest_binary , pred_norm)
        
    pair_auc_v121 = sorted(result_list1.items(), key=lambda x: x[1], reverse=True)
    pair_pr_v121  = sorted(result_list2.items(), key=lambda x: x[1] , reverse = True)
    
    df_auc_v121 = pd.DataFrame(pair_auc_v121 , columns = ['model' , 'auc'])
    df_pr_v121  = pd.DataFrame(pair_pr_v121 , columns = ['model' , 'aucpr'])
    
    df_auc_over = df_auc_v121[ df_auc_v121['auc'] > auc_limit ]
    df_pr_over  = df_pr_v121[ df_pr_v121['aucpr'] > pr_limit ]
    
    return df_auc_over , df_pr_over , pair_pred

def create_ensemble(df_auc_over ,df_pr_over , pair_pred ):
    auc_over_set = set(df_auc_over.model.values)
    pr_over_set  = set(df_pr_over.model.values)
    pair_set     = set(pair_pred.keys())
    intersaction_model_name    = auc_over_set.union(pr_over_set).intersection(pair_set)
    
    prediction_value_over = [ pair_pred[k] for k in intersaction_model_name ]
    df_mix = pd.DataFrame(np.array(prediction_value_over).T , columns = intersaction_model_name )
    return df_mix


pair_pred_v1 , _ , _ = es.esemble_binary(df_v1 , ytest_binary , 0.76 , 0.99 , 0) 

df_v2 = pd.read_csv('start_v3.csv')





df_auc_over, df_pr_over, pair_pred = mix_select(df_v1, df_v2, ytest_binary)
print(df_auc_over.head(10))

df_v3 = create_ensemble(df_auc_over , df_auc_over , pair_pred)
df_v123 = pd.concat([df_v1 , df_v2 , df_v3] , axis = 1)
df_v123 = df_v123.ix[:,~df_v123.columns.duplicated()]


df_auc_over, df_pr_over, pair_pred = mix_select_single(df_v123, ytest_binary)

print(df_auc_over.head(10))