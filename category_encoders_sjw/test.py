# my lib
import sys
sys.path.append(r'E:\01_PProj\ML_LIB')
sys.path.append(r'D:\00_work\ML_LIB')
import category_encoders_sjw as ce


import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.model_selection import StratifiedKFold, KFold

category = ['A','A','A','A','B','B','B','C','C','D','D']
category2 = ['a','a','b','b','c','c','c','c','d','d','d']
target = [0,1,0,1,0,1,0,1,0,1,0]

test_cat = ['A', 'B', 'C', 'D', 'Z', 'D']
category = ['A','A','A','A','B','B','B','C','C','D','D']
category2 = ['a','a','b','b','c','c','c','c','d','d','d']
target = [0,1,0,1,0,1,0,1,0,1,0]

test_cat = ['A','B','C','D','Z','D']

train = pd.DataFrame()
test = pd.DataFrame()

train['cat'] = category
train['cat2'] = category2
#train['target']= target
test['cat'] = test_cat
test['cat2'] = test_cat
#test['target']= None

feature_list = train.columns

encoder_list = [ce.BackwardDifferenceEncoder(feature_list),
#ce.BaseNEncoder(feature_list),
#ce.BinaryEncoder(feature_list),
#ce.CatBoostEncoder(feature_list),
#ce.MeanEncoder(feature_list),
ce.HelmertEncoder(feature_list),
#ce.JamesSteinEncoder(feature_list),
#ce.LeaveOneOutEncoder(feature_list),
#ce.MEstimateEncoder(feature_list),
#ce.OneHotEncoder(feature_list),
#ce.OrdinalEncoder(feature_list),
#ce.SumEncoder(feature_list),

#ce.PolynomialEncoder(feature_list),
#ce.TargetEncoder(feature_list),
ce.WOEEncoder(feature_list)]

method_name_list = []
encoder_fit_list = OrderedDict()
result_list = OrderedDict()


for encoder in [ce.MeanEncoder(feature_list), ce.HelmertEncoder(feature_list)]:
    
    
    method_name = encoder.__class__.__name__
    print(method_name)
    method_name_list = method_name

    method_naming = method_name.replace('Encoder','')
  
    # run
    if method_name == 'MeanEncoder':
        train_en , test_en = encoder.fit_transform(train ,test, target)
    else:
        train_en = encoder.fit_transform(train , target)
        test_en = encoder.transform(test)
    
    train_en.columns = train_en.columns.map(lambda x : x + '_' + method_naming)
    test_en.columns = test_en.columns.map(lambda x : x + '_' + method_naming)
    
    encoder_fit_list[method_naming] = encoder
    result_list[method_naming] = [train_en ,test_en]
    
    ## 이제 데이터가 완성이 되었다. 
    train_data = pd.concat([train.drop(feature_list,axis= 1) , train_en] , axis=  1)
    test_data = pd.concat([test.drop(feature_list,axis= 1) , test_en] , axis=  1)
    
    
    
print('Done')


def fit_transform(self , df_train, test_data,target, reg_method='k_fold',
                alpha=5, add_random=False, rmean=0, rstd=0.1, folds=4):
    target_col = 'target'
    train_data = df_train.copy()
    train_data [target_col] = target
    columns = self.feature_names
    encoded_cols = []
    #self.target_mean_global = train_data[target_col].mean()
    target_mean_global = train_data[target_col].mean()
    for col in columns:
        # Getting means for test data
        nrows_cat = train_data.groupby(col)[target_col].count()
        target_means_cats = train_data.groupby(col)[target_col].mean()
        target_means_cats_adj = (target_means_cats*nrows_cat + 
                                 target_mean_global*alpha)/(nrows_cat+alpha)
        # Mapping means to test data
        encoded_col_test = test_data[col].map(target_means_cats_adj)
        # Getting a train encodings
        if reg_method == 'expanding_mean':
            train_data_shuffled = train_data.sample(frac=1, random_state=1)
            cumsum = train_data_shuffled.groupby(col)[target_col].cumsum() - train_data_shuffled[target_col]
            cumcnt = train_data_shuffled.groupby(col).cumcount()
            encoded_col_train = cumsum/(cumcnt)
            encoded_col_train.fillna(target_mean_global, inplace=True)
            if add_random:
                encoded_col_train = encoded_col_train + normal(loc=rmean, scale=rstd, 
                                                               size=(encoded_col_train.shape[0]))
        elif (reg_method == 'k_fold') and (folds > 1):
            kfold = KFold( folds, shuffle=True, random_state=1)
            parts = []
            for tr_in, val_ind in kfold.split(train_data[target_col].values):
                # divide data
                df_for_estimation, df_estimated = train_data.iloc[tr_in], train_data.iloc[val_ind]
                # getting means on data for estimation (all folds except estimated)
                nrows_cat = df_for_estimation.groupby(col)[target_col].count() # 각 카테고리별 샘플 수 
                target_means_cats = df_for_estimation.groupby(col)[target_col].mean()
                target_means_cats_adj = (target_means_cats*nrows_cat + target_mean_global*alpha)/(nrows_cat+alpha)
                encoded_col_train_part = df_estimated[col].map(target_means_cats_adj)
                if add_random:
                    encoded_col_train_part = encoded_col_train_part + normal(loc=rmean, scale=rstd, 
                                                                             size=(encoded_col_train_part.shape[0]))
                # Saving estimated encodings for a fold
                parts.append(encoded_col_train_part) # 1개의 폴드에 대한 엔코딩값
            encoded_col_train = pd.concat(parts, axis=0) # 모든 폴드에 대한 엔코딩 값
            encoded_col_train.fillna(target_mean_global, inplace=True) # 빠진 값은 글로벌 평균으로 채워넣기 
        else:
            encoded_col_train = train_data[col].map(target_means_cats_adj)
            if add_random:
                encoded_col_train = encoded_col_train + normal(loc=rmean, scale=rstd, 
                                                               size=(encoded_col_train.shape[0]))
        # Saving the column with means
        encoded_col = pd.concat([encoded_col_train, encoded_col_test], axis=0)
        encoded_col[encoded_col.isnull()] = target_mean_global
        encoded_cols.append(  pd.DataFrame({col + '_' + 'mean_'+target_col : encoded_col})  ) # 리스트, 안쪽은 1컬럼짜리 시리즈
    all_encoded = pd.concat(encoded_cols, axis=1)
    return (all_encoded.loc[train_data.index,:], all_encoded.loc[test_data.index, :])