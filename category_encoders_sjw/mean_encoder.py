"""Target Encoder"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders.ordinal import OrdinalEncoder
import category_encoders.utils as util
from sklearn.model_selection import KFold



class MeanEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names ):
        self.feature_names = feature_names

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
                    parts.append(encoded_col_train_part)  # 1개의 폴드에 대한 엔코딩값
                   

                encoded_col_train = pd.concat(parts, axis=0)  # 모든 폴드에 대한 엔코딩 값
                
                
                encoded_col_train.fillna(target_mean_global, inplace=True) # 빠진 값은 글로벌 평균으로 채워넣기 
            else:
                encoded_col_train = train_data[col].map(target_means_cats_adj)
                if add_random:
                    encoded_col_train = encoded_col_train + normal(loc=rmean, scale=rstd, 
                                                                   size=(encoded_col_train.shape[0]))

            # Saving the column with means
            encoded_col = pd.concat([encoded_col_train, encoded_col_test], axis=0)
           
            encoded_col[encoded_col.isnull()] = target_mean_global
            encoded_cols.append(pd.DataFrame({col + '_' + 'mean_' + target_col: encoded_col}))  # 리스트, 안쪽은 1컬럼짜리 시리즈
            
  
        all_encoded = pd.concat(encoded_cols, axis=1)
        all_encoded = all_encoded.reset_index(drop = True)
       
        return (all_encoded.iloc[ :train_data.shape[0] , :], all_encoded.iloc[train_data.shape[0]:,:])
        

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split , KFold
if __name__ == '__main__':
    data = load_iris()

    x = data.data
    y = data.target
    cols = data.feature_names

    df = pd.DataFrame(x, columns=cols)
    df['sepal length (cm)'] = pd.qcut(df['sepal length (cm)'], 4 , [1,2,3,4])
    df['sepal width (cm)'] = pd.qcut(df['sepal width (cm)'], 4, [0, 1, 2, 3])
    xtrain, xtest, ytrain, ytest = train_test_split(df, y)

    ee = MeanEncoder(['sepal length (cm)' , 'sepal width (cm)'])

    a, b = ee.fit_transform(xtrain, xtest, ytrain)
    
    print(xtrain.shape, a.shape)
    print(xtest.shape , b.shape)



