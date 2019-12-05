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
