from dataclasses import dataclass
from typing import List
import pandas as pd 
import mllib
from dotwiz import DotWiz


# @dataclass
# class FixedTestData:
#     '''
#     Use train set and test set
#     '''
#     model_name: str
#     test_pred: DotMap
#     test_score: DotMap
#     oof_pred: DotMap
#     oof_score: DotMap
#     fold_model: mllib.model_interface

class FixedTestData:
    def __init__(self, fold_predict , fold_oof , fold_metric , fold_model ) -> None:
        self.result = DotWiz()
        self.result.test_pred = fold_predict 
        self.result.oof_pred = fold_oof 
        self.result.metric = fold_metric 
        self.result.model = fold_model 
        


class FixedTestResult:
    def __init__(self, dict_FixedTestData ) -> None:
        result = DotWiz()
        for k in dict_FixedTestData.keys():
            result[k] = dict_FixedTestData[k]
        self.result = result        


class FixedTestData2():
    
    def __init__(self) -> None:
        pass