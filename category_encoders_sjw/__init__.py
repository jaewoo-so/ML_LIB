"""

.. module:: category_encoders
  :synopsis:
  :platform:

"""

from category_encoders_sjw.backward_difference import BackwardDifferenceEncoder
from category_encoders_sjw.binary import BinaryEncoder
from category_encoders_sjw.count import CountEncoder
from category_encoders_sjw.hashing import HashingEncoder
from category_encoders_sjw.helmert import HelmertEncoder
from category_encoders_sjw.one_hot import OneHotEncoder
from category_encoders_sjw.ordinal import OrdinalEncoder
from category_encoders_sjw.sum_coding import SumEncoder
from category_encoders_sjw.polynomial import PolynomialEncoder
from category_encoders_sjw.basen import BaseNEncoder
from category_encoders_sjw.leave_one_out import LeaveOneOutEncoder
from category_encoders_sjw.target_encoder import TargetEncoder
from category_encoders_sjw.woe import WOEEncoder
from category_encoders_sjw.m_estimate import MEstimateEncoder
from category_encoders_sjw.james_stein import JamesSteinEncoder
from category_encoders_sjw.cat_boost import CatBoostEncoder
from category_encoders_sjw.mean_encoder import MeanEncoder

__version__ = '2.1.0'

__author__ = 'willmcginnis'

__all__ = [
    'BackwardDifferenceEncoder',
    'BinaryEncoder',
    'HashingEncoder',
    'HelmertEncoder',
    'OneHotEncoder',
    'OrdinalEncoder',
    'SumEncoder',
    'PolynomialEncoder',
    'BaseNEncoder',
    'LeaveOneOutEncoder',
    'TargetEncoder',
    'WOEEncoder',
    'MEstimateEncoder',
    'JamesSteinEncoder',
    'CatBoostEncoder'
]
