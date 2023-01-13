import numpy as np
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import matplotlib
from matplotlib import pyplot as plt
from functools import *
from pathlib import Path
import seaborn as sns

# -- helper
def check_data_type(unique_number):
    if unique_number == 2:
        return 'binary'
    elif unique_number < 50:
        return 'category'
    else:
        return 'numeric'

def first_look(df):
    '''
    'size':  ( 샘플 갯수 )
    'count':( null값 뺀 샘플 갯수)
    'nunique': (null 뺴고 유니크 값 카운팅)
    '''


    df_res = df.agg(['dtype' , 'size','count', 'nunique' , 'max' , 'min' ]).T
    df_res['col_name'] = df_res.index
    df_res = df_res[['col_name' , 'dtype' , 'size','count', 'nunique' , 'max' , 'min']]

    df_res['null_count'] = df_res['size'] - df_res['count']
    df_res['null_ratio'] =  df_res['null_count'] / df_res['size']
    df_res['data_type'] = df_res['nunique'].apply(lambda x: check_data_type(x)) 
    df_res= df_res.reset_index(drop=True)
    return df_res

def grouping_columns(df_res):
    '''
    get columns list of 3 catergory : 'binary' , 'category' , 'numeric'
    input : Result of first_look , null_value_check
    dataframe with column name : 'data_type' 
    ''' 
    res_dic = {}
    df_group = df_res.groupby(by = 'data_type')
    for k in df_group.groups.keys():
        res_dic[k] = df_group.get_group(k)['col_name'].values.tolist()
        #res_dic[k] = df_res.loc[df_res['data_type'] == k]['col_name'].values.tolist() # 같은 기능
    return res_dic

def null_value_check(df_src , is_save = False):
    #eda.null_value_check(df_src)

    #  null값이 있는 컬럼만 가져오기
    null_cols = df_src.isna().sum(axis = 0).where(lambda x :  x > 0).dropna().index.tolist()
    null_count = df_src.isna().sum(axis = 0).where(lambda x :  x > 0).dropna()

    df_null = pd.DataFrame()
    df_null['col_name'] = null_count.index
    df_null['count'] = null_count.values
    df_null['ratio%'] = df_null['count'] / df_src.shape[0] * 100

    # null 값이 있는 컬럼에 대한 유니크 값 조사
    df_include_null = df_src[null_cols]
    df_include_null = df_include_null.agg(['dtype' ,  'nunique' , ]).T
    df_include_null['col_name'] = df_include_null.index
    df_include_null = df_include_null[['col_name' , 'dtype' , 'nunique' ]]
    df_include_null['data_type'] = df_include_null['nunique'].apply(lambda x: check_data_type(x)) 

    #df_res= df_res.reset_index(drop=True)
    df_res_type = pd.merge(df_null , df_include_null , on = 'col_name' )

    # plot null count
    ax = null_count.plot(kind = 'barh' , figsize = (14,8))
    x_offset = +0.04
    y_offset = +0.02
    for p in ax.patches:
        b = p.get_bbox()
        val = "{:.0f}".format(b.x1)  
        ax.annotate(val, (b.x1 + x_offset, (b.y0 +b.y1)/2))
    ax.set_title('Count Null Value ', fontsize = 24)
    plt.tight_layout()
    if is_save : plt.savefig('Count Null Value.png')

    return df_res_type

def check_unique_count(df_src , count):
    '''
    only full dataframe, and unique count 
    check_unique_count(df_tail , 2)
    '''
    cols = df_src.columns
    under_10 = []
    under_10_names = []

    unique_list = []

    print('------ Size ------')
    for col in cols:
        size = df_src[col].unique().size
        print('{:20s} : {:3}'.format(col , size))
        unique_list.append(size)
        if size <= count:
            under_10.append(size)
            under_10_names.append(col)
            
    print('------ Size Under {} ------'.format(count))
    for v , n in zip(under_10 , under_10_names):
        print('{:20s} : {:3}'.format(n , v))
    print('--- Size Over {} list ---'.format(count))    
    print( [ x for x in cols if x not in under_10_names] )
    print('--- Size Under {} list ---'.format(count))
    print(under_10_names)
    print()

    df_uni = pd.DataFrame()
    df_uni['name'] = cols
    df_uni['size'] = unique_list

    return df_uni


def plot_binary_value(df_src , col_binary):
    '''
    need to pass only binary value columns names
    '''
    bin_col = col_binary
    zero_list = []
    one_list = []
    for col in col_binary:
        zero_list.append((df_src[col]==0).sum())
        one_list.append((df_src[col]==1).sum())
        
        
    trace1 = go.Bar(
        x=bin_col,
        y=zero_list ,
        name='zero count'
    )
    trace2 = go.Bar(
        x=bin_col,
        y=one_list,
        name='one count'
    )
    
    data = [trace1, trace2]
    layout = go.Layout(
        barmode='stack',
        title='Count of 1 and 0 in binary variables'
    )
    
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='stacked-bar')






# color map을 가져온 후에 , 내 데이터를 컬러맵의 범위로 바꿔준다.
import matplotlib
from functools import *
from pathlib import Path

# for continuos values feature
def dist_compare(df_list , color_list , df_name_list , col_list , figsize = (14,8) , save_fig = False, prefix = 'Distribution'):
    '''
    cmap = matplotlib.cm.get_cmap('tab10')
    color_list = cmap.colors[:3]
    '''
    assert len(df_list) == len(color_list) 
    counts = len(df_list)
    
    for col in col_list:
        data_list = [ x[col] for x in df_list ]
        fig , ax = plt.subplots(figsize = figsize)
        
        for i in range(len(color_list)):
            sns.distplot(data_list[i] , color = color_list[i] , ax = ax , label = df_name_list[i] )
            
        
        ax.set_title('{} Distribution'.format(col))
        ax.legend()
        Path('dist_compare').mkdir(parents=True, exist_ok=True) 
        if save_fig:
            Path('./dist_compare/').mkdir(patrent = True)
            plt.savefig( './dist_compare/{}_{}.png'.format(col , prefix) )
        plt.show()
        
        
# for categorical values feature
def count_compare(df_list , df_name_list , col_list , color_map = 'tab20c' , figsize = (12,4) , save_fig = False,  prefix = 'All_Counts_Ratio'):
    '''
    #example   
    dfs = [df_a , df_tr , df_te]
    df_names = ['all' , 'train' , 'test']
    cols_i = ['Gender', 'Smoking', 'ECOG', 'pathology' ]
    count_compare(dfs , df_names , cols_i)
    '''
    collen = len(col_list)
    counts = len(df_list)
    fig , ax = plt.subplots(collen , 2 , figsize = (12,4*collen) )
    
    for c, col in enumerate(col_list):
        data_list = [ x[col].values.astype(np.int) for x in df_list ]
        
        
        len_list = [len(x) for x in data_list]
        
        labels = [ [df_name_list[i]]*len_list[i] for i in range(len(len_list)) ]

        # unroll
        #data_list_com =  list(reduce(lambda f , s : f + s , data_list ))
        data_list_com =  np.concatenate( data_list )
        labels_com =  list(reduce(lambda f , s : f + s , labels ))

        df_cat = pd.DataFrame( )
        df_cat['df_name'] = labels_com
        df_cat['value'] = data_list_com
        
        # simple count plot
        sns.countplot(x = 'value' , hue = 'df_name' , ax = ax[c,0] , data = df_cat , palette=color_map )   
        ax[c,0].set_title('{} Counts Plot'.format(col))

        
        x , y , hue = 'value' , 'ratio' , 'df_name' 
        df_cat[x].groupby(df_cat[hue]) \
                .value_counts(normalize=True) \
                .rename(y) \
                .reset_index() \
                .pipe((sns.barplot, "data"), x=x,y=y, hue=hue , ax = ax[c,1] , palette=color_map)

        #sns.barplot(x=x, y=y, hue=hue, data=prop_df, ax=ax)


        ax[c,1].set_title('{} Ratio Plot'.format(col))
        
        ax[c,0].legend()
        ax[c,1].legend()
        Path('count_compare').mkdir(parents=True, exist_ok=True) 
    
    plt.tight_layout()
    if save_fig:
        Path('./count_compare/').mkdir(patrent = True)
        plt.savefig( './count_compare/{}.png'.format(col,prefix) )
    plt.show()