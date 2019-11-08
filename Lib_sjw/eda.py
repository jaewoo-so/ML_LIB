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

def check_unique_count(df_src , count):
    '''
    only full dataframe, and unique count 
    check_unique_count(df_tail , 2)
    '''
    cols = df_src.columns
    under_10 = []
    under_10_names = []
    print('------ Size ------')
    for col in cols:
        size = df_src[col].unique().size
        print('{:20s} : {:3}'.format(col , size))
        
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