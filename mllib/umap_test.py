import sys
sys.path.append('D:\01_mylib\ML_LIB')

import os
print(os.environ.get("PATH"))

import os
os.chdir(os.path.dirname(__file__))
import umapLib 
import numpy as np

xs = np.random.randint(0 , 10 , (100, 10 ))
ys = np.zeros(100)
ys[50:] = 1




#umapLib.plot_umap_unsupervised(xs,ys , True , 4 , 0.3 , 'kulsinski', figsize = (8,6))

print('done')
dist_list = umapLib.dist_list
n_neighbor_list = umapLib.n_neighbor_list
min_dist_list = umapLib.min_dist_list


umapLib.umap_combination_unsupervised(xs,ys , dist_list ,  n_neighbor_list , np.arange(0.1 , 1.0 , 0.2) ,  (6,6) , True )