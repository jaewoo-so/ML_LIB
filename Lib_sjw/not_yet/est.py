import matplotlib.pyplot as plt
from time import time
import umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold, datasets

def Dimensionality_reduction_set(X,y,n_neighbors = 12 , n_components = 8 , use_umap = False, umap_neighber = 8 , umap_min_dist = 0.5 ):
    # result container
    name_box = []
    result_box = []

    # set color
    color = y

    min_dist_value = umap_min_dist
    nnb_value = umap_neighber

    fig = plt.figure(figsize=(15, 8))
    methods = ['standard', 'ltsa', 'hessian', 'modified']
    labels = ['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE']

    #start
    for i, method in enumerate(methods):
        if method == 'hessian':
            n_neighbors = int(n_components * (n_components + 3) / 2 + 1)
        else:
            n_neighbors = 12
        t0 = time()
        Y = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                            eigen_solver='auto',method=method).fit_transform(X)

        if use_umap:
            Y = umap.UMAP(n_neighbors=nnb_value,
                                  min_dist=min_dist_value,
                                  metric='euclidean').fit_transform(Y, color)

        t1 = time()
        print("%s: %.2g sec" % (methods[i], t1 - t0))

        ax = fig.add_subplot(252 + i)
        plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
        plt.title("%s (%.2g sec)" % (labels[i], t1 - t0))
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')
        name_box.append(labels[i])
        result_box.append(Y)

    n_neighbors = 12
    n_components = 8

    t0 = time()
    Y = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
    if use_umap:
        Y = umap.UMAP(n_neighbors=nnb_value,
                              min_dist=min_dist_value,
                              metric='euclidean').fit_transform(Y, color)
    t1 = time()
    print("Isomap: %.2g sec" % (t1 - t0))
    ax = fig.add_subplot(257)
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.title("Isomap (%.2g sec)" % (t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    name_box.append('Isomap')
    result_box.append(Y)

    t0 = time()
    mds = manifold.MDS(n_components, max_iter=100, n_init=1)
    Y = mds.fit_transform(X)
    if use_umap:
        Y = umap.UMAP(n_neighbors=nnb_value,
                              min_dist=min_dist_value,
                              metric='euclidean').fit_transform(Y, color)
    t1 = time()
    print("MDS: %.2g sec" % (t1 - t0))
    ax = fig.add_subplot(258)
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.title("MDS (%.2g sec)" % (t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    name_box.append('MDS')
    result_box.append(Y)

    t0 = time()
    se = manifold.SpectralEmbedding(n_components=n_components,
                                    n_neighbors=n_neighbors)
    Y = se.fit_transform(X)

    if use_umap:
        Y = umap.UMAP(n_neighbors=nnb_value,
                              min_dist=min_dist_value,
                              metric='euclidean').fit_transform(Y, color)
    t1 = time()
    print("SpectralEmbedding: %.2g sec" % (t1 - t0))
    ax = fig.add_subplot(259)
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.title("SpectralEmbedding (%.2g sec)" % (t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    name_box.append('SpectralEmbedding')
    result_box.append(Y)

    t0 = time()
    tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
    Y = tsne.fit_transform(X)
    if use_umap:
        Y = umap.UMAP(n_neighbors=nnb_value,
                              min_dist=min_dist_value,
                              metric='euclidean').fit_transform(Y, color)
    t1 = time()
    print("t-SNE: %.2g sec" % (t1 - t0))
    ax = fig.add_subplot(2, 5, 10)
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.title("t-SNE (%.2g sec)" % (t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    name_box.append('t-SNE')
    result_box.append(Y)

    plt.show()
    return result_box , name_box