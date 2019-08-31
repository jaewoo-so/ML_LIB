from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from matplotlib import pyplot as plt
import numpy as np

## method : calulate P-R 
'''

prs , rcs , avgpr , thres= GetPR(ytest,y_score,n_classes)
ytest = onehot

'''
def GetPR(Y_test,y_score,n_classes):
    precision = dict()
    recall = dict()
    thres = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], thres[i] = precision_recall_curve(Y_test[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])
    
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
        y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                         average="micro")
    average_precision["macro"] = average_precision_score(Y_test, y_score,
                                                         average="macro")
    print('Average precision score, micro-averaged over all classes: {0:0.4f}'.format(average_precision["micro"]))
    print('Average precision score, macro-averaged over all classes: {0:0.4f}'.format(average_precision["macro"]))
    return precision , recall , average_precision , thres




## method : visualize 
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


'''
ytest2 = np.argmax(ytest , axis = 1)
y_score2 = np.argmax(y_score , axis = 1)

plot_confusion_matrix(ytest2 , y_score2 , ys_names , figsize = (12,12))
plt.show()
one hot으로 하면 안된다. 
'''

def plot_confusion_matrix(y_true, y_pred, classes,normalize=False,title=None , figsize = (20,20),cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    

    fig, ax = plt.subplots( figsize = figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.4f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return cm , ax 

# method : plot 
'''
prplot(prs,rcs , avgpr , list(range(n_classes)) , ys_names.tolist() , figsize = (18,24))
plt.show()

사용한다. 

'''
import matplotlib

# class_sample_num = key : name , val : class sample num
def prplot(precision , recall , average_precision, n_classes , class_names = None , class_sample_num = None , figsize = (20, 20)):
    #cmap = matplotlib.cm.get_cmap('Spectral')
    cmap = matplotlib.cm.get_cmap('terrain')
    rgba = cmap(0.5)
    
    plt.figure(figsize=figsize)
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.4f})'''.format(average_precision["micro"]))
    lines.append(l)
    labels.append('macro-average Precision-recall (area = {0:0.4f})'''.format(average_precision["macro"]))
    
    for count,i in enumerate(n_classes):
        l, = plt.plot(recall[i], precision[i], color=cmap(count/len(n_classes)), lw=2)
        lines.append(l)
        
        name = class_names[i] if class_names != None else i  
        
        if class_sample_num == None:
            labels.append('Precision-recall for class {0} (area = {1:0.4f})'
                      ''.format(name, average_precision[i]))
        else:
            info = class_sample_num[name] 
            labels.append('Precision-recall for class {0} (area = {1:0.4f} , sample size = {2})'
                      ''.format(name, average_precision[i] , info))
        
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc=(0.01, 0.01), prop=dict(size=14))
    


    

