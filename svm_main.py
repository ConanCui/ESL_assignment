from data_utils import DataManager
from sklearn import svm,metrics
import numpy as np
from time import time



def main():
    fold = 5
    #need attention !!
    #Support Vector Machine algorithms are not scale invariant,
    #so it is highly recommended to scale your data.
    #For example, scale each attribute on the input vector X to [0,1]
    #or [-1,+1], or standardize it to have mean 0 and variance 1.
    #load, split, normalize CIFAR-10
    data_manager = DataManager('/DATA2/data/kncui/cifar-10-batches-py/')
    data_manager.load_and_cv(fold=fold,is_subsample=True)
    data_manager.pick_fold(which_fold=1,is_scaled=True)
    print('loading data finished')
    # Create a classifier: a support vector classifier
    # Penalty parameter C of the error term(smaller it is, more complex the svm is), gama is the parameters of RBF
    classifier = svm.SVC(gamma=0.001,C = 1)

    # learn on training sets
    classifier.fit(data_manager.data_dict['image_tr'],
                   data_manager.data_dict['label_tr'])

    predicted = classifier.predict(data_manager.data_dict['image_val'])
    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(data_manager.data_dict['label_val'],
                                                       predicted)))
    print((data_manager.data_dict['label_val']==predicted).sum())

def cross_validation_rbf():
    fold = 5
    #need attention !!
    #Support Vector Machine algorithms are not scale invariant,
    #so it is highly recommended to scale your data.
    #For example, scale each attribute on the input vector X to [0,1]
    #or [-1,+1], or standardize it to have mean 0 and variance 1.
    #load, split, normalize CIFAR-10
    data_manager = DataManager('/DATA2/data/kncui/cifar-10-batches-py/')
    data_manager.load_and_cv(fold=fold,is_subsample=True)

    C_ = [1,0.01,0.1]
    gama_ = [0.001, 0.01, 0.1]
    accuracy_C_gama = {}

    for C in C_:
        accuracy_C_gama.setdefault(C,{})
        for gama in gama_:
            accuracy_C_gama[C].setdefault(gama, [])
            time1 = time()
            for which_fold in range(1, fold + 1):
                data_manager.pick_fold(which_fold=which_fold, is_scaled=True)
                classifier = svm.SVC(gamma=gama, C=C)
                classifier.fit(data_manager.data_dict['image_tr'],
                           data_manager.data_dict['label_tr'])
                predicted = classifier.predict(data_manager.data_dict['image_val'])
                accuracy = float(np.sum(predicted == data_manager.data_dict['label_val'])/len(predicted))
                accuracy_C_gama[C][gama] = accuracy_C_gama[C][gama] + [accuracy]
                print("C", "{:}".format(C),"gama", "{:}".format(gama), "accuracy", "{:}".format(accuracy),time()-time1)

    for C in C_:
        for gama in gama_:
            print("C", "{:}".format(C),"gama", "{:}".format(gama), "accuracy", "{:}".format(accuracy_C_gama[C][gama]))



def cross_validation_linear():
    fold = 5
    #need attention !!
    #Support Vector Machine algorithms are not scale invariant,
    #so it is highly recommended to scale your data.
    #For example, scale each attribute on the input vector X to [0,1]
    #or [-1,+1], or standardize it to have mean 0 and variance 1.
    #load, split, normalize CIFAR-10
    data_manager = DataManager('/DATA2/data/kncui/cifar-10-batches-py/')
    data_manager.load_and_cv(fold=fold,is_subsample=True)

    C_ = [1,0.01,0.1]
    accuracy_C_gama = {}

    for C in C_:
        accuracy_C_gama.setdefault(C,[])
        time1 = time()
        for which_fold in range(1, fold + 1):
            data_manager.pick_fold(which_fold=which_fold, is_scaled=True)
            classifier = svm.LinearSVC(C=C)
            classifier.fit(data_manager.data_dict['image_tr'],
                       data_manager.data_dict['label_tr'])
            predicted = classifier.predict(data_manager.data_dict['image_val'])
            accuracy = float(np.sum(predicted == data_manager.data_dict['label_val'])/len(predicted))
            accuracy_C_gama[C] = accuracy_C_gama[C] + [accuracy]
            print("C", "{:}".format(C), "accuracy", "{:}".format(accuracy),"time_consuming",time()-time1)

    for C in C_:
        print("C", "{:}".format(C), "accuracy", "{:}".format(accuracy_C_gama[C]))
if __name__ == '__main__':
    cross_validation_linear()
    # cross_validation_rbf()