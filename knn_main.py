from model import k_Nearest_Neighbour
from data_utils import DataManager
import time

if __name__ == "__main__":

    fold = 5
    data_manager = DataManager('/DATA2/data/kncui/cifar-10-batches-py/')
    data_manager.load_and_cv(fold=fold)


    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
    k_to_accurancie = {}


    for k_ in   k_choices:
        k_to_accurancie.setdefault(k_, [])
        print("start the experiment of k is {}".format(k_))

        for which_fold in range(1,fold+1):
            print("pick {:} th fold data as validation".format(which_fold))
            data_manager.pick_fold(which_fold=which_fold)
            knn = k_Nearest_Neighbour()
            time1 = time.time()
            knn.fit(data_manager.data_dict['images_train'],data_manager.data_dict['label_tr'])
            time2 = time.time()
            accuracy = knn.get_accuracy(k=k_,
                                        X=data_manager.data_dict['image_val'],
                                        Y=data_manager.data_dict['label_val'])
            k_to_accurancie[k_] = k_to_accurancie[k_] + [accuracy]
            print("k_ {:} fold {:} accuracy{:} time{:}".format(k_,which_fold,accuracy,time2-time1))

    for k_ in k_choices:
        print("k_","{:}".format(k_),"accuracy","{:}".format(k_to_accurancie[k_]))