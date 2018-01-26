from six.moves import cPickle as pickle
import random
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
from PIL import Image
plt.switch_backend('agg')

class DataManager(object):

  "different fold may have unbalance problem for the class"
  def __init__(self,root_name):
    self.root_name = root_name
    pass

  def _load_CIFAR10_batch(self,filename):
    '''load data from single CIFAR-10 file'''

    with open(filename, 'rb') as f:
      if sys.version_info[0] < 3:
        dict = pickle.load(f)
      else:
        dict = pickle.load(f, encoding='latin1')
      x = dict['data']
      y = dict['labels']
      x = x.astype(float)
      y = np.array(y)
    return x, y

  def _mean_sub(self):

    mean_image = np.mean(self.data_dict['image_tr'],axis=0)
    self.data_dict['image_tr'] -= mean_image
    self.data_dict['image_val'] -= mean_image

  def _scale_image(self):
    self.data_dict['image_tr'] /= 255
    self.data_dict['image_val'] /= 255
    self.data_dict['image_test'] /= 255

  def load_data(self, is_scaled = True, is_subsample = True):
    '''
    :param is_scaled:  scale data to [0,1] by (value-min)/(max-min)
    :param is_subsample: subsample data for code execution for some model, such as svm and knn
    :return:
    '''
    '''load all CIFAR-10 data '''

    image = []
    label = []
    for i in range(1, 6):
      filename = self.root_name + 'data_batch_' + str(i)
      X, Y = self._load_CIFAR10_batch(filename)
      image.append(X)
      label.append(Y)

    images_tr = np.concatenate(image)
    labels_tr = np.concatenate(label)

    image_test, label_test = self._load_CIFAR10_batch(self.root_name+'test_batch')


    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck']

    if is_scaled:
      self._scale_image()
    # # Normalize Data
    # mean_image = np.mean(x_train, axis=0)
    # x_train -= mean_image
    # x_test -= mean_image
    if is_subsample:
      end_index = 5000
    else:
      end_index = -1

    self.data_dict = {
      'images_train': images_tr[0:end_index],
      'labels_train': labels_tr[0:end_index],
      'images_test': image_test[0:end_index],
      'labels_test': label_test[0:end_index],
      'classes': classes
    }
    a = 0
    # self._prepare_indices()

  def load_and_cv(self,fold, is_subsample = False):
    '''load all CIFAR-10 data and merge training  and validation batches'''
    self.fold = fold
    if is_subsample:
      end_index_tr_val = 5000
      end_index_test = 500
    else:
      end_index_tr_val = 50000
      end_index_test = 1000

    image_train_val = []
    label_train_val = []
    for i in range(1, 6):
      filename = self.root_name + 'data_batch_' + str(i)
      X, Y = self._load_CIFAR10_batch(filename)
      image_train_val.append(X)
      label_train_val.append(Y)

    image_test, label_test = self._load_CIFAR10_batch(self.root_name + 'test_batch')


    # split data to folds
    self.images_fold = np.array_split(np.concatenate(image_train_val)[0:end_index_tr_val], fold)
    self.labels_fold = np.array_split(np.concatenate(label_train_val)[0:end_index_tr_val],fold)

    # save the test data
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck']
    self.data_dict = {
      'image_test': image_test[0:end_index_test],
      'label_test': label_test[0:end_index_test],
      'classes': classes
    }
  def pick_fold(self,which_fold,is_scaled= True):
    '''use the (which_fold) th fold as the validation, remaining as the training data
    which_fold = 1,2,3,4,5.... num of folds
    attention: the size of five fold must be same, or it will cause error
    '''
    image_val = self.images_fold[which_fold-1]
    label_val = self.labels_fold[which_fold-1]

    if which_fold-1 ==0:
      image_tr = np.vstack(self.images_fold[which_fold:]).reshape((-1, image_val.shape[-1]))
      label_tr = np.vstack(self.labels_fold[which_fold:]).reshape((-1))
    elif which_fold ==self.fold:
      image_tr = np.vstack(self.images_fold[0:which_fold-1]).reshape((-1,image_val.shape[-1]))
      label_tr = np.vstack(self.labels_fold[0:which_fold-1]).reshape((-1))
    else:
      image_tr = np.vstack((self.images_fold[0:which_fold - 1], self.images_fold[which_fold:])).reshape((-1, image_val.shape[-1]))
      label_tr = np.vstack((self.labels_fold[0:which_fold - 1], self.labels_fold[which_fold:])).reshape((-1))

    self.data_dict['image_val'] = image_val
    self.data_dict['label_val'] = label_val
    self.data_dict['image_tr'] = image_tr
    self.data_dict['label_tr'] = label_tr

    if is_scaled:
      self._scale_image()

    self._prepare_indices()


  def get_trainsize(self):
    return len(self.data_dict['image_tr'])

  def next_batch(self,batch_size):

    batch_image = []
    batch_label = []

    for i in range(batch_size):
      index = self.image_indices[self.used_image_index]
      image = self.data_dict['image_train'][index]
      batch_image.append(image)

      label = self.data_dict['label_train'][index]
      batch_label.append(label)

      self.used_image_index += 1

      if self.used_image_index >= self.get_trainsize():
        self._prepare_indices()

    return batch_image,batch_label

  def _prepare_indices(self):
    self.image_indices = list(range(self.get_trainsize()))
    random.shuffle(self.image_indices)
    self.used_image_index =0

  def show_examples(self):
    num_classes = len(self.data_dict['classes'])
    samples_per_class = 7
    for y, cls in enumerate(self.data_dict['classes']):
        idxs = np.flatnonzero(self.data_dict['label_tr'] == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(self.data_dict['image_tr'][idx].reshape(3,32,32).transpose(1,2,0).astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    # plt.show()
    plt.savefig('Sample.png')

class TestDataManager(Dataset):
  '''test dataset'''
  def __init__(self, root_dir, transform):
    self.root_dir = root_dir
    self.transform = transform
  def __len__(self):
    return 2175
  def __getitem__(self, idx):
    num2str = str(idx)
    str_len = len(num2str)
    for i in range(4-str_len):
      num2str = '0' + num2str
    img_name = os.path.join(self.root_dir,'img-'+num2str+'.jpg',)
    image = Image.open(img_name)
    sample = image

    if self.transform:
      sample = self.transform(sample)
    return sample


if __name__ == '__main__':
  pass


