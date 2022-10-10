import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import glob
from sklearn.utils import shuffle
import numpy as np


class DataSet(object):
    #构造函数
    def __init__(self, images, labels, img_names, cls):
    #获取图像的总数量
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._img_names = img_names
        self._cls = cls
        #目前正在第几个epoch
        self._epochs_done = 0
        #在每个epoch里面，正在处理第几张图像
        self._index_in_epoch = 0
        #方便训练文件py获取数据
        def images(self):
            return self._images

        def labels(self):
            return self._labels

        def img_names(self):
            return self._img_names

        def cls(self):
            return self._cls

        def num_example(self):
            return self._num_examples

        def epochs_done(self):
            return self._epochs_done

    #每次获取batch_size大小的图像
    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        #如果大于总数，就重新开始
        if self._index_in_epoch > self._num_examples:
            # After each epoch we update this
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        #返回数据
        return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]


def load_train(train_path, img_size, classes):
    images = []
    labels = []
    img_names = []
    cls = []

    for fields in classes:#['cat', 'dog']
        index = classes.index(fields)#当前是猫或者狗的索引
        path = os.path.join(train_path, fields, '*g')#拼接字符串，获得绝对路径
        files = glob.glob(path)#获取路径下满足条件的所有文件
        for f1 in files:#遍历每一张图
            try:
                image = cv2.imread(f1)
            #异常处理
            except:
                print("读取异常")
            try:
            #resize为64， 64大小的图
                image = cv2.resize(image, (img_size, img_size), 0, 0, cv2.INTER_LINEAR)
            except:
                print("resize异常")
                #转化为float32，
            image = image.astype(np.float32)
            #归一化处理
            image = np.multiply(image, 1.0 / 255.0)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            #获取路径名字
            fibase = os.path.basename(f1)
            img_names.append(fibase)
            cls.append(fields)
            #由list转化为ndarray格式，方便管理
    images = np.array(images)
    labels = np.array(labels)
    cls = np.array(cls)
    img_names = np.array(img_names)
    return images, labels, img_names, cls

def read_train_sets(train_path, imag_size, classes, validation_size):
    class DataSets(object):
        pass
      #声明对象
    data_sets = DataSets()
    #加载数据
    images, labels, img_names, cls = load_train(train_path, imag_size, classes)
    images, labels, img_names, cls = shuffle(images, labels, img_names, cls)
    #判断valiation_size变量类型
    if isinstance(validation_size, float):#valiation_size为0.2
        #0.2乘以图像数量
        validation_size = int(validation_size * images.shape[0])
    #切片切分数据
    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    validation_img_name = img_names[:validation_size]
    validation_cls = cls[:validation_size]

    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    train_img_names = img_names[validation_size:]
    train_cls = cls[validation_size:]
    #转到类里面的构造函数，初始化变量
    data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
    data_sets.valid = DataSet(validation_images, validation_labels, validation_img_name, validation_cls)
    #返回已经分类的数据
    return data_sets
