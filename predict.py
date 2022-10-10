"""
tensorflow 总结如下
1.张量在Tensorflow中的实现并不是直接采用数组的形式
它只是对Tensorflow中运算结果的引用，在张量中并没有真正保存数字，它保存的是如何得到这些数字的计算过程。
2.有一个计算图的概念，可以理解为普通的计算过程，TensorFlow对计算做了并行优化
3.启动一个session才开始计算
"""

import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import argparse

image_size = 64
num_channels = 3
images= []

path = 'test_data/dog_test3.jpeg'
image = cv2.imread(path)
image = cv2.resize(image, (image_size, image_size), 0 ,0, cv2.INTER_LINEAR)
images.append(image)
images = np.array(images, dtype = np.uint8)
images = np.multiply(images, 1.0 / 255.0)

x_batch = images.reshape(1,image_size,image_size, num_channels)
sess = tf.Session()
saver = tf.train.import_meta_graph('./dog-cat-model/cat-dog.ckpt-7900.meta')
saver.restore(sess,'./dog-cat-model/cat-dog.ckpt-7900')

graph = tf.get_default_graph()

y_pred = graph.get_tensor_by_name("prediction:0")
x = graph.get_tensor_by_name("x_data:0")
y_true = graph.get_tensor_by_name("y_data:0")
y_test_images = np.zeros((1, 2))
feed_dict_testing = {x:x_batch, y_true:y_test_images}
result = sess.run(y_pred, feed_dict = feed_dict_testing)

res_label = ['cat', 'dog'] # 标签
print(res_label[result.argmax()])