import cv2
import dataset
import numpy as np
from numpy.random import seed
import dataset
#对于2.x版本的Tensorflow，我们可以将运行环境转换为1.x，并disable 2.x的特性
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

#epoch 每个Epoch就是将所有训练样本训练一次
#batch 将整个训练样本分成若干个Batch。
#iteration  训练一个Batch就是一次Iteration（这个概念跟程序语言中的迭代器相似）
#

print(cv2.__version__)



#指定随机种子
seed(10)
tf.random.set_random_seed(20)

#设置超参数
batch_size = 32            # 每次迭代32张图片。一共1000张
classes = ['cats', 'dogs'] # 标签
num_classes = len(classes) # 分类种类
validation_size = 0.2      # 验正集 占0.2
img_size = 64              # 使图片大小保存一致
num_channels = 3           # 3意思是图片为彩色
train_path = 'training_data' # 图片路径

#data_set_dog_cat为读取py文件的文件名
data = dataset.read_train_sets(train_path, img_size, classes, validation_size = validation_size)
print("----------读取数据完毕---------")
print("训练集数据长度" + str(len(data.train._labels)))
print("验证集数据长度" + str(len(data.valid._labels)))


#第一层卷积层卷积核大小以及卷积核数量
filter_size_conv1 = 3
num_filter_conv1 = 32
#第二层卷积层卷积核大小以及卷积核数量
filter_size_conv2 = 3
num_filter_conv2 = 32
#第三层卷积层卷积核大小以及卷积核数量
filter_size_conv3 = 3
num_filter_conv3 = 64
#第一层全连接层的深度
fc_layer_size = 1024


def create_weights(shape):
#生成高斯分布，方差为0.05，大小为shape数据
    return tf.Variable(tf.random_normal(shape, stddev=0.05))
#   return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
#生成大小为size，值为0.05的一维常量
def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

#input输入图像，num_put_channels通道数,conv_filter_size卷积核大小，最后一个为卷积核数量
def create_convolutional_layer(input, num_input_channels, conv_filter_size, num_filters):
#随机生成权重参数
    Weight = create_weights([conv_filter_size, conv_filter_size, num_input_channels, num_filters])
#随机生成偏置项
    biasese = create_biases(num_filters)
#进行操作，卷积过后，图像的shape未发生改变，因为padding取为SAME,shape为[-1, 64，64，32],-1代表让计算机计算图片的数量是多少
    layer = tf.nn.conv2d(input, Weight, strides=[1, 1, 1, 1], padding='SAME')
    layer = tf.add(layer, biasese)
    #一般都用relu进行激活，由线性转化为非线性
    layer = tf.nn.relu(layer)
#池化，步长为2，则一次pooling后图片大小都变为原来两倍，深度不变，此时shape为[-1, 32, 32, 32]
    pooling = tf.nn.max_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return pooling

def create_flatten_layer(layer):
#获取当前图像的shape值为[num_iamges, width, height, channels]
    layer_shape = layer.get_shape()
    #切片方式获得后三个数据，并且获得总数
    num_features = layer_shape[1:4].num_elements()
#将原来的layer，resiae为规定要求（-1，为计算机自动计算num_iamges）
    layer = tf.reshape(layer, [-1, num_features])
    return layer

#注意的是activation_function的值，因为有两层全连接层，第一层需要进行激活，但是最后一个全连接就不需要了，因为需要获得最后未处理的结果
def create_fully_connection(inputs, num_inputs, num_outputs, activation_function=True):
    weight = create_weights([num_inputs, num_outputs])
    biases = create_biases(num_outputs)
    fully_connection = tf.add(tf.matmul(inputs, weight), biases)
    #dropout神经元，减少过拟合的风险
    fully_connection = tf.nn.dropout(fully_connection, rate = 0.3)
    if activation_function is True:
        fully_connection = tf.nn.relu(fully_connection)
    return fully_connection

#因为卷积输入的shape要四个参数，因此shape如下
x_data = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x_data')
#为二分类问题最后输出结果为num_classes个结果
y_data = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_data')
#获得结果中最大值的索引，0代表列中最大，1代表行最大
y_data_class = tf.argmax(y_data, 1)

layer_conv1 = create_convolutional_layer(input=x_data, num_input_channels=num_channels, conv_filter_size=filter_size_conv1, num_filters=num_filter_conv1)
layer_conv2 = create_convolutional_layer(input=layer_conv1, num_input_channels=num_filter_conv1, conv_filter_size=filter_size_conv2, num_filters=num_filter_conv2)
layer_conv3 = create_convolutional_layer(input=layer_conv2, num_input_channels=num_filter_conv2, conv_filter_size=filter_size_conv3, num_filters=num_filter_conv3)
layer_flat = create_flatten_layer(layer_conv3)
fc_1 = create_fully_connection(inputs=layer_flat, num_inputs=layer_flat.get_shape()[1:4].num_elements(), num_outputs=fc_layer_size, activation_function=True)

fc_2 = create_fully_connection(inputs=fc_1, num_inputs=fc_layer_size, num_outputs=num_classes, activation_function=False)

prediction = tf.nn.softmax(fc_2, name='prediction')
prediction_class = tf.argmax(prediction, 1)
cross_entrory = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_data, logits=fc_2)
loss = tf.reduce_mean(cross_entrory)
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)



sess = tf.Session()
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter("./tensorboard", sess.graph)


#比较预测值与真实值，返回True与False
correct_prediction = tf.equal(y_data_class, prediction_class)
#将True与False转化为float32，True为1，False为0，求均值就是准确率
accurary = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def show_progress(epoch, feed_dict_train, feed_dict_valid, train_loss, i):
    acc = sess.run(accurary, feed_dict=feed_dict_train)
    val_acc = sess.run(accurary, feed_dict=feed_dict_valid)
    print("epoch:", str(epoch + 1) + ",i:", str(i) +",acc:", str(acc) + ",val_acc:", str(val_acc) + ",train_loss:", str(train_loss))

total_iteration = 0
saver = tf.train.Saver()

def train(num_iteration):
#迭代总数
    global total_iteration
    for i in range(total_iteration, num_iteration+total_iteration):#（0， 8000）
   # next_batch自定义函数，每次获取batch_size大小的数据，训练集
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        #每次获取batch_size大小的数据，验正集
        x_valid_batch, y_valid_true_batch, _, cls_valid_batch = data.train.next_batch(batch_size)
        #准备喂入的训练数据
        feed_dict_train = {x_data:x_batch, y_data:y_true_batch}
        #准备喂入的验正集数据
        feed_dict_valid = {x_data:x_valid_batch, y_data:y_valid_true_batch}
        #每次run优化器，将loss减少
        sess.run(optimizer, feed_dict=feed_dict_train)
        #num_examples除以batch_size,就是有多少个epoch
        if i % int(data.train._num_examples/batch_size) == 0:
            train_loss = sess.run(loss, feed_dict=feed_dict_train)
            epoch = i/int(data.train._num_examples/batch_size)
            #打印
            show_progress(epoch, feed_dict_train, feed_dict_valid, train_loss, i)
            #保存网络
            saver.save(sess, './dog-cat-model/cat-dog.ckpt', global_step=i)

train(8000)