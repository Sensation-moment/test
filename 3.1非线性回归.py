import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''
关于np.newaxis的理解:
x1 = np.array([1, 2, 3, 4, 5])
the shape of x1 is (5,)
x1_new = x1[:, np.newaxis]
now, the shape of x1_new is (5, 1)
array([[1],
       [2],
       [3],
       [4],
       [5]])
x1_new = x1[np.newaxis,:]
now, the shape of x1_new is (1, 5)
array([[1, 2, 3, 4, 5]])
'''

# 使用numpy来生成200个点 从-0.5到0.5均匀分布的200个点
# 后面的也即将一行200列转为200行一列(前面生成的数据存在列表的:里面，后面的np.newaxis即增加一个维度)
x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]
# 生成一些正态分布随机值 形状和上面一样
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data) + noise

# 定义两个placeholder 行不确定 但是一列确定
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

# 定义神经网络中间层 输入层是一个神经元 输出层是十个神经元
Weights_L1 = tf.Variable(tf.random_normal([1,10]))
biases_L1 = tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L1 = tf.matmul(x,Weights_L1) + biases_L1
# L1层的输出
L1 = tf.nn.tanh(Wx_plus_b_L1)

# 定义神经网络输出层 中间层有十个神经元 输出层有一个神经元
Weights_L2 = tf.Variable(tf.random_normal([10,1]))
# 输出层只有一个神经元 所以偏置值只有一个神经元
biases_L2 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2 = tf.matmul(L1,Weights_L2) + biases_L2
# 预测的结果要通过一个激活函数
prediction = tf.nn.tanh(Wx_plus_b_L2)

# 二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))
# 使用梯度下降法训练
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    #变量初始化
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})

    # 获得预测值
    prediction_value = sess.run(prediction,feed_dict={x:x_data})
    # 画图
    plt.figure()
    plt.scatter(x_data,y_data)
    # r-代表红色实线 宽度设置为5
    plt.plot(x_data,prediction_value,"r-",lw=5)
    plt.show()
