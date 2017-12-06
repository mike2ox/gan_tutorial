import tensorflow as tf
import numpy as np
import datetime

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

z_dimensions = 100 # == noise
total_epoch = 2000
batch_size = 50

real_input = 28*28
n_noise = 128

X = tf.placeholder(tf.float32, shape=[None, 28,28,1], name='X_place')
# Y = tf.placeholder(tf.float32, [None, n_class])
Z = tf.placeholder(tf.float32, [None, z_dimensions], name='Z_place')



# 일반 CNN의 느낌?
def discriminator(data, reuse=None):
    """
    Args
    :param data:
    :param labels:
    :param reuse:

    return:
     input data를 구별한 결과(0~1)
    """
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()

        # CNN layer w,b
        d_w1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.01, name='d_w1'))
        d_b1 = tf.Variable(tf.zeros([32], name='d_b1'))
        d_w2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.01, name='d_w2'))
        d_b2 = tf.Variable(tf.zeros([64], name='d_b2'))

        # FC layer w,b
        d_w3 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.01, name='d_w3'))
        d_b3 = tf.Variable(tf.zeros([1024],  name='d_b3'))
        d_w4 = tf.Variable(tf.truncated_normal([1024, 1], stddev=0.01, name='d_w4'))
        d_b4 = tf.Variable(tf.zeros([1], name='d_b4'))

        # data : 28 * 28 fixel img
        d_L1 = tf.nn.conv2d(input=data, filter=d_w1, strides=[1,1,1,1], padding='SAME')
        d_L1 = tf.nn.relu(d_L1 + d_b1)
        d_L1 = tf.nn.avg_pool(d_L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        d_L2 = tf.nn.conv2d(input=d_L1, filter=d_w2, strides=[1,1,1,1], padding='SAME')
        d_L2 = tf.nn.relu(d_L2 + d_b2)
        d_L2 = tf.nn.avg_pool(d_L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        d_L3 = tf.reshape(d_L2, [-1, 7*7*64])
        d_L3 = tf.matmul(d_L3, d_w3)
        d_L3 = tf.nn.relu(d_L3 + d_b3)

        output = tf.matmul(d_L3, d_w4) + d_b4

        # checkpoint : 이부분이 확실한지 모르겠음. oreilly랑 tutorial 참조해서 한거라...
        # output = tf.nn.sigmoid(tf.matmul(d_L2, 1))

        return output

    '''
    inputs = tf.concat([data, labels], 1)
    hidden = tf.layers.dense(inputs, 256,
                             activation=tf.nn.relu)
    output = tf.layers.dense(hidden, 1,
                             activation=None)
    
    return output
    '''
# 역 CNN
def generative(z, batch_size, z_dim):
    # g_w1 = tf.Variable(tf.truncated_normal([z_dim, 3136], stddev=0.01, name='g_w1'), dtype=tf.float32)
    # g_b1 = tf.Variable(tf.truncated_normal([3136], stddev=0.01, name='g_b1'))
    g_w1 = tf.get_variable('g_w1', [z_dim, 3136], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b1 = tf.get_variable('g_b1', [3136], initializer=tf.truncated_normal_initializer(stddev=0.02))

    g_L1 = tf.matmul(z, g_w1) + g_b1
    g_L1 = tf.reshape(g_L1, [-1, 56, 56, 1])
    # with tf.variable_scope('bn1'):
    #    g_L1 = tf.nn.batch_normalization(g_L1, variance_epsilon=1e-5)
    g_L1 = tf.contrib.layers.batch_norm(g_L1, epsilon=1e-5, scope='bn1')

    g_L1 = tf.nn.relu(g_L1)

    # Generate 50 features
    # g_w2 = tf.Variable(tf.truncated_normal([3,3,1,z_dim/2], stddev=0.01, name='g_w2'), dtype=tf.float32)
    # g_b2 = tf.Variable(tf.truncated_normal([z_dim/2], stddev=0.01, name='g_b2'))

    g_w2 = tf.get_variable('g_w2', [3, 3, 1, z_dim / 2], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b2 = tf.get_variable('g_b2', [z_dim / 2], initializer=tf.truncated_normal_initializer(stddev=0.02))

    g_L2 = tf.nn.conv2d(g_L1, g_w2, strides=[1,2,2,1], padding='SAME')
    g_L2 = g_L2 + g_b2

    # with tf.variable_scope('bn2'):
    #    g_L2 = tf.nn.batch_normalization(g_L2, variance_epsilon=1e-5)
    g_L2 = tf.contrib.layers.batch_norm(g_L2, epsilon=1e-5, scope='bn2')

    g_L2 = tf.nn.relu(g_L2)
    # 이미지 resize
    g_L2 = tf.image.resize_images(g_L2, [56, 56])

    # Generate 25 features
    # g_w3 = tf.Variable(tf.truncated_normal([3, 3, z_dim / 2, z_dim / 4], stddev=0.01, name='g_w3'), dtype=tf.float32)
    # g_b3 = tf.Variable(tf.truncated_normal([z_dim / 4], stddev=0.01, name='g_b3'))

    g_w3 = tf.get_variable('g_w3', [3, 3, z_dim / 2, z_dim / 4], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b3 = tf.get_variable('g_b3', [z_dim / 4], initializer=tf.truncated_normal_initializer(stddev=0.02))

    g_L3 = tf.nn.conv2d(g_L2, g_w3, strides=[1, 2, 2, 1], padding='SAME')
    g_L3 = g_L3 + g_b3

    # with tf.variable_scope('bn3'):
    #    g_L3 = tf.nn.batch_normalization(g_L3, variance_epsilon=1e-5)
    g_L3 = tf.contrib.layers.batch_norm(g_L3, epsilon=1e-5, scope='bn3')

    g_L3 = tf.nn.relu(g_L3)
    # 이미지 resize
    g_L3 = tf.image.resize_images(g_L3, [56, 56])

    # Final convolution with one output channel
    #    g_w4 = tf.Variable(tf.truncated_normal([1, 1, z_dim / 4, 1], stddev=0.01, name='g_w4'), dtype=tf.float32)
    # g_b4 = tf.Variable(tf.truncated_normal([z_dim / 4], stddev=0.01, name='g_b4'))
    g_w4 = tf.get_variable('g_w4', [1, 1, z_dim / 4, 1], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b4 = tf.get_variable('g_b4', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))

    g_L4 = tf.nn.conv2d(g_L3, g_w4, strides=[1, 2, 2, 1], padding='SAME')
    g_L4 = g_L4 + g_b4
    g_L4 = tf.sigmoid(g_L4)

    # Dimensions of g4: batch_size x 28 x 28 x 1
    return g_L4

Gz = generative(Z, batch_size, z_dimensions)

Dx = discriminator(X)

Dg = discriminator(Gz, reuse=True)

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.ones_like(Dx)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.zeros_like(Dg)))
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg)))

# 식별자와 생성자에서 사용될 변수 리스트 생성
tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

# train 식별자
d_trainer_fake = tf.train.AdamOptimizer(0.0003).minimize(d_loss_fake, var_list=d_vars)
d_trainer_real = tf.train.AdamOptimizer(0.0003).minimize(d_loss_real, var_list=d_vars)

# Train 생성자
g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)

# checkpoint : 이걸 왜 써야 하는가? --> 식별자 안에 구현해 놓음 --> 다시 소생
# From this point forward, reuse variables
tf.get_variable_scope().reuse_variables()



####### 신경망 모델 학습
sess = tf.Session()

# Send summary statistics to TensorBoard
tf.summary.scalar('Generator_loss', g_loss)
tf.summary.scalar('Discriminator_loss_real', d_loss_real)
tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)

images_for_tensorboard = generative(Z, batch_size, z_dimensions)
tf.summary.image('Generated_images', images_for_tensorboard, 5)

merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

sess.run(tf.global_variables_initializer())

# Pre-train discriminator
for i in range(300):
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
    _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                           {X: real_image_batch, Z: z_batch})

# Train generator and discriminator together
for i in range(100000):
    real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])

    # Train discriminator on both real and fake images
    _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                           {X: real_image_batch, Z: z_batch})

    # Train generator
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    _ = sess.run(g_trainer, feed_dict={Z: z_batch})

    if i % 10 == 0:
        # Update TensorBoard with summary statistics
        z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
        summary = sess.run(merged, {Z: z_batch, X: real_image_batch})
        writer.add_summary(summary, i)

