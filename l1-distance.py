import tensorflow as tf
# Creates a graph.
import numpy as np

batch  = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

with tf.device('/gpu:1'):
    #a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3,2], name='a')
    a = tf.placeholder(tf.float32, shape=[1,2], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0,1.0,2.0,3.0,4.0,5.0,6.0], shape=[6,2], name='b')
    #a_normed = tf.nn.l2_normalize(a, dim=1)
    #b_normed = tf.nn.l2_normalize(b, dim=1)

    #c = b - tf.matmul(b, tf.transpose(a))
    absdiff = tf.abs(tf.subtract(a,b))
    c = tf.reduce_sum(absdiff, axis = 1)
# Creates a session with log_device_placement set to True.
# Setting soft_placement to true will allow tensorflow to choose whichever device is available
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.

l1diff = []
for b in batch:
    print ("b shape : ", b.shape)
    x = np.reshape(b, (1,2))
    print ("x shape : ", x.shape)
    feed_dict = {a:x}
    temp = sess.run(c, feed_dict = feed_dict)
    #print (temp)
    #print (temp.shape)
    result = np.reshape(temp, (1,6))
    print ("result shape : ", result.shape)
    l1diff.append(result[0])

for l in l1diff:
    print (l, "-X-")
