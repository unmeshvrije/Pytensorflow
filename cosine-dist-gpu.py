import tensorflow as tf
# Creates a graph.


with tf.device('/gpu:1'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3,2], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0,1.0,2.0,3.0,4.0,5.0,6.0], shape=[6,2], name='b')
    a_normed = tf.nn.l2_normalize(a, dim=1)
    b_normed = tf.nn.l2_normalize(b, dim=1)

    c = tf.matmul(b_normed, tf.transpose(a_normed, [1,0]))

# Creates a session with log_device_placement set to True.
# Setting soft_placement to true will allow tensorflow to choose whichever device is available
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print (sess.run(c))
