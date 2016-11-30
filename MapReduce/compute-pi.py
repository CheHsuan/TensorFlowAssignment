import numpy as np
import tensorflow as tf
import sys

cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})

point_num = int(sys.argv[1])

x = tf.placeholder(tf.float32, point_num)
y = tf.placeholder(tf.float32, point_num)

with tf.device("/job:local/task:1"):
    batch_x1 = tf.slice(x, [0], [point_num/2])
    batch_y1 = tf.slice(y, [0], [point_num/2])
    result1 = tf.add(tf.square(batch_x1), tf.square(batch_y1))

with tf.device("/job:local/task:0"):
    batch_x2 = tf.slice(x, [point_num/2], [-1])
    batch_y2 = tf.slice(y, [point_num/2], [-1])
    result2 = tf.add(tf.square(batch_x2), tf.square(batch_y2))
    distance = tf.concat(0,[result1, result2])

with tf.Session("grpc://localhost:2222") as sess:
    result = sess.run(distance, feed_dict={x: np.random.random(point_num), y: np.random.random(point_num)})
    sum = 0;
    # count the point in the quadrant
    for i in range(point_num):
        if result[i] < 1:
            sum += 1;
    print "pi = %f" % (float(sum) / point_num  * 4)
