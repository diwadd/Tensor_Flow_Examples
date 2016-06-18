import csv
import tensorflow as tf





sess = tf.Session()

a = tf.constant([1, 1, 1])
b = tf.constant([2, 3, 4])
c = tf.Variable(tf.zeros([0, 0, 0]))
c = tf.add(a,b)

tf.initialize_all_variables()


result = sess.run(c)

print(type(result))
print(result)