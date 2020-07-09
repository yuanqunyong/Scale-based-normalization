import tensorflow as tf
v = tf.Variable(1)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(10):
        v.load(i, sess)
        print(v.eval())
train_writer = tf.summary.FileWriter('/tmp/tmp')
train_writer.add_graph(tf.get_default_graph())