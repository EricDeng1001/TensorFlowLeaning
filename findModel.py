import tensorflow as tf

x_train = input("input x\n")
y_train = input("input y\n")

W = tf.Variable([0.3], dtype = tf.float32)
b = tf.Variable([-0.3], dtype = tf.float32)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
linear_model = W * x + b

# loss
loss = tf.reduce_sum(tf.square(linear_model - y))
# optimizer
optimizer = tf.train.GradientDescentOptimizer(1e-2)
train = optimizer.minimize(loss)

# training data

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(1000):
  sess.run(train, {x: x_train, y: y_train})
  print "training : %d" % i

curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
Accuracy = 1 - curr_loss / 1
print("the model is y=%fx+%f,Accuracy: %f" % (curr_W, curr_b, Accuracy))
