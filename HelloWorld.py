#import tensorflow as tf
#
#node1 = tf.constant( 3.0 , dtype = tf.float32 )
#node2 = tf.constant( 4.0 ) # also tf.float32 implicitly
## a constant node can store const value and is initlized on create
#node3 = tf.add( node1 , node2 )
## a add node that recieve node1 and node2
#
#sess = tf.Session()
##a session can run a network
#
#a = tf.placeholder(tf.float32)
#b = tf.placeholder(tf.float32)
##a placeholder is a promised input that not be provided yet , but must provide
## when running the network
#adder_node = a + b  # + provides a shortcut for tf.add(a, b)
## this add node recieve two inputs. inputs are vars itself
#add_and_triple = adder_node * 3.0
##this mult node recieve the adder_node's internal value and stroe it by 3.0
#
#
#W = tf.Variable([0.3], dtype=tf.float32)
#b = tf.Variable([-0.3], dtype=tf.float32)
## Variable is not esstentially inputs , it can be mid-var
## Variable is initlized when init are excuted
#x = tf.placeholder(tf.float32)
## another must-give
#linear_model = W * x + b
## our network model
#
#init = tf.global_variables_initializer()
##the Variable initlizer
#sess.run(init)
##network run init
#
#fixW = tf.assign(W,[-1.0])
#fixb = tf.assign(b,[1.0])
#sess.run([fixW,fixb])
#
#y = tf.placeholder(tf.float32)
#squared_deltas = tf.square(linear_model - y)
##the squared_deltas give one squared delta a data-runing time
#loss = tf.reduce_sum(squared_deltas)
#
#optimizer = tf.train.GradientDescentOptimizer(0.01)
#train = optimizer.minimize(loss)
#
#sess.run(init) # reset values to incorrect defaults.
#for i in range(1000):
#  sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
#
#print(sess.run([W, b]))
#
##print(sess.run(loss, {x: [1, 2, 3, 4 ], y: [0, -1, -2, -3 ]}))

import tensorflow as tf

# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x: x_train, y: y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))
