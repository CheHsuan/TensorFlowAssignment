import gym
import numpy as np
import tensorflow as tf
import sys

input_dim = 4
batch_size = 1
output_dim = 2
num_hidden_neurons = 10
reference = 0

# define the game environment - carpole
env = gym.make('CartPole-v0')
# define the policy network - a 3-layer nerual network
graph = tf.Graph()
with graph.as_default():
    # define input and output dimension
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, input_dim))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, output_dim))
    tf_test_dataset = tf.placeholder(tf.float32, shape=(batch_size, input_dim))
  
    # Variables.
    weights1 = tf.Variable(tf.truncated_normal([input_dim, num_hidden_neurons]))
    biases1 = tf.Variable(tf.zeros([num_hidden_neurons]))
    weights2 = tf.Variable(tf.truncated_normal([num_hidden_neurons, output_dim]))
    biases2 = tf.Variable(tf.zeros([output_dim]))
  
    # Training computation.
    tf_train_h_dataset = tf.matmul(tf_train_dataset, weights1) + biases1
    tf_train_h_dataset = tf.nn.relu(tf_train_h_dataset)
    logits = tf.matmul(tf_train_h_dataset, weights2) + biases2 
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
 
    # action
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1), weights2) + biases2)

def prediction(observation):
    output = test_prediction.eval(feed_dict={tf_test_dataset : [observation]})
    result = np.zeros(2)
    #print output
    result[np.argmax(output[0])] = 1
    #print result
    return output[0]

def training(obervations, actions, reward):
    discount_factor = 0.8
    for i in range(len(observations)):
        actions[i] = np.multiply(actions[i], reward)
        actions[i] = np.multiply(actions[i], discount_factor)
        
        #print test_prediction.eval(feed_dict={tf_test_dataset : [observations[i]]})    

        feed_dict = {tf_train_dataset : [observations[i]], tf_train_labels : [actions[i]]}
        session.run(optimizer, feed_dict=feed_dict)

        #print test_prediction.eval(feed_dict={tf_test_dataset : [observations[i]]})
           
        discount_factor = 0.95 * discount_factor

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    #set episode time to 10
    for i_episode in range(int(sys.argv[1])):
        # reset the env
        observation = env.reset()
        observations = list()
        actions = list()
        for t in range(1000):
            env.render()
            action = prediction(observation)
            observation, reward, done, info = env.step(np.argmax(action))
            observations.append(observation)
            actions.append(action)
            if done:
                if t >= reference:
                    training(observations, actions, 1 / ((t - reference) + 1))
                    reference = t
                else:
                    training(observations, actions, -1 / (abs(t-reference) + 1))
                break
        print("Episode finished after {} timesteps".format(t+1))
        del observations[:]
        del actions[:]
