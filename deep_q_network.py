#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque
from create_network import createNetwork
import copy
import os
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations

# OBSERVE = 100000. # timesteps to observe before training
# EXPLORE = 2000000. # frames over which to anneal epsilon
# INITIAL_EPSILON = 0.0001 # starting value of epsilon
OBSERVE = 2000. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon

FINAL_EPSILON = 0.0001 # final value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1

TARGET_NETWORK_UPDATE_INTERVAL = 100

def trainNetwork(s, q, st, q_t, dup_main_to_target, sess):
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(q, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    if not os.path.exists("DDQN_saved_networks"):
        os.makedirs("DDQN_saved_networks")
    # checkpoint = tf.train.get_checkpoint_state("DDQN_saved_networks")
    # if checkpoint and checkpoint.model_checkpoint_path:
    #     saver.restore(sess, checkpoint.model_checkpoint_path)
    #     print("Successfully loaded:", checkpoint.model_checkpoint_path)
    # else:
    #     print("Could not find old network weights")

    #================================================================================================================#
    #-------------------------------------------Train starts here----------------------------------------------------#
    #================================================================================================================#
    t = 0
    epsilon = INITIAL_EPSILON
    reward_saver = []
    sess.run(dup_main_to_target)
    while "flappy bird" != "angry bird":
        # choose an action epsilon greedily
        readout_t = q.eval(feed_dict={s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1 # do nothing

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # store the transition in D
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = q_t.eval(feed_dict = {st : s_j1_batch})
            for i in range(0, len(minibatch)):
                Terminal = minibatch[i][4]
                # if terminal, only equals reward
                if Terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient step
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch}
            )

            # evaluate network every 1000 timesteps
            if t%1000==0:
                average_reward = evaluateNetwork(s, q)
                reward_saver.append(average_reward)

        # update the old values
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations and plot the average reward
        if t % 10000 == 0:
            saver.save(sess, 'DDQN_saved_networks/' + GAME + '-dqn', global_step = t)
            fig, ax = plt.subplots(1)
            ax.plot(range(len(reward_saver)), reward_saver)
            ax.set_xlabel('time steps/1000')
            ax.set_ylabel('average_reward')
            fig.savefig('images/reward_images/average_reward_'+str(t)+'.png')

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        if t % TARGET_NETWORK_UPDATE_INTERVAL == 0:
            sess.run(dup_main_to_target)
            print("TIMESTEP", t, "/ STATE", state, \
                "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
                "/ Q_MAX %e" % np.max(readout_t))

    #================================================================================================================#
    #---------------------------------------------Train ends here----------------------------------------------------#
    #================================================================================================================#

def evaluateNetwork(s, readout):
    print('start evaluation')
    n_trails = 10
    max_timestep = 2000
    # run n trials and calculate the average score
    average_reward = []
    for i in range(n_trails):
        game_state = game.GameState()
        # get the first state by doing nothing and preprocess the image to 80x80x4
        do_nothing = np.zeros(ACTIONS)
        do_nothing[0] = 1
        x_t, r_0, terminal = game_state.frame_step(do_nothing)
        x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

        t = 0
        reward_list = []
        while (not terminal) and (t<max_timestep):
            # choose an action epsilon greedily
            readout_t = readout.eval(feed_dict={s : [s_t]})[0]
            a_t = np.zeros([ACTIONS])
            if t % FRAME_PER_ACTION == 0:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
            else:
                a_t[0] = 1 # do nothing
            x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
            reward_list.append(r_t)
            x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
            ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
            x_t1 = np.reshape(x_t1, (80, 80, 1))
            # s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
            s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

            s_t = s_t1
            t += 1
        average_reward.append(sum(reward_list)/len(reward_list))
    print('stop evaluation and the average reward is '+ str(sum(average_reward)/len(average_reward)))
    return sum(average_reward)/len(average_reward)

def playGame():
    sess = tf.InteractiveSession()

    # Main deep neural network
    input_state, q = createNetwork()
    q_nn_para = tf.trainable_variables()    # save the parameter
    #q_nn = q

    # Target deep neural network with delay trick
    input_state_t, qt = createNetwork()
    q_nn_t_para = tf.trainable_variables()[len(q_nn_para) :]    # save the parameter
    #q_nn_t = qt

    # Copy over the main nn to target nn
    dup_main_to_target = [q_nn_t_para[i].assign(q_nn_para[i]) for i in range(len(q_nn_para))]

    trainNetwork(input_state, q, input_state_t, qt,  dup_main_to_target, sess)

if __name__ == "__main__":
    playGame()
