import pandas as pd
import numpy as np
import random
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

from operator import add
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dropout,Dense

class entropy(object):
    def __init__(self):
        self.recent_memory=np.array([])
        self.data=pd.DataFrame()
        self.memory=[]
        self.actual=[]
        self.epsilon=0
        self.reward=0
        self.learning_rate=0.0005
        self.gamma=0.9
        self.qtarget=1
        self.qpredict=0
        self.model = self.nn()

    def nn(self,saved_weights=None):
        model=Sequential()
        model.add(Dense(120, activation='relu', input_dim=11))
        model.add(Dropout(0.15))
        model.add(Dense(120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(3, activation='softmax'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)
        if saved_weights:
            model.load_weights(saved_weights)
        return model

    def current_state(self,env,snake,apple):
        state = [
            (snake.del_x == 20 and snake.del_y == 0 and (
                        (list(map(add,snake.pos[-1], [20, 0])) in snake.pos) or
                        snake.pos[-1][0] + 20 >= (env.width - 20))) or (
                        snake.del_x == -20 and snake.del_y == 0 and (
                            (list(map(add, snake.pos[-1], [-20, 0])) in snake.pos) or
                            snake.pos[-1][0] - 20 < 20)) or (snake.del_x == 0 and snake.del_y == -20 and (
                        (list(map(add, snake.pos[-1], [0, -20])) in snake.pos) or
                        snake.pos[-1][-1] - 20 < 20)) or (snake.del_x == 0 and snake.del_y == 20 and (
                        (list(map(add, snake.pos[-1], [0, 20])) in snake.pos) or
                        snake.pos[-1][-1] + 20 >= (env.height - 20))),

            (snake.del_x == 0 and snake.del_y == -20 and (
                        (list(map(add, snake.pos[-1], [20, 0])) in snake.pos) or
                        snake.pos[-1][0] + 20 > (env.width - 20))) or (
                        snake.del_x == 0 and snake.del_y == 20 and ((list(map(add, snake.pos[-1],
                                                                                      [-20, 0])) in snake.pos) or
                                                                            snake.pos[-1][0] - 20 < 20)) or (
                        snake.del_x == -20 and snake.del_y == 0 and ((list(map(
                    add, snake.pos[-1], [0, -20])) in snake.pos) or snake.pos[-1][-1] - 20 < 20)) or (
                        snake.del_x == 20 and snake.del_y == 0 and (
                        (list(map(add, snake.pos[-1], [0, 20])) in snake.pos) or snake.pos[-1][
                    -1] + 20 >= (env.height - 20))),

            (snake.del_x == 0 and snake.del_y == 20 and (
                        (list(map(add, snake.pos[-1], [20, 0])) in snake.pos) or
                        snake.pos[-1][0] + 20 > (env.width - 20))) or (
                        snake.del_x == 0 and snake.del_y == -20 and ((list(map(
                    add, snake.pos[-1], [-20, 0])) in snake.pos) or snake.pos[-1][0] - 20 < 20)) or (
                        snake.del_x == 20 and snake.del_y == 0 and (
                        (list(map(add, snake.pos[-1], [0, -20])) in snake.pos) or snake.pos[-1][
                    -1] - 20 < 20)) or (
                    snake.del_x == -20 and snake.del_y == 0 and (
                        (list(map(add, snake.pos[-1], [0, 20])) in snake.pos) or
                        snake.pos[-1][-1] + 20 >= (env.height - 20))),

            snake.del_x==-20,
            snake.del_x==20,
            snake.del_y==-20,
            snake.del_y==20,
            apple.app_x < snake.x,
            apple.app_x > snake.x,
            apple.app_y < snake.y,
            apple.app_y > snake.y
        ]

        for i in range(len(state)):
            if state[i]:
                state[i]=1
            else:
                state[i]=0
        return np.asarray(state)

    def reward_rules(self,snake,dead):
        self.reward=0
        if dead:
            self.reward=-10
            return self.reward
        if snake.consumed:
            self.reward=10
        return self.reward

    def new_memory_replay(self,memory):
        if len(memory)>1000:
            minibatch=random.sample(memory,1000)
        else:
            minibatch=memory

        for state,action,reward,next_state,done in minibatch:
            target=reward
            if not done:
                target=reward+self.gamma*np.amax(self.model.predict(np.array([next_state]))[0])
            target_f=self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)]=target
            self.model.fit(np.array([state]),target_f,epochs=1,verbose=0)

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))

    def short_memory_training(self,state,action,reward,next_state,done):
        target=reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, 11)))[0])
        target_f = self.model.predict(state.reshape((1, 11)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, 11)), target_f, epochs=1, verbose=0)
