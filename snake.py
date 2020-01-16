import numpy as np
import pygame
import random
from ai import entropy
from keras.utils import to_categorical
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("--speed", help="Set the speed of Serpinco",type=int)
parser.add_argument("--render", type=str2bool, nargs='?',const=True,help="Render or not")
args = parser.parse_args()

render = args.render
speed=100-args.speed
black=(0,0,0)

class Snake_env():
    def __init__(self,width,height):
        self.width=width
        self.height=height
        self.dead=False
        pygame.display.set_caption('7enTropy7')
        self.gameDisplay=pygame.display.set_mode((width,height))
        self.snake=Snake(self)
        self.apple=Apple()
        self.game_score=0


class Snake(object):
    def __init__(self, env):
        self.pos=[]
        x=env.width*0.45
        self.x=x-x%20
        y=env.height*0.5
        self.y=y-y%20
        self.pos.append([self.x,self.y])
        self.apple=1
        self.consumed=False
        self.body_image=pygame.image.load('elements/square.png')
        self.del_x=20
        self.del_y=0

    def slither(self,choice,x,y,env,apple,agent):
        temp=[self.del_x,self.del_y]
        if self.consumed:
            self.pos.append([self.x,self.y])
            self.consumed=False
            self.apple=self.apple+1

        if np.array_equal(choice,[1,0,0]):
            temp=self.del_x,self.del_y
        elif np.array_equal(choice,[0,1,0]) and self.del_y==0:
            temp=[0,self.del_x]
        elif np.array_equal(choice, [0, 1, 0]) and self.del_x==0:
            temp=[-self.del_y,0]
        elif np.array_equal(choice, [0, 0, 1]) and self.del_y==0:
            temp=[0,-self.del_x]
        elif np.array_equal(choice, [0, 0, 1]) and self.del_x==0:
            temp = [self.del_y,0]

        self.del_x,self.del_y=temp
        self.x=x+self.del_x
        self.y=y+self.del_y

        if self.x<20 or self.x>env.width-40 or self.y<20 or self.y>env.height-40 or [self.x,self.y] in self.pos:
            env.dead=True
        eat_apple(self,apple,env)
        self.refresh_pos(self.x,self.y)

    def refresh_pos(self,x,y):
        if self.pos[-1][0]!=x or self.pos[-1][1]!=y:
            if self.apple>1:
                for i in range(0,self.apple-1):
                    self.pos[i][0],self.pos[i][1]=self.pos[i+1]
            self.pos[-1][0]=x
            self.pos[-1][1]=y

    def show_snake(self,x,y,apple,env):
        self.pos[-1][0]=x
        self.pos[-1][1]=y
        if env.dead!=True:
            for i in range(apple):
                tx,ty = self.pos[len(self.pos)-i-1]
                env.gameDisplay.blit(self.body_image,(tx,ty))
            pygame.display.update()
        else:
            pygame.time.wait(200)


class Apple(object):
    def __init__(self):
        self.app_x=240
        self.app_y=200
        self.apple_image=pygame.image.load('elements/apple.png')

    def show_apple(self,x,y,env):
        env.gameDisplay.blit(self.apple_image,(x,y))
        pygame.display.update()

    def apple_pos(self,env,snake):
        rx=random.randint(20,env.width-40)
        self.app_x=rx-rx%20
        ry = random.randint(20, env.height - 40)
        self.app_y=ry-ry%20
        if [self.app_x,self.app_y] not in snake.pos:
            return self.app_x,self.app_y
        else:
            self.apple_pos(env,snake)


def eat_apple(snake,apple,env):
    if snake.x==apple.app_x and snake.y==apple.app_y:
        apple.apple_pos(env,snake)
        snake.consumed=True
        env.game_score=env.game_score+1

def highscore(game_score,record_score):
    if game_score>=record_score:
        return game_score
    else:
        return record_score

def display_screen(snake,apple,env):
    env.gameDisplay.fill(black)
    snake.show_snake(snake.pos[-1][0],snake.pos[-1][1],snake.apple,env)
    apple.show_apple(apple.app_x,apple.app_y,env)

def start_game(snake,env,apple,agent):
    start_state1=agent.current_state(env,snake,apple)
    action=[1,0,0]
    snake.slither(action,snake.x,snake.y,env,apple,agent)
    start_state2=agent.current_state(env,snake,apple)
    rew=agent.reward_rules(snake,env.dead)
    agent.remember(start_state1,action,rew,start_state2,env.dead)
    agent.new_memory_replay(agent.memory)


def train_snake():
    pygame.init()
    agent=entropy()
    epoch=0
    highest_record=0
    while epoch<1000:
        env=Snake_env(440,440)
        serpico=env.snake
        food=env.apple

        start_game(serpico,env,food,agent)
        if render:
            display_screen(serpico,food,env)

        while not env.dead:
            agent.epsilon=80-epoch
            old_state=agent.current_state(env,serpico,food)
            if random.randint(0,200)<agent.epsilon:
                final_move=to_categorical(random.randint(0,2),num_classes=3)
            else:
                prediction = agent.model.predict(old_state.reshape((1, 11)))
                final_move = to_categorical(np.argmax(prediction[0]), num_classes=3)

            serpico.slither(final_move,serpico.x,serpico.y,env,food,agent)
            new_state=agent.current_state(env,serpico,food)

            reward=agent.reward_rules(serpico,env.dead)
            agent.short_memory_training(old_state,final_move,reward,new_state,env.dead)
            agent.remember(old_state,final_move,reward,new_state,env.dead)
            highest_record=highscore(env.game_score,highest_record)

            if render:
                display_screen(serpico,food,env)
                pygame.time.wait(speed)

        agent.new_memory_replay(agent.memory)
        epoch+=1
        print('Game : ',epoch,' ------------ Score : ',env.game_score, ' ------------ Highscore : ',highest_record)
    agent.model.save_weights('serpico.h5')

train_snake()
