import gym
import torch
import time
import os
import ray
import numpy as np

from tqdm import tqdm
from random import uniform, randint

import io
import base64
# from IPython.display import HTML

from dqn_model import DQNModel
from dqn_model import _DQNModel
from memory import ReplayBuffer

import matplotlib.pyplot as plt
# %matplotlib inline

from memory_remote import ReplayBuffer_remote
from dqn_model import _DQNModel
import torch
from custom_cartpole import CartPoleEnv

FloatTensor = torch.FloatTensor


# Set the Env name and action space for CartPole
ENV_NAME = 'CartPole_distributed'
# Move left, Move right
ACTION_DICT = {
    "LEFT": 0,
    "RIGHT":1
}
# Register the environment
env_CartPole = CartPoleEnv()

# Set result saveing floder
result_floder = ENV_NAME
result_file = ENV_NAME + "/results.txt"
if not os.path.isdir(result_floder):
    os.mkdir(result_floder)
torch.set_num_threads(12)

def plot_result(total_rewards ,learning_num, legend):
    print("\nLearning Performance:\n")
    episodes = []
    for i in range(len(total_rewards)):
        episodes.append(i * learning_num + 1)
        
    plt.figure(num = 1)
    fig, ax = plt.subplots()
    plt.plot(episodes, total_rewards)
    plt.title('performance')
    plt.legend(legend)
    plt.xlabel("Episodes")
    plt.ylabel("total rewards")
    plt.savefig("distributed_DQN.png")

hyperparams_CartPole = {
    'epsilon_decay_steps' : 100000, 
    'final_epsilon' : 0.1,
    'batch_size' : 32, 
    'update_steps' : 10, 
    'memory_size' : 2000, 
    'beta' : 0.99, 
    'model_replace_freq' : 2000,
    'learning_rate' : 0.0003,
    'use_target_model': True
}

ray.shutdown()
ray.init(include_webui=False, ignore_reinit_error=True, redis_max_memory=100000000, object_store_memory=1000000000)

@ray.remote
class Model_server():
    def __init__(self, memory_server, env, hyper_params, ew_id, ew_num, action_space):
        self.memory_server = memory_server
        self.env = env
        self.max_episode_steps = env._max_episode_steps
        self.batch_size = hyper_params['batch_size']
        self.beta = hyper_params['beta']
        state = env.reset()
        input_len = len(state)
        output_len = action_space
        self.eval_model = DQNModel(input_len, output_len, learning_rate = hyper_params['learning_rate'])
        self.target_model = DQNModel(input_len, output_len)
        self.ew_id = ew_id
        self.ew_num = ew_num
        self.all_results = []

        
        
    def update(self):
        batch = ray.get(self.memory_server.sample.remote(self.batch_size))
        (states, actions, reward, next_states, is_terminal) = batch
        states = states
        next_states = next_states
        terminal = FloatTensor([1 if t else 0 for t in is_terminal])
        reward = FloatTensor(reward)
        batch_index = torch.arange(self.batch_size, dtype=torch.long)

        # Current Q Values
        _, q_values = self.eval_model.predict_batch(states)
        q_values = q_values[batch_index, actions]

        # use target model
        actions, q_next = self.target_model.predict_batch(next_states)

        q_target = reward + self.beta * torch.max(q_next, 1)[0] * (1 - terminal)

        # update model
        self.eval_model.fit(q_values, q_target)
    
    def predict(self, state):
        return self.eval_model.predict(state)
    
    def model_replace(self):
        self.target_model.replace(self.eval_model)

    def call_evaluate(self, trials = 30):
        workers_id = []
        result_list = []
        for i in range(self.ew_num):
            workers_id.append(self.ew_id[i].evaluate.remote(self.eval_model, trials // self.ew_num))
        ray.wait(workers_id, len(workers_id))
        
        result_list = ray.get(workers_id)
        result = 0
        for i in range(len(result_list)):
            result += result_list[i]
        result = result / self.ew_num
        print(result)
        self.all_results.append(result)
        
    def get_results(self):
        return self.all_results

        
@ray.remote
class DQN_collector(object):
    def __init__(self, env, hyper_params, action_space = len(ACTION_DICT)):
        self.env = env
        self.max_episode_steps = env._max_episode_steps
        self.initial_epsilon = 1
        self.final_epsilon = hyper_params['final_epsilon']
        self.epsilon_decay_steps = hyper_params['epsilon_decay_steps']
        self.episode = 0
        self.steps = 0
        self.action_space = action_space
        self.batch_size = hyper_params['batch_size']
        self.update_steps = hyper_params['update_steps']
        self.model_replace_freq = hyper_params['model_replace_freq']
        
    def linear_decrease(self, initial_value, final_value, final_decay_steps):
        decay_rate = curr_steps / final_decay_steps
        if decay_rate > 1:
            decay_rate = 1
        return initial_value - (initial_value - final_value) * decay_rate

    def explore_or_exploit_policy(self, state):
        p = uniform(0,1)
        epsilon = self.linear_decrease(self.initial_epsilon, self.final_epsilon, self.epsilon_decay_steps)
        if p < epsilon:
            # return action
            return randint(0, self.action_space - 1)
        else:
            # return action
            return self.greedy_policy(state)
                
    def greedy_policy(self, state):
        return ray.get(self.model_server.predict.remote(state))

    def learn(self, model_server, memory_server, training_episodes, test_interval = 50):
        test_num = training_episodes // test_interval

        for i in tqdm(range(test_num), desc="Training"):   
            for episode in range(test_interval):
                state = self.env.reset()
                done = False
                steps = 0

                while steps < self.max_episode_steps and not done:
                    action = self.explore_or_exploit_policy(state)
                    next_state, reward, done, _ = self.env.step(action)
                    memory_server.add.remote(state, action, reward, next_state, done)
                    
                    if self.steps % self.update_steps == 0:
                        model_server.update.remote()
                    if self.steps % self.model_replace_freq == 0:
                        model_server.model_replace.remote()
                    
                    steps += 1
                    self.steps += 1
                    state = next_state
            # call evaluate
            model_server.call_evaluate.remote()


@ray.remote
class DQN_evaluator(object):
    def __init__(self, env):
        # self.eval_model = eval_model
        self.env = env
        self.best_reward = 0
        self.max_episode_steps = env._max_episode_steps
        
    def evaluate(self, eval_model, trials = 30):
        total_reward = 0
        for trial in range(trials):
            state = self.env.reset()
            done = False
            steps = 0

            while steps < self.max_episode_steps and not done:
                steps += 1
                action = eval_model.predict(state)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
        avg_reward = total_reward / trials
        # print(avg_reward)
        # f = open(result_file, "a+")
        # f.write(str(avg_reward) + "\n")
        # f.close()
        # if avg_reward >= self.best_reward:
        #     self.best_reward = avg_reward
        #     self.save_model(eval_model)
        return avg_reward

    # save model
    def save_model(self, eval_model):
        eval_model.save(result_floder + '/best_model.pt')



class Distributed_DQN():
    def __init__(self, env, hyper_params, cw_num, ew_num, action_space = len(ACTION_DICT)):
        self.env = env
        self.hyper_params = hyper_params
        self.action_space = action_space
        self.cw_num = cw_num
        self.ew_num = ew_num
        self.test_interval = 100
        self.trials = 30
        self.memory_server = ReplayBuffer_remote.remote(hyper_params['memory_size'])

        
    def learn_and_evaluate(self, training_episodes, test_interval):

        cw_id = []
        ew_id = []
        for _ in range(self.cw_num):
            cw_id.append(DQN_collector.remote(self.env, self.hyper_params, self.action_space))
        for _ in range(self.ew_num):
            ew_id.append(DQN_evaluator.remote(self.env))        

        model_server = Model_server.remote(self.memory_server, self.env, self.hyper_params, 
                                            ew_id, self.ew_num, self.action_space)        
        
        test_num = training_episodes // test_interval
        all_results = []
        workers_id = []
        results = []
        
        for i in range(self.cw_num):
            workers_id.append(cw_id[i].learn.remote(model_server, self.memory_server, 
                                                    training_episodes // self.cw_num))
        ray.wait(workers_id, len(workers_id))
        # get result from model_server
        all_results = ray.get(model_server.get_results.remote())
        return all_results
        
env_CartPole.reset()
cw_num = 10
ew_num = 5
training_episodes = 10000
test_interval = 50
curr_steps = 0
Distri_DQN = Distributed_DQN(env_CartPole, hyperparams_CartPole, cw_num, ew_num)
time0 = time.time()
result = Distri_DQN.learn_and_evaluate(training_episodes, test_interval)
time1 = time.time()
# print(result)
print(time1 - time0)
plot_result(result, test_interval, ["Distributed_full_DQN"])