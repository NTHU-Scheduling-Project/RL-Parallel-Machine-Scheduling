import logging
import os
import random
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from tensorboardX import SummaryWriter

from env.env import PMSPEnv

class Runner:
    def __init__(self, args, run_config: dict, env, N_F):
        self.args = args
        self.run_config = run_config
        self.N_F = N_F
        
        self.val_frequency = run_config["val"]["frequency"]
        self.env = env
        self.writer = SummaryWriter(logdir=os.path.join(args.output))
        self.device = (
            torch.device("cuda:{}".format(args.gpu))
            if torch.cuda.is_available() and args.gpu >= 0
            else torch.device("cpu")
        )
        self.model_dir = os.path.join(args.output, args.model_dir)
        if args.epi is not None:
            self.model_dir = os.path.join(self.model_dir, "epi_{}".format(args.epi))

    def train(self, agent):
        shortest_makespan = np.inf
        for i in range(1, self.run_config["train"]["episodes"] + 1):
            phase = "train"
            obs = self.env.reset()
            state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            legal_actions = np.arange(self.N_F * self.N_F)
            R = 0  # return (sum of rewards)
            t = 0  # time step
            while True:
                action = agent.select_action(state, legal_actions)
                obs, reward, done, legal_actions, info = self.env.step(action)
                next_state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                R += reward
                agent.add_transition(state.squeeze(0).numpy(), action, reward, next_state.squeeze(0).numpy(), done, legal_actions)
                state = next_state
                agent.learn()

                #print(legal_actions)
                t += 1
                if done:
                    #print(obs.shape)
                    break
            
            statistics = agent.get_statistics()
            self.add_scalar(phase + "/episode_reward", R, i)
            self.add_scalar(phase + "/makespan", self.env.makespan, i)
            self.add_scalar(phase + "/average_q", statistics[0][1], i)
            self.add_scalar(phase + "/average_loss", statistics[1][1], i)
            self.add_scalar(phase + "/epsilon", agent.epsilon_threshold, i)

            if i % 10 == 0:
                logging.info("episode: {}, makespan: {}, episode reward: {}, info: {}.".format(i, self.env.makespan, R, info))
                # save model
                agent.save(self.model_dir)
            if i > 0 and i % self.val_frequency == 0:
                makespan = self.validate(agent, start_i=i // self.val_frequency)

                # save model with best val performance
                if makespan < shortest_makespan:
                    shortest_makespan = makespan
                    agent.save(os.path.join(self.model_dir, "best"))
                #agent.save(os.path.join(self.model_dir, "epi_{}".format(i)))

    def validate(self, agent, start_i=1, phase="val"):
        if phase == "test":
            #agent.load(self.model_dir)
            agent.load(os.path.join(self.model_dir, "best"))

        n_episodes = 1
        count = 0
        total_makespan = 0
        #with agent.eval_mode():
        pbar = tqdm(range((start_i - 1) * n_episodes + 1, start_i * n_episodes + 1))
        for i in pbar:
            obs = self.env.reset()
            state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            count += 1
            legal_actions = np.arange(self.N_F * self.N_F)
            R = 0  # return (sum of rewards)
            t = 0  # time step
            while True:
                action = agent.select_action(state, legal_actions, "val")
                obs, reward, done, legal_actions, info = self.env.step(action)
                next_state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                #print(legal_actions)
                R += reward
                state = next_state
                t += 1
                if done:
                    if phase == "test":
                        self.env.draw_gantt(count)
                    break

            print("\n")
            pbar.set_description(f"Instance {count}: makespan={self.env.makespan}, utilization={R}")
            
            statistics = agent.get_statistics()
            self.add_scalar(phase + "/episode_reward", R, i)
            self.add_scalar(phase + "/makespan", self.env.makespan, i)
            self.add_scalar(phase + "/average_q", statistics[0][1], i)
            self.add_scalar(phase + "/average_loss", statistics[1][1], i)
            total_makespan += self.env.makespan
        return total_makespan / n_episodes
    
    def add_scalar(self, tag: str, scalar_value, global_step: int):
        if self.writer is not None:
            self.writer.add_scalar(tag, scalar_value, global_step)