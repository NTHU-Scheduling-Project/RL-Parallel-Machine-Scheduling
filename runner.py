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
    def __init__(self, args, run_config: dict, env, N_F, training_data, validation_data):
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

        self.job_family_dataset, self.processing_time_dataset = np.split(training_data, [2], axis=2)
        self.job_family_dataset = self.job_family_dataset.astype(int)
        self.processing_time_dataset = self.processing_time_dataset.astype(int)
        self.validation_dataset = validation_data

    def train(self, agent):
        highest_utilization = 0.0
        for i in range(1, self.run_config["train"]["episodes"] + 1):
            instance_index = random.randint(0, self.processing_time_dataset.shape[0] - 1)
            processing_time = self.processing_time_dataset[instance_index]
            job_family = self.job_family_dataset[instance_index]
            phase = "train"
            state = self.env.reset(processing_time, job_family)
            #state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            legal_actions = np.arange(self.N_F * self.N_F)
            R = 0  # return (sum of rewards)
            t = 0  # time step
            while True:
                action = agent.select_action(torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0), legal_actions)
                obs, reward, done, legal_actions, info = self.env.step(action)
                #next_state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                next_state = obs
                R += reward
                agent.add_transition(state, action, reward, next_state, done, legal_actions)
                state = next_state
                agent.learn()

                #print(legal_actions)
                t += 1
                if done:
                    #print(obs.shape)
                    break
            
            statistics = agent.get_statistics()
            self.add_scalar(phase + "/episode_reward", R, i)
            #self.add_scalar(phase + "/makespan", self.env.makespan, i)
            self.add_scalar(phase + "/average_q", statistics[0][1], i)
            self.add_scalar(phase + "/average_loss", statistics[1][1], i)
            self.add_scalar(phase + "/epsilon", agent.epsilon_threshold, i)

            if i % 10 == 0:
                logging.info("episode: {}, episode reward: {}, info: {}.".format(i, R, info))
                # save model
                agent.save(self.model_dir)
            if i > 0 and i % self.val_frequency == 0:
                utilization = self.validate(agent, start_i=i // self.val_frequency)

                # save model with best val performance
                if utilization > highest_utilization:
                    highest_utilization = utilization
                    agent.save(os.path.join(self.model_dir, "best"))
                #agent.save(os.path.join(self.model_dir, "epi_{}".format(i)))

    def validate(self, agent, start_i=1, phase="val"):
        if phase == "test":
            #agent.load(self.model_dir)
            agent.load(os.path.join(self.model_dir, "best"))

        n_episodes = len(self.validation_dataset)
        count = 0
        total_utilization = 0
        #with agent.eval_mode():
        pbar = tqdm(range((start_i - 1) * n_episodes + 1, start_i * n_episodes + 1))
        for i in pbar:
            processing_time = self.validation_dataset[count][0]
            job_family = self.validation_dataset[count][1]
            state = self.env.reset(processing_time, job_family)
            #state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            count += 1
            legal_actions = np.arange(self.N_F * self.N_F)
            R = 0  # return (sum of rewards)
            t = 0  # time step

            while True:
                action = agent.select_action(torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0), legal_actions, "val")
                obs, reward, done, legal_actions, info = self.env.step(action)
                #next_state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                next_state = obs
                #print(legal_actions)
                R += reward
                state = next_state
                t += 1
                if done:
                    #if phase == "test":
                        #self.env.draw_gantt(count)
                    break

            #print("\n")
            pbar.set_description(f"Instance {count}: makespan={self.env.makespan}, utilization={R}")
            
            '''statistics = agent.get_statistics()
            self.add_scalar(phase + "/episode_reward", R, i)
            #self.add_scalar(phase + "/makespan", self.env.makespan, i)
            self.add_scalar(phase + "/average_q", statistics[0][1], i)
            self.add_scalar(phase + "/average_loss", statistics[1][1], i)'''
            total_utilization += R

        statistics = agent.get_statistics()
        self.add_scalar(phase + "/episode_reward", total_utilization / n_episodes, start_i - 1)
        #self.add_scalar(phase + "/makespan", self.env.makespan, i)
        self.add_scalar(phase + "/average_q", statistics[0][1], start_i - 1)
        self.add_scalar(phase + "/average_loss", statistics[1][1], start_i - 1)
        
        return total_utilization / n_episodes
    
    def add_scalar(self, tag: str, scalar_value, global_step: int):
        if self.writer is not None:
            self.writer.add_scalar(tag, scalar_value, global_step)