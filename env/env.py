import os
import copy
import numpy as np
import numpy.ma as ma
import pandas as pd
import gym
import random
from gym.core import ActType, ObsType

class PMSPEnv(gym.Env):
    def __init__(self, args, job_size, machine_size, family_size, processing_time, job_family):
        self.args = args
        self.N_J = job_size
        self.N_M = machine_size
        self.N_F = family_size
        self.T = int(np.mean(processing_time[:, 0]) / 2)
        self.H_p = int(np.max(processing_time[:, 0]) / self.T)
        self.processing_time = processing_time
        self.job_family = job_family
        # state dependent
        self.remaining_processing_time = None
        self.family_setup_time = np.zeros(self.N_F)
        for i in range(self.job_family.shape[0]):
            self.family_setup_time[self.job_family[i, 0] - 1] = self.job_family[i, 1]

        self.setup_time = np.zeros((self.N_F, self.N_F))
        for i in range(self.N_F):
            for j in range(self.N_F):
                if i == j:
                    self.setup_time[i, j] = 0
                else:
                    self.setup_time[i, j] = self.family_setup_time[i]

        self.machine_setup_status = np.zeros(self.N_M, dtype=int)
        self.setup_time_state = np.zeros((family_size, family_size))
        self.in_progress_job = np.zeros(job_size)
        self.in_progress_job_state = np.zeros((self.N_F, self.H_p))
        self.utilization_state = np.zeros((self.N_F, 3))
        self.action_history_state = np.zeros((self.N_F, 2))
        #self.other_history_state = np.zeros(3)
        self.makespan = None
        self.step_count = None
        self.reset()

        feasible_actions = np.ones(self.N_F * self.N_F)
        while True:
            obs, reward, done, feasible_actions, info = self.step(np.random.choice(np.where(feasible_actions == 1)[0]))
            print(feasible_actions)
            if done:
                break

    def init_data(self, **kwargs):
        self.makespan = 0
        self.step_count = 0
        self.remaining_processing_time = copy.deepcopy(self.processing_time[:, 0])
        self.in_progress_job = np.zeros((self.N_J, 2), dtype=int)
        for i in range(self.N_J):
            self.in_progress_job[i, 0] = 0
            self.in_progress_job[i, 1] = self.job_family[i, 0] - 1
        self.machine_setup_status = np.zeros(self.N_M, dtype=int)
        self.waiting_job = np.zeros((self.N_J, 2), dtype=int)
        for i in range(self.N_J):
            self.waiting_job[i, 0] = 1
            self.waiting_job[i, 1] = self.job_family[i, 0] - 1
        self.waiting_job_per_family = np.zeros(self.N_F)
        for i in range(self.N_J):
            self.waiting_job_per_family[self.job_family[i, 0] - 1] += 1
        self.num_of_jobs_per_family = copy.deepcopy(self.waiting_job_per_family)

        self.accumulated_setup_time_per_family = np.zeros(self.N_F)
        self.machine_finishing_time = np.zeros(self.N_M)
        self.job_start_time = np.ones(self.N_J) * np.inf
        self.job_end_time = np.ones(self.N_J) * np.inf

        self.feasible_actions = np.zeros((self.N_F, self.N_F))
        

    def reset(self, **kwargs):
        self.init_data(**kwargs)
        self.in_progress_job_state = np.zeros((self.N_F, self.H_p))
        self.setup_time_state = copy.deepcopy(self.setup_time)
        self.setup_time_max = np.max(self.setup_time_state)
        self.utilization_state = np.zeros((self.N_F, 3))
        self.action_history_state = np.zeros((self.N_F, 2))
        #self.other_history_state = np.zeros(3)

        # 一開始每台機器隨機分配 job family 並 setup
        for i in range(self.N_M):
            family = random.randint(1, self.N_F)
            self.machine_finishing_time[i] = self.family_setup_time[family - 1]
            self.machine_setup_status[i] = (family - 1)

        obs = self.get_observations(self.in_progress_job_state, self.setup_time_state, self.utilization_state, self.action_history_state)
        return obs
    
    def step(self, action: ActType):
        print(f'Current time step: {(self.step_count) * self.T} ~ {(self.step_count + 1) * self.T}')
        # Transform flatten action index to a_k = (f_k, g_k). e.g. action = 7, N_F = 4 --> (f, g) = (1, 3)
        f_k = int(action / self.N_F)
        g_k = int(action % self.N_F)
        print(f'a_k = ({f_k}, {g_k})')
        M = [index for index in range(self.N_M) if self.machine_setup_status[index] == g_k]
        if len(M) != 0:
            M_star = random.choice(M) # Select machine M* randomly from M
        else:
            M_star = ''
        print(M)
        while np.max(self.waiting_job_per_family) > 0 and np.min(self.machine_finishing_time) < ((self.step_count + 1) * self.T):
            M_i = np.argmin(self.machine_finishing_time)
            g = int(self.machine_setup_status[M_i])

            if M_star == M_i:
                f = int(f_k)
            else:
                f = np.argmin(ma.masked_where(self.waiting_job_per_family <= 0, self.setup_time[:, g]))
            
            masked_list = ma.masked_where(self.waiting_job[:, 1] != f, self.processing_time[:, 0])
            masked_list = ma.masked_where(self.waiting_job[:, 0] == 0, masked_list)
            job = np.argmax(masked_list)

            if self.waiting_job_per_family[f] == 0: # if there's no feasible action
                break
            # Assign the job on machine M_i
            self.job_start_time[job] = self.machine_finishing_time[M_i] + self.setup_time[f, g]
            self.job_end_time[job] = self.job_start_time[job] + self.processing_time[job, 0]
            self.accumulated_setup_time_per_family[f] += self.setup_time[f, g]
            self.machine_setup_status[M_i] = f
            self.machine_finishing_time[M_i] = self.job_end_time[job]
            # Remove the job from the waiting list
            self.waiting_job_per_family[f] -= 1
            self.waiting_job[job, 0] = 0
        
        # for time step k+1
        for job in range(self.N_J):
            if self.job_start_time[job] <= ((self.step_count + 1) * self.T) < self.job_end_time[job]:
                self.in_progress_job[job, 0] = 1
                self.remaining_processing_time[job] = self.job_end_time[job] - (self.step_count + 1) * self.T
            elif self.job_end_time[job] <= ((self.step_count + 1) * self.T):
                self.in_progress_job[job, 0] = 0
                self.remaining_processing_time[job] = 0

        # Obtain transited state s_k+1
        # in-progress jobs state
        self.in_progress_job_state = np.zeros(self.in_progress_job_state.shape)
        for job in range(self.N_J):
            if self.in_progress_job[job, 0] == 1:
                index = int(self.remaining_processing_time[job] / self.T)
                if index >= (self.H_p - 1):
                    self.in_progress_job_state[self.in_progress_job[job, 1], self.H_p - 1] += 1
                else:
                    self.in_progress_job_state[self.in_progress_job[job, 1], index] += 1
        
        # setup time state
        idle_machine_per_family = np.zeros(self.N_F)
        for i in range(self.N_M):
            if self.machine_finishing_time[i] < ((self.step_count + 2) * self.T):
                idle_machine_per_family[self.machine_setup_status[i]] += 1
        self.feasible_actions = np.zeros((self.N_F, self.N_F))
        for f in range(self.N_F):
            if self.waiting_job_per_family[f] > 0:
                for g in range(self.N_F):
                    if idle_machine_per_family[g] > 0:
                        self.feasible_actions[f, g] = 1
        for f in range(self.N_F):
            for g in range(self.N_F):
                if self.feasible_actions[f, g] == 1:
                    self.setup_time_state[f, g] = self.setup_time[f, g]
                else:
                    self.setup_time_state[f, g] = self.setup_time_max
        

        # utilization state
        accumulated_processing_time = self.processing_time[:, 0] - self.remaining_processing_time
        accumulated_processing_time_per_family = np.zeros(self.N_F)
        for j in range(self.N_J):
            accumulated_processing_time_per_family[self.job_family[j, 0] - 1] += accumulated_processing_time[j]
        num_of_finished_job_per_family = np.zeros(self.N_F)
        for i in range(self.remaining_processing_time.shape[0]):
            if self.remaining_processing_time[i] == 0:
                num_of_finished_job_per_family[self.job_family[i, 0] - 1] += 1
        self.utilization_state[:, 0] = accumulated_processing_time_per_family
        self.utilization_state[:, 1] = self.accumulated_setup_time_per_family
        self.utilization_state[:, 2] = num_of_finished_job_per_family
        
        self.action_history_state = np.zeros((self.N_F, 2))
        self.action_history_state[f_k, 0] = 1
        self.action_history_state[g_k, 1] = 1

        obs = self.get_observations(self.in_progress_job_state, self.setup_time_state, self.utilization_state, self.action_history_state)
                
        self.step_count += 1
        #obs = None
        reward = None
        done = True if np.max(self.waiting_job_per_family) == 0 else False
        return obs, reward, done, self.feasible_actions.flatten(), {}


    def get_observations(self, in_progress_job_state, setup_time_state, utilization_state, action_history_state):
        return np.hstack((in_progress_job_state, setup_time_state, utilization_state, action_history_state))
