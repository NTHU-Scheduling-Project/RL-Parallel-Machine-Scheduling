import os
import copy
import pandas as pd
import numpy as np
from util import global_util
from util.util import parse_args
from env.env import PMSPEnv
from agent.dqn import DQN
from runner import Runner

if __name__ == '__main__':
    args = parse_args()
    args.output = os.path.join(args.output, "{}_{}".format(args.net, "dueling") if args.dueling else args.net)
    
    global_util.setup_logger()

    run_config = global_util.load_yaml("data/run_config.yml")
    # read input file
    #pt_tmp = pd.read_excel("data/PMSP_dataset.xlsx",sheet_name="Processing Time",index_col =[0])
    #jt_tmp = pd.read_excel("data/PMSP_dataset.xlsx",sheet_name="Job Type",index_col =[0])

    training_data = pd.read_excel("data/training_dataset.xlsx", sheet_name = "processing_time", index_col = [0, 1])
    training_data = np.array(training_data).reshape(65, 20, 3)
    
    validation_data = []
    for i in range(1, 17):
        pt_tmp = pd.read_excel(f"data/validation/dataset_{i}.xlsx",sheet_name="Processing Time",index_col =[0])
        jt_tmp = pd.read_excel(f"data/validation/dataset_{i}.xlsx",sheet_name="Job Type",index_col =[0])
        validation_data.append([np.array(pt_tmp), np.array(jt_tmp).astype(int)])

    job_family, processing_time = np.split(training_data, [2], axis=2)
    job_family = job_family.astype(int)
    processing_time = processing_time.astype(int)

    N_J = training_data.shape[1] # number of jobs per instance
    N_M = 4 # number of machines
    N_F = int(np.max(training_data[:, :, 0]))

    env = PMSPEnv(args, N_J, N_M, N_F, processing_time, job_family)

    agent = DQN(n_actions = (N_F * N_F))

    runner = Runner(args, run_config, env, N_F, training_data, validation_data)
    if args.test:
        runner.validate(agent, start_i=1, phase="test")
    else:
        runner.train(agent)
    