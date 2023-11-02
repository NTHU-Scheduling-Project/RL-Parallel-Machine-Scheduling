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
    pt_tmp = pd.read_excel("data/PMSP_dataset.xlsx",sheet_name="Processing Time",index_col =[0])
    jt_tmp = pd.read_excel("data/PMSP_dataset.xlsx",sheet_name="Job Type",index_col =[0])

    N_J = pt_tmp.shape[0] # number of jobs
    N_M = 8 # number of machines

    processing_time = [list(map(int, pt_tmp.iloc[i])) for i in range(N_J)]
    job_family = [list(map(int,jt_tmp.iloc[i])) for i in range(N_J)]
    processing_time = np.array(processing_time)
    job_family = np.array(job_family)
    remaining_processing_time = np.zeros(N_J)

    N_F = np.max(job_family[:, 0])

    env = PMSPEnv(args, N_J, N_M, N_F, processing_time, job_family)

    agent = DQN(n_actions = (N_F * N_F))

    runner = Runner(args, run_config, env, N_F)
    if args.test:
        runner.validate(agent, start_i=1, phase="test")
    else:
        runner.train(agent)
    