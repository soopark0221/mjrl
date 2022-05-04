"""
Job script to learn policy using MOReL
"""

from os import environ
environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
environ['MKL_THREADING_LAYER']='GNU'
import numpy as np
import copy
import torch
import torch.nn as nn
import pickle
import mjrl.envs
import time as timer
import argparse
import os
import json
import mjrl.samplers.core as sampler
import mjrl.utils.tensor_utils as tensor_utils
from tqdm import tqdm
from tabulate import tabulate
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.utils.gym_env import GymEnv
from mjrl.utils.logger import DataLog
from mjrl.utils.make_train_plots import make_train_plots
from mjrl.algos.mbrl.nn_dynamics import WorldModel
from mjrl.algos.mbrl.model_based_npg import ModelBasedNPG
from mjrl.algos.mbrl.sampling import sample_paths, evaluate_policy
import time 
# ===============================================================================
# Get command line arguments
# ===============================================================================

parser = argparse.ArgumentParser(description='Model accelerated policy optimization.')
parser.add_argument('--output', '-o', type=str, required=True, help='location to store results')
parser.add_argument('--config', '-c', type=str, required=True, help='path to config file with exp params')
parser.add_argument('--include', '-i', type=str, required=False, help='package to import')
parser.add_argument('--mdl', default = 'ensemble', type=str)
parser.add_argument('--param_dict_fname', default='param_dict', type=str)
parser.add_argument('--pess', default=3.0, type=float)

args = parser.parse_args()
OUT_DIR = args.output
if not os.path.exists(OUT_DIR): os.mkdir(OUT_DIR)
if not os.path.exists(OUT_DIR+'/iterations'): os.mkdir(OUT_DIR+'/iterations')
if not os.path.exists(OUT_DIR+'/logs'): os.mkdir(OUT_DIR+'/logs')
with open(args.config, 'r') as f:
    job_data = eval(f.read())
if args.include: exec("import "+args.include)

# Unpack args and make files for easy access
logger = DataLog()
ENV_NAME = job_data['env_name']
EXP_FILE = OUT_DIR + '/job_data.json'
SEED = job_data['seed']

# base cases
if 'eval_rollouts' not in job_data.keys():  job_data['eval_rollouts'] = 0
if 'save_freq' not in job_data.keys():      job_data['save_freq'] = 10
if 'device' not in job_data.keys():         job_data['device'] = 'cpu'
if 'hvp_frac' not in job_data.keys():       job_data['hvp_frac'] = 1.0
if 'start_state' not in job_data.keys():    job_data['start_state'] = 'init'
if 'learn_reward' not in job_data.keys():   job_data['learn_reward'] = True
if 'num_cpu' not in job_data.keys():        job_data['num_cpu'] = 1
if 'npg_hp' not in job_data.keys():         job_data['npg_hp'] = dict()
if 'act_repeat' not in job_data.keys():     job_data['act_repeat'] = 1
if 'model_file' not in job_data.keys():     job_data['model_file'] = None

assert job_data['start_state'] in ['init', 'buffer']
assert 'data_file' in job_data.keys()
with open(EXP_FILE, 'w') as f:  json.dump(job_data, f, indent=4)
del(job_data['seed'])
job_data['base_seed'] = SEED


# ===============================================================================
# Helper functions
# ===============================================================================
def buffer_size(paths_list):
    return np.sum([p['observations'].shape[0]-1 for p in paths_list])


# ===============================================================================
# Setup functions and environment
# ===============================================================================

np.random.seed(SEED)
torch.random.manual_seed(SEED)

if ENV_NAME.split('_')[0] == 'dmc':
    # import only if necessary (not part of package requirements)
    import dmc2gym
    backend, domain, task = ENV_NAME.split('_')
    e = dmc2gym.make(domain_name=domain, task_name=task, seed=SEED)
    e = GymEnv(e, act_repeat=job_data['act_repeat'])
else:
    e = GymEnv(ENV_NAME, act_repeat=job_data['act_repeat'])
    e.set_seed(SEED)

# check for reward and termination functions
if 'reward_file' in job_data.keys():
    import sys
    splits = job_data['reward_file'].split("/")
    dirpath = "" if splits[0] == "" else os.path.dirname(os.path.abspath(__file__))
    for x in splits[:-1]: dirpath = dirpath + "/" + x
    filename = splits[-1].split(".")[0]
    sys.path.append(dirpath)
    exec("from "+filename+" import *")
if 'reward_function' not in globals():
    reward_function = getattr(e.env.env, "compute_path_rewards", None)
    print(f'reward_function is {reward_function}')
    job_data['learn_reward'] = False if reward_function is not None else True
if 'termination_function' not in globals():
    termination_function = getattr(e.env.env, "truncate_paths", None)
if 'obs_mask' in globals(): e.obs_mask = obs_mask

# ===============================================================================
# Setup policy, model, and agent
# ===============================================================================

if job_data['model_file'] is not None:
    model_trained = True
    models = pickle.load(open(job_data['model_file'], 'rb'))
else:
    model_trained = False
    models = [WorldModel(state_dim=e.observation_dim, act_dim=e.action_dim, seed=SEED+i, 
                     **job_data) for i in range(job_data['num_models'])]

# Construct policy and set exploration level correctly for NPG
if 'init_policy' in job_data.keys():
    policy = pickle.load(open(job_data['init_policy'], 'rb'))
    policy.set_param_values(policy.get_param_values())
    init_log_std = job_data['init_log_std']
    min_log_std = job_data['min_log_std']
    if init_log_std:
        params = policy.get_param_values()
        params[:policy.action_dim] = tensor_utils.tensorize(init_log_std)
        policy.set_param_values(params)
    if min_log_std:
        policy.min_log_std[:] = tensor_utils.tensorize(min_log_std)
        policy.set_param_values(policy.get_param_values())
else:
    policy = MLP(e.spec, seed=SEED, hidden_sizes=job_data['policy_size'], 
                    init_log_std=job_data['init_log_std'], min_log_std=job_data['min_log_std'])

baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=256, epochs=1,  learn_rate=1e-3,
                       device=job_data['device'])
if args.mdl == 'ensemble' or args.mdl == 'swag_ens':               
    agent = ModelBasedNPG(learned_model=models, env=e, policy=policy, baseline=baseline, seed=SEED,
                      normalized_step_size=job_data['step_size'], save_logs=True, 
                      reward_function=reward_function, termination_function=termination_function,
                      mdl=args.mdl, param_dict_fname=None,
                      **job_data['npg_hp'])
elif args.mdl == 'swag' or args.mdl == 'multiswag':
    agent = ModelBasedNPG(learned_model=models, env=e, policy=policy, baseline=baseline, seed=SEED,
                      normalized_step_size=job_data['step_size'], save_logs=True, 
                      reward_function=reward_function, termination_function=termination_function,
                      mdl=args.mdl, param_dict_fname=args.param_dict_fname,
                      **job_data['npg_hp'])
# ===============================================================================
# Model training loop
# ===============================================================================

paths = pickle.load(open(job_data['data_file'], 'rb'))
init_states_buffer = [p['observations'][0] for p in paths]
best_perf = -1e8
ts = timer.time()
s = np.concatenate([p['observations'][:-1] for p in paths])
a = np.concatenate([p['actions'][:-1] for p in paths])
sp = np.concatenate([p['observations'][1:] for p in paths])
r = np.concatenate([p['rewards'][:-1] for p in paths])
rollout_score = np.mean([np.sum(p['rewards']) for p in paths])
num_samples = np.sum([p['rewards'].shape[0] for p in paths])
logger.log_kv('fit_epochs', job_data['fit_epochs'])
logger.log_kv('rollout_score', rollout_score)
logger.log_kv('iter_samples', num_samples)
logger.log_kv('num_samples', num_samples)
try:
    rollout_metric = e.env.env.evaluate_success(paths)
    logger.log_kv('rollout_metric', rollout_metric)
except:
    pass
if not model_trained:
    for i, model in enumerate(models):
        dynamics_loss = model.fit_dynamics(s, a, sp, **job_data)
        logger.log_kv('dyn_loss_' + str(i), dynamics_loss[-1])
        loss_general = model.compute_loss(s, a, sp) # generalization error
        logger.log_kv('dyn_loss_gen_' + str(i), loss_general)
        if job_data['learn_reward']:
            reward_loss = model.fit_reward(s, a, r.reshape(-1, 1), **job_data)
            logger.log_kv('rew_loss_' + str(i), reward_loss[-1])
else:
    for i, model in enumerate(models):
        loss_general = model.compute_loss(s, a, sp)
        #logger.log_kv('dyn_loss_gen_' + str(i), loss_general)
tf = timer.time()
logger.log_kv('model_learning_time', tf-ts)
print("Model learning statistics")
print_data = sorted(filter(lambda v: np.asarray(v[1]).size == 1,
                            logger.get_current_log().items()))
print(tabulate(print_data))
pickle.dump(models, open(OUT_DIR + '/models.pickle', 'wb'))
logger.log_kv('act_repeat', job_data['act_repeat']) # log action repeat for completeness

# ===============================================================================
# Pessimistic MDP parameters
# ===============================================================================
if args.mdl == 'ensemble' or args.mdl == 'swag_ens':
    delta = np.zeros(s.shape[0])
    for idx_1, model_1 in enumerate(models):
        pred_1 = model_1.predict(s, a)
        for idx_2, model_2 in enumerate(models):
            if idx_2 > idx_1:
                pred_2 = model_2.predict(s, a)
                disagreement = np.linalg.norm((pred_1-pred_2), axis=-1)
                delta = np.maximum(delta, disagreement)

elif args.mdl == 'swag':
    param_dict = pickle.load(open(args.param_dict_fname, 'rb'))
    pred = models[0].swag_predict(param_dict, s, a)
    #print(f'swag pred result is {pred}, len is {len(pred)}')
    delta = np.zeros(s.shape[0])
    for i in range(len(pred)):
        for j in range(len(pred)):
            if j>i:
                dis = np.linalg.norm((pred[i]-pred[j]), axis=-1)
                delta = np.maximum(delta, dis)

elif args.mdl == 'multiswag':
    delta = np.zeros(s.shape[0])
    pred = []
    for idx_1, model_1 in enumerate(models):
        param_dict = pickle.load(open(f'{args.param_dict_fname}{idx_1}', 'rb'))
        pred += models[idx_1].swag_predict(param_dict, s, a)
    #print(f'swag pred result is {pred}, len is {len(pred)}')
    for i in range(len(pred)):
        for j in range(len(pred)):
            if j>i:
                dis = np.linalg.norm((pred[i]-pred[j]), axis=-1)
                delta = np.maximum(delta, dis)
print(f'delta is {delta}, len delta is {len(delta)}')
if 'pessimism_coef' in job_data.keys():
    if job_data['pessimism_coef'] is None or job_data['pessimism_coef'] == 0.0:
        truncate_lim = None
        print("No pessimism used. Running naive MBRL.")
    else:
        #truncate_lim = (1.0 / job_data['pessimism_coef']) * np.max(delta)
        truncate_lim = (1.0 / args.pess) * np.max(delta)

        print("Maximum error before truncation (i.e. unknown region threshold) = %f" % truncate_lim)
    job_data['truncate_lim'] = truncate_lim
    job_data['truncate_reward'] = job_data['truncate_reward'] if 'truncate_reward' in job_data.keys() else 0.0
else:
    job_data['truncate_lim'] = None
    job_data['truncate_reward'] = 0.0

with open(EXP_FILE, 'w') as f:
    job_data['seed'] = SEED
    json.dump(job_data, f, indent=4)
    del(job_data['seed'])
