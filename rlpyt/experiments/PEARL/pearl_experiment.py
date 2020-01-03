""" The entry point to run a single experiment for PEARL
"""
import sys

from exptools.launching.affinity import affinity_from_code
from exptools.launching.variant import load_variant
from exptools.logging.context import logger_context

from rlpyt.samplers.serial.sampler import SerialMultitaskSampler
from rlpyt.envs.meta_env.rand_param_env_wrappers import RandParamEnv
from rlpyt.algos.qpg.pearl_sac import PEARL_SAC
from rlpyt.agents.qpg.pearl_sac_agent import PearlSacAgent
from rlpyt.runners.meta_rl import MetaRlBase

from rand_param_envs.walker2d_rand_params import Walker2DRandParamsEnv

def build_and_train(affinity_code, log_dir, run_ID, **kwargs):
    # I prefer put all tunable default configs into launch file
    affinity = affinity_from_code(affinity_code)
    config = load_variant(log_dir)

    # make a environment and extract tasks, sample at once prevents tasks duplicate
    # NOTE: this instance will no be used in training/testing
    env_ = Walker2DRandParamsEnv()
    tasks = env_.sample_tasks(config["tasks"]["n_train_tasks"] + config["tasks"]["n_eval_tasks"])
    train_tasks = tasks[:config["tasks"]["n_train_tasks"]]
    eval_tasks = tasks[config["tasks"]["n_train_tasks"]]
    train_env_kwargs = dict([
        (task, dict(task= task)) for task in train_tasks
    ])
    eval_env_kwargs = dict([
        (task, dict(task= task)) for task in eval_tasks
    ])

    sampler = SerialMultitaskSampler(
        EnvCls= RandParamEnv,
        tasks_env_kwargs= train_env_kwargs,
        eval_tasks_env_kwargs= eval_env_kwargs,
        **config["sampler"]
    )
    algo = PEARL_SAC(**config["algo"])
    agent = PearlSacAgent(**config["agent"])
    runner = MetaRlBase(
        algo= algo,
        agent= agent,
        sampler= sampler,
        n_steps= 50e6,
        log_interval_steps= 1e5,
        affinity= affinity
    )
    name = "pearl_walker2d_rand_params"

    with logger_context(log_dir, run_ID, name, config):
        runner.train()

if __name__ == "__main__":
    build_and_train(*sys.argv[1:])