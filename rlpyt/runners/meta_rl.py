import torch

from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.seed import set_seed, make_seed
from rlpyt.runners.base import BaseRunner
from rlpyt.runners.minibatch_rl import MinibatchRlBase, MinibatchRl, MinibatchRlEval

from exptools.logging import logger
from exptools.launching.affinity import encode_affinity

class MetaRlBase(MinibatchRl, MinibatchRlEval, BaseRunner): # MinibatchRlBase is its grandparent
    ''' Runner controlling meta RL algorithm.
        In terms of meta-training, the tasks are pre-defined as a list.
        Environment should provide `reset_task(self, task)` method, that set the env to given task
            The `task` is returned by environment, should be a function called `get_train_task` and `get_test_task`.
        This runner contains samplers for each tasks (indexed by task)
    '''
    # def __init__(self, ...): the same as MinibatchRlBase

    def startup(self):
        ''' Since I did not know much about processor affinity, I didn't implement this yet
        '''
        # start system configurations (not experiment configuration)
        logger.log(f"Get affinity code: {encode_affinity(self.affinity)}")
        if self.seed is None:
            self.seed = make_seed()
        set_seed(self.seed)
        self.rank = rank = getattr(self, "rank", 0)
        self.world_size = world_size = getattr(self, "world_size", 1)

        # initialize all experiment components
        tasks_examples = self.sampler.initialize(
            agent=self.agent,
            affinity=self.affinity,
            seed=self.seed + 1,
            bootstrap_value=getattr(self.algo, "bootstrap_value", False),
            traj_info_kwargs=self.get_traj_info_kwargs(),
            rank=rank,
            world_size=world_size,
        )
        self.itr_batch_size = self.sampler.batch_spec.size * world_size
        n_itr = self.get_n_itr()
        self.agent.to_device(self.affinity.get("cuda_idx", None))
        if world_size > 1:
            self.agent.data_parallel()
        self.algo.initialize(
            agent=self.agent,
            n_itr=n_itr,
            batch_spec=self.sampler.batch_spec,
            mid_batch_reset=self.sampler.mid_batch_reset,
            tasks_examples=tasks_examples,
            world_size=world_size,
            rank=rank,
        )
        self.initialize_logging()
        return n_itr


    def train(self):
        ''' Perform collecting and training and evaluating
        '''
        n_itr = self.startup()

        # start the main experiment loop
        for itr in range(n_itr):
            with logger.prefix(f"itr #{itr} "):
                self.agent.sample_mode(itr)
                tasks_samples, tasks_traj_infos = self.sampler.obtain_samples(itr)
                self.agent.train_mode(itr)
                opt_info = self.algo.optimize_agent(itr, tasks_samples)
                self.store_diagnostics(itr, tasks_traj_infos, opt_info)
                if itr % self.log_interval_itrs == 0:
                    # NOTE: This has to be done at itr==0, because self.pbar needs initialization from here
                    tasks_eval_traj_infos, eval_time = self.evaluate_agent(itr)
                    self.log_diagnostics(itr, tasks_eval_traj_infos, eval_time)
        self.shutdown()

    def store_diagnostics(self, itr, tasks_traj_infos, opt_info):
        # Each value in tasks_traj_infos should be a list of TrajInfoCls instance.
        # I just add them together
        traj_infos = [i for j in tasks_traj_infos for i in j]
        super(MetaRlBase, self).store_diagnostics(itr, traj_infos, opt_info)

    def log_diagnostics(self, itr, tasks_eval_traj_infos, eval_time):
        # Each value in tasks_traj_infos should be a list of TrajInfoCls instance.
        # I just add them together
        traj_infos = [i for j in tasks_eval_traj_infos for i in j]
        super(MetaRlBase, self).log_diagnostics(itr)
