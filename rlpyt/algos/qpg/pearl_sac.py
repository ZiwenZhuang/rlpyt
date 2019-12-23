''' A re-implementation of PEARL algorithm https://github.com/katerakelly/oyster
'''
import numpy as np
import torch
from collections import namedtuple

from exptools.logging import logger
from rlpyt.utils.quick_args import save__init__args
from rlpyt.replays.non_sequence.uniform import (UniformReplayBuffer,
    AsyncUniformReplayBuffer)
from rlpyt.replays.multitask import MultitaskReplayBuffer
from rlpyt.algos.base import MetaRlAlgorithm
from rlpyt.algos.qpg.sac import SAC

class PEARL_SAC(MetaRlAlgorithm, SAC):
    ''' Based on the need of algorithm,
            PEARL agent need following interface:
                pi_parameters(), q1_parameters(), q2_parameters(),
                .qf1, .qf2, .vf several pytorch module

    '''

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset, tasks_examples,
            world_size=1, rank=0):
        """Used in basic or synchronous multi-GPU runners, not async.
        Parameters
        ----------
            agent: SacAgent
        """
        self.agent = agent
        self.n_itr = n_itr
        self.mid_batch_reset = mid_batch_reset
        self.sampler_bs = sampler_bs = batch_spec.size
        self.updates_per_optimize = int(self.replay_ratio * sampler_bs /
            self.batch_size)
        logger.log(f"From sampler batch size {sampler_bs}, training "
            f"batch size {self.batch_size}, and replay ratio "
            f"{self.replay_ratio}, computed {self.updates_per_optimize} "
            f"updates per iteration.")
        self.min_itr_learn = self.min_steps_learn // sampler_bs
        agent.give_min_itr_learn(self.min_itr_learn)
        self.initialize_replay_buffer(tasks_examples, batch_spec)
        self.optim_initialize(rank)

    def initialize_replay_buffer(self, tasks_example, batch_spec, async_=False):
        ''' Build a replay_buffer and assign it to `self.replay_buffer
        '''
        if getattr(self, "bootstrap_time_limit", False):
            # I didn't figure out what this attribute is, so I didn't implement it.
            raise NotImplementedError
        SingleReplayCls = AsyncUniformReplayBuffer if async_ else UniformReplayBuffer
        replay_kwargs = dict(
            SingleReplayCls= SingleReplayCls,
            tasks_example= tasks_example,
            size=self.replay_size,
            B=batch_spec.B,
            n_step_return=self.n_step_return,
        )
        self.replay_buffer = MultitaskReplayBuffer(**replay_kwargs)

    def optimize_agent(self, itr, tasks_samples= None, sampler_itr= None):
        assert sampler_itr is None, "Not implemented async version for PEARL SAC"

        
        
        

        