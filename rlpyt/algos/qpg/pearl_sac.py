''' A re-implementation of PEARL algorithm https://github.com/katerakelly/oyster
'''
import numpy as np
import torch
from collections import namedtuple

from exptools.logging import logger
from rlpyt.utils.quick_args import save__init__args
from rlpyt.replays.non_sequence.uniform import (UniformReplayBuffer,
    AsyncUniformReplayBuffer)
from rlpyt.replays.non_sequence.n_step import SamplesFromReplay
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.tensor import infer_leading_dims
from rlpyt.samplers.collections import Context
from rlpyt.replays.multitask import MultitaskReplayBuffer
from rlpyt.algos.base import MetaRlAlgorithm
from rlpyt.algos.qpg.sac import SAC, OptInfo

def merge_same_task_batch(T, B, *tensors):
    """ Assuming each tensor is a torch.Tensor with leading dim (T, B) \\
    According to the sampler with one task, the (T,B) dimension can be merged together
    as a batch of data for the same task.
    """
    result = []
    for tensor in tensors:
        assert tensor.shape[0] == T and tensor.shape[1] == B
        result.append(tensor.view(T * B, -1))
    return (*result)

def merge_different_tasks_batch(*tasks_tensors):
    """  Merge and transpose the tensors so that it can be infered directly by context
    encoder model. The returned each data should be in (T*B, Task, ...) shape
        NOTE: element in `tasks_tensors` should be a list, the element in that "list"
        should be batch-wise feature with (T*B, ...) shape, placed in the tasks order.
    """
    result = []
    for tasks_tensor in tasks_tensors:
        tasks_tensor = torch.stack(tasks_tensor, dim=0).transpose(0,1)
        result.append(tasks_tensor.contigous())
    return (*result)

def embed_tasks_into_batch(tasks_samples_from_replay):
    """ squeeze (Task, T, B) leading dimension into (T*B, Task), where Task dimension is
    originally put as dictionary items
        NOTE: Currently only PEARL is merging (T,B) into time dimension and put different \
tasks into batch dimension. To improve, implementing transpose method in \
namedarraytuple should be better.
        Some strange constant numbers, please refer to how samples from replay is constructed.
        (e.g. rlpyt.replays.non_sequence.n_step.NStepReturnBuffer.extract_batch)
    """
    tasks_agent_inputs = ([],[],[])
    tasks_samples_from_replay_rests = ([],[],[],[],[],[])
    tasks_target_inputs = ([],[],[])

    for task, samples_from_replay in tasks_samples_from_replay.items():
        _, T, B, _ = infer_leading_dims(samples_from_replay.reward, dim=1)
        agent_inputs = merge_same_task_batch(T, B, *samples_from_replay.agent_inputs)
        samples_from_replay_rests = merge_same_task_batch(T, B,
            samples_from_replay.action,
            samples_from_replay.return_,
            samples_from_replay.reward,
            samples_from_replay.next_observation,
            samples_from_replay.done,
            samples_from_replay.done_n,
        )
        target_inputs = merge_same_task_batch(T, B, *samples_from_replay.target_inputs)
        for tasks_agent_input, agent_input in zip(tasks_agent_inputs, agent_inputs):
            tasks_agent_input.append(agent_input)
        for tasks_samples_from_replay_rest, samples_from_replay_rest in zip(tasks_samples_from_replay_rests, samples_from_replay_rests):
            tasks_samples_from_replay_rest.append(samples_from_replay_rest)
        for tasks_target_input, target_input in zip(tasks_target_inputs, target_inputs):
            tasks_target_input.append(target_input)
    tasks_agent_inputs = merge_different_tasks_batch(*tasks_agent_inputs)
    tasks_samples_from_replay_rests = merge_different_tasks_batch(*tasks_samples_from_replay_rests)
    tasks_target_inputs = merge_different_tasks_batch(*tasks_target_inputs)
    # now, each element in this three variable should be tensor with shape (T*B, Tasks, ...)
    samples_tasks_from_replay = SamplesFromReplay(
        agent_inputs = AgentInputs(*tasks_agent_inputs),
        action = tasks_samples_from_replay_rests[0],
        return_ = tasks_samples_from_replay_rests[1],
        reward = tasks_samples_from_replay_rests[2],
        next_observation = tasks_samples_from_replay_rests[3],
        done = tasks_samples_from_replay_rests[4],
        done_n = tasks_samples_from_replay_rests[5],
        target_inputs = AgentInputs(*tasks_target_inputs),
    )
    return samples_tasks_from_replay

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
        self.tasks = tasks_examples.keys()
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

    def samples_to_buffer(self, tasks_samples):
        return dict([
            (task, super().samples_to_buffer(samples)) for task, samples in tasks_samples.items()
        ])

    def loss(self, tasks_samples_from_replay):
        """ SAC said: Samples have leading batch dimension [B,..] (but not time). \\
            But the source code seems not show this behavior
        """
        
        # build context from samples in batch for the agent encoder
        samples_tasks_from_replay = embed_tasks_into_batch(tasks_samples_from_replay)
        agent_inputs, target_inputs, action, reward, next_observation, done = buffer_to((
            samples_tasks_from_replay.agent_inputs,
            samples_tasks_from_replay.target_inputs,
            samples_tasks_from_replay.action,
            samples_tasks_from_replay.reward,
            samples_tasks_from_replay.next_observation,
            samples_tasks_from_replay.done,
        ))

        # ################## start calculating and make computation graph ##############
        self.agent.reset()
        self.agent.infer_posterior(Context(
            observation=agent_inputs.observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done,
        ))
        latent_z = self.agent.zs
        # as designed, latent_z is stored in self.agent, we don't need to put it on the table for now.
        q1, q2 = self.agent.q(*agent_inputs, action, latent_z)
        with torch.no_grad():
            target_action, target_log_pi, _ = self.agent.pi(*target_inputs, latent_z)
            target_q1, target_q2 = self.agent.target_q(*target_inputs, target_action, latent_z)
        min_target_q = torch.min(target_q1, target_q2)
        target_value = min_target_q - self._alpha * target_log_pi
        disc = self.discount ** self.n_step_return
        y = (self.reward_scale * samples_tasks_from_replay.return_ + 
            (1 - samples_tasks_from_replay.done_n.float()) * disc * target_value)
        # y: target for Q functions, target_value

        q1_loss = 0.5 * torch.mean((y - q1) ** 2)
        q2_loss = 0.5 * torch.mean((y - q2) ** 2)

        new_action, log_pi, (pi_mean, pi_log_std) = self.agent.pi(*agent_inputs, latent_z)
        if not self.reparameterize:
            new_action = new_action.detach()  # No grad.
        log_target1, log_target2 = self.agent.q(*agent_inputs, new_action, latent_z)



    def optimize_agent(self, itr, tasks_samples= None, sampler_itr= None):
        assert sampler_itr is None, "Not implemented async version for PEARL SAC"
        if tasks_samples is not None:
            tasks_samples_to_buffer = self.samples_to_buffer(tasks_samples)
            self.replay_buffer.append_samples(tasks_samples_to_buffer)
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        if itr < self.min_itr_learn:
            return opt_info
        for _ in range(self.updates_per_optimize):
            # NOTE: now is different from original sac
            tasks_choices = np.random.randint(len(self.tasks))
            tasks_batch = self.tasks[tasks_choices]
            tasks_samples_from_replay = self.replay_buffer.sample_batch(
                tasks= tasks_batch, batch_B= self.batch_size
            )
            # Now, tasks_samples_from_replays are a dictionary with (num_tasks, batch_size, feat)
            
