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
from rlpyt.algos.qpg.sac import SAC, SamplesToBuffer

OptInfo = namedtuple("OptInfo",
    ["q1Loss", "q2Loss", "piLoss",
    "q1GradNorm", "q2GradNorm", "piGradNorm",
    "q1", "q2", "piMu", "piLogStd", "qMeanDiff", "alpha", "klLoss", "klGradNorm"])

def merge_TB_in_each_task_batch(T, B, *tensors):
    """ Assuming each tensor is a torch.Tensor with leading dim (T, B) \\
    According to the sampler with one task, the (T,B) dimension can be merged together
    as a batch of data for the same task.
    """
    result = []
    for tensor in tensors:
        # assert tensor.shape[0] == T and tensor.shape[1] == B # No need to do this
        result.append(tensor.view(T * B, -1))
    return tuple(result)

def merge_different_tasks_batch(*tasks_tensors):
    """  Merge and transpose the tensors so that it can be infered directly by context
    encoder model. The returned each data should be in (T*B, Task, ...) shape
        NOTE: element in `tasks_tensors` should be a list, the element in that "list"
        should be batch-wise feature with (T*B, ...) shape, placed in the tasks order.
    """
    result = []
    for tasks_tensor in tasks_tensors:
        tasks_a_tensor = torch.stack(tasks_tensor, dim=0).transpose(0,1)
        result.append(tasks_a_tensor.contiguous())
    return tuple(result)

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

    for samples_from_replay in tasks_samples_from_replay:
        _, T, B, _ = infer_leading_dims(samples_from_replay.reward, dim=0)
        agent_inputs = merge_TB_in_each_task_batch(T, B, *samples_from_replay.agent_inputs)
        samples_from_replay_rests = merge_TB_in_each_task_batch(T, B,
            samples_from_replay.action,
            samples_from_replay.return_,
            samples_from_replay.reward,
            samples_from_replay.next_observation,
            samples_from_replay.done,
            samples_from_replay.done_n,
        )
        target_inputs = merge_TB_in_each_task_batch(T, B, *samples_from_replay.target_inputs)
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
    # NOTE: Due to data merging function, some of the scalar value need to be squeezed
    samples_tasks_from_replay = SamplesFromReplay(
        agent_inputs = AgentInputs(*tasks_agent_inputs),
        action = tasks_samples_from_replay_rests[0],
        return_ = tasks_samples_from_replay_rests[1].squeeze(2),
        reward = tasks_samples_from_replay_rests[2].squeeze(2),
        next_observation = tasks_samples_from_replay_rests[3],
        done = tasks_samples_from_replay_rests[4].squeeze(2),
        done_n = tasks_samples_from_replay_rests[5].squeeze(2),
        target_inputs = AgentInputs(*tasks_target_inputs),
    )
    return samples_tasks_from_replay

class PEARL_SAC(MetaRlAlgorithm, SAC):
    ''' Based on the need of algorithm,
            PEARL agent need following interface:
                pi_parameters(), q1_parameters(), q2_parameters(),
                .qf1, .qf2, .vf several pytorch module

    '''
    def __init__(self,
            n_tasks_per_update= 5, # the number of tasks to sample in one optim.step() call
            kl_lambda= 1, # the scaling factor when computing context encoder KL_div loss
            **kwargs,
            ):
        self.n_tasks_per_update = n_tasks_per_update
        self.kl_lambda= kl_lambda
        super(PEARL_SAC, self).__init__(**kwargs)

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
        self.updates_per_optimize = max(1, int(self.replay_ratio * sampler_bs /
            self.batch_size))
        logger.log(f"From sampler batch size {sampler_bs}, training "
            f"batch size {self.batch_size}, and replay ratio "
            f"{self.replay_ratio}, computed {self.updates_per_optimize} "
            f"updates per iteration.")
        self.min_itr_learn = self.min_steps_learn // sampler_bs
        agent.give_min_itr_learn(self.min_itr_learn)
        self.n_tasks = len(tasks_examples)
        self.initialize_replay_buffer(tasks_examples, batch_spec)
        self.optim_initialize(rank)

    def optim_initialize(self, rank= 0):
        super(PEARL_SAC, self).optim_initialize(rank)
        self.context_optimizer = self.OptimCls(self.agent.encoder_model_parameters(),
            lr= self.learning_rate, **self.optim_kwargs)

    def initialize_replay_buffer(self, tasks_example, batch_spec, async_=False):
        ''' Build a replay_buffer and assign it to `self.replay_buffer
        '''
        if self.bootstrap_timelimit:
            # I didn't figure out what this attribute is, so I didn't implement it.
            raise NotImplementedError
        SingleReplayBufferCls = AsyncUniformReplayBuffer if async_ else UniformReplayBuffer
        tasks_example_to_buffer = []
        for example in tasks_example:
            example_to_buffer = SamplesToBuffer(
                observation=example["observation"],
                action=example["action"],
                reward=example["reward"],
                done=example["done"],
                next_observation=example["next_observation"]
            )
            tasks_example_to_buffer.append(example_to_buffer)
        replay_kwargs = dict(
            SingleReplayBufferCls= SingleReplayBufferCls,
            tasks_example= tasks_example_to_buffer,
            size=self.replay_size,
            B=batch_spec.B,
            n_step_return=self.n_step_return,
        )
        self.replay_buffer = MultitaskReplayBuffer(**replay_kwargs)

    def samples_to_buffer(self, tasks_samples):
        return [
            super(PEARL_SAC, self).samples_to_buffer(samples) for samples in tasks_samples
        ]

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
            tasks_samples_from_replay = self.replay_buffer.sample_batch(
                batch_B= self.batch_size
            )
            # To make the batch size smaller, pick a subset of tasks to train. All tasks will be trained
            _num_upd8s = max(1, int(self.n_tasks / self.n_tasks_per_update))
            for ite_i in range(_num_upd8s):
                tasks_samples_from_replay_batch = tasks_samples_from_replay[(ite_i*self.n_tasks_per_update):((ite_i+1)*self.n_tasks_per_update)]
                # Now, tasks_samples_from_replays are a dictionary with (num_tasks, batch_size, feat)
                losses, values = self.loss(tasks_samples_from_replay_batch)
                q1_loss, q2_loss, pi_loss, alpha_loss, kl_loss = losses

                # ### Context model optimization is above all procedures
                self.context_optimizer.zero_grad()

                if kl_loss is not None:
                    kl_loss.backward(retain_graph= True)
                    kl_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.encoder_model_parameters(),
                        self.clip_grad_norm)
                else:
                    kl_grad_norm = 0

                if alpha_loss is not None:
                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                    self._alpha = torch.exp(self._log_alpha.detach())

                self.pi_optimizer.zero_grad()
                pi_loss.backward()
                pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.pi_parameters(),
                    self.clip_grad_norm)
                self.pi_optimizer.step()

                # Step Q's last because pi_loss.backward() uses them?
                self.q1_optimizer.zero_grad()
                q1_loss.backward()
                q1_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.q1_parameters(),
                    self.clip_grad_norm)
                self.q1_optimizer.step()

                self.q2_optimizer.zero_grad()
                q2_loss.backward()
                q2_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.q2_parameters(),
                    self.clip_grad_norm)
                self.q2_optimizer.step()

                self.context_optimizer.step()

                grad_norms = (q1_grad_norm, q2_grad_norm, pi_grad_norm, kl_grad_norm)

                self.append_opt_info_(opt_info, losses, grad_norms, values)
                self.update_counter += 1
                if self.update_counter % self.target_update_interval == 0:
                    self.agent.update_target(self.target_update_tau)

        return opt_info

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
        self.agent.reset(batch_size= samples_tasks_from_replay.reward.shape[1])
        self.agent.infer_posterior(Context(
            observation=agent_inputs.observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done,
        ))
        latent_z = self.agent.zs

        # Compute KL divergence of context encoder (whose prediction), and name it kl_loss
        if self.agent.encoder_model_kwargs["use_information_bottleneck"]:
            kl_loss = self.kl_lambda * torch.mean(self.agent.compute_latent_KL())
        else:
            kl_loss = None

        # compute as original SAC doing
        q1, q2 = self.agent.q(*agent_inputs, action, latent_z.detach())
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

        new_action, log_pi, (pi_mean, pi_log_std) = self.agent.pi(*agent_inputs, latent_z.detach())
        if not self.reparameterize:
            raise NotImplementedError
            # new_action = new_action.detach()  # No grad.
        log_target1, log_target2 = self.agent.q(*agent_inputs, new_action, latent_z)
        min_log_target = torch.min(log_target1, log_target2)
        prior_log_pi = self.get_action_prior(new_action.cpu())

        if self.reparameterize:
            pi_losses = self._alpha * log_pi - min_log_target - prior_log_pi
        else:
            raise NotImplementedError

        pi_loss = torch.mean(pi_losses)

        if self.target_entropy is not None:
            alpha_losses = - self._log_alpha * (log_pi.detach() + self.target_entropy)
            alpha_loss = torch.mean(alpha_losses)
        else:
            alpha_loss = None

        losses = (q1_loss, q2_loss, pi_loss, alpha_loss, kl_loss)
        assert not (torch.isnan(q1_loss) or torch.isnan(q2_loss) or torch.isnan(pi_loss) or torch.isnan(alpha_loss) or torch.isnan(kl_loss))
        values = tuple(val.detach() for val in (q1, q2, pi_mean, pi_log_std))
        return losses, values


    def append_opt_info_(self, opt_info, losses, grad_norms, values):
        """ append all the `losses` and `grad_norms` and `values` into each attribute 
            of `opt_info`
        """
        q1_loss, q2_loss, pi_loss, alpha_loss, kl_loss = losses
        q1_grad_norm, q2_grad_norm, pi_grad_norm, kl_grad_norm = grad_norms
        q1, q2, pi_mean, pi_log_std = values
        opt_info.q1Loss.append(q1_loss.item())
        opt_info.q2Loss.append(q2_loss.item())
        opt_info.piLoss.append(pi_loss.item())
        opt_info.q1GradNorm.append(q1_grad_norm)
        opt_info.q2GradNorm.append(q2_grad_norm)
        opt_info.piGradNorm.append(pi_grad_norm)
        opt_info.q1.extend(q1[::10].numpy())  # Downsample for stats.
        opt_info.q2.extend(q2[::10].numpy())
        opt_info.piMu.extend(pi_mean[::10].numpy())
        opt_info.piLogStd.extend(pi_log_std[::10].numpy())
        opt_info.qMeanDiff.append(torch.mean(abs(q1 - q2)).item())
        opt_info.alpha.append(self._alpha.item())
        opt_info.klLoss.append(kl_loss.item())
        opt_info.klGradNorm.append(kl_grad_norm)
