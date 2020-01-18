
import numpy as np

from rlpyt.samplers.collectors import BaseEvalCollector
from rlpyt.samplers.collections import Context
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.buffer import buffer_from_example, torchify_buffer, numpify_buffer
from exptools.logging import logger
from rlpyt.utils.quick_args import save__init__args

# For sampling, serial sampler can use Cpu collectors.


class SerialEvalCollector(BaseEvalCollector):
    """Does not record intermediate data."""

    def __init__(
            self,
            envs,
            agent,
            TrajInfoCls,
            max_T,
            max_trajectories=None,
            ):
        save__init__args(locals())

    def collect_evaluation(self, itr):
        traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]
        completed_traj_infos = list()
        observations = list()
        for env in self.envs:
            observations.append(env.reset())
        observation = buffer_from_example(observations[0], len(self.envs))
        for b, o in enumerate(observations):
            observation[b] = o
        action = buffer_from_example(self.envs[0].action_space.null_value(),
            len(self.envs))
        reward = np.zeros(len(self.envs), dtype="float32")
        obs_pyt, act_pyt, rew_pyt = torchify_buffer((observation, action, reward))
        self.agent.reset()
        self.agent.eval_mode(itr)
        for t in range(self.max_T):
            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)
            for b, env in enumerate(self.envs):
                o, r, d, env_info = env.step(action[b])
                traj_infos[b].step(observation[b], action[b], r, d,
                    agent_info[b], env_info)
                if getattr(env_info, "traj_done", d):
                    completed_traj_infos.append(traj_infos[b].terminate(o))
                    traj_infos[b] = self.TrajInfoCls()
                    o = env.reset()
                if d:
                    action[b] = 0  # Prev_action for next step.
                    r = 0
                    self.agent.reset_one(idx=b)
                observation[b] = o
                reward[b] = r
            if (self.max_trajectories is not None and
                    len(completed_traj_infos) >= self.max_trajectories):
                logger.log("Evaluation reached max num trajectories "
                    f"({self.max_trajectories}).")
                break
        if t == self.max_T - 1:
            logger.log("Evaluation reached max num time steps "
                f"({self.max_T}).")
        return completed_traj_infos

class SerialContextEvalCollector(BaseEvalCollector):
    """ Different from SerialEvalCollector
    """
    def __init__(self,
            envs,
            agent,
            TrajInfoCls,
            max_T,
            infer_context_period,
            max_trajectories=None,
            ):
        save__init__args(locals())
        self.build_context_buffer()

    def build_context_buffer(self):
        """ Build a Context whose leading dimension will be (infer_context_period, batch_B)
            and write to `self.context_buffer`
        """
        observation_shape = self.envs[0].observation_space.shape
        action_shape = self.envs[0].action_space.shape
        observations_buffer = np.zeros((
            self.infer_context_period, len(self.envs),
            *observation_shape
        )).astype("float32")
        actions_buffer = np.zeros((
            self.infer_context_period, len(self.envs),
            *action_shape
        )).astype("float32")
        rewards_buffer = np.zeros((
            self.infer_context_period, len(self.envs),
            1
        )).astype("float32")
        next_observations_buffer = np.zeros((
            self.infer_context_period, len(self.envs),
            *observation_shape
        )).astype("float32")
        dones_buffer = np.zeros((
            self.infer_context_period, len(self.envs),
            1
        )).astype("float32")
        self.context_buffer = Context(
            observation= observations_buffer,
            action= actions_buffer,
            reward= rewards_buffer,
            next_observation= next_observations_buffer,
            done= dones_buffer
        )

    def collect_evaluation(self, itr):
        traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]
        completed_traj_infos = list() # a list of TrajInfoCls instance
        observations = list()
        for env in self.envs:
            observations.append(env.reset())
        observation = buffer_from_example(observations[0], len(self.envs))
        for b, o in enumerate(observations):
            observation[b] = o
        action = buffer_from_example(self.envs[0].action_space.null_value(),
            len(self.envs))
        reward = np.zeros(len(self.envs), dtype="float32")
        obs_pyt, act_pyt, rew_pyt = torchify_buffer((observation, action, reward))
        self.agent.reset(batch_size=len(self.envs))
        self.agent.eval_mode(itr)
        for t in range(self.max_T):
            self.context_buffer.observation[t % self.infer_context_period] = observation
            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)
            self.context_buffer.action[t % self.infer_context_period] = action
            for b, env in enumerate(self.envs):
                o, r, d, env_info = env.step(action[b])
                traj_infos[b].step(observation[b], action[b], r, d,
                    agent_info[b], env_info)
                if getattr(env_info, "traj_done", d):
                    completed_traj_infos.append(traj_infos[b].terminate(o))
                    traj_infos[b] = self.TrajInfoCls()
                    o = env.reset()
                if d:
                    action[b] = 0  # Prev_action for next step.
                    r = 0
                    self.agent.reset_one(idx=b)
                observation[b] = o
                reward[b] = r
                self.context_buffer.done[t % self.infer_context_period, b] = d
            self.context_buffer.reward[t % self.infer_context_period] = reward
            self.context_buffer.next_observation[t % self.infer_context_period] = observation
            if t > 0 and t % self.infer_context_period == 0:
                context = torchify_buffer(self.context_buffer)
                self.agent.infer_posterior(context)
            if (self.max_trajectories is not None and
                    len(completed_traj_infos) >= self.max_trajectories):
                logger.log("Evaluation reached max num trajectories "
                    f"({self.max_trajectories}).")
                break
        if t == self.max_T - 1:
            logger.log("Evaluation reached max num time steps "
                f"({self.max_T}).")
        return completed_traj_infos