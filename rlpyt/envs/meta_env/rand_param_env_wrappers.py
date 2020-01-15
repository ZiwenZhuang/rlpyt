""" https://github.com/dennisl88/rand_param_envs/tree/4d1529d61ca0d65ed4bd9207b108d4a4662a4da0
"""

from rand_param_envs.base import MetaEnv # NOTE: the import here is to tell you what this file
# is trying to do to wrap rand-param-envs into use.
from rlpyt.spaces.float_box import FloatBox
from rlpyt.envs.meta_env.base import MultitaskEnv
from rlpyt.envs.gym import build_info_tuples, info_to_nt
from rlpyt.envs.base import EnvStep, EnvSpaces

import numpy as np

class RandParamEnv(MultitaskEnv):
    """ A interface wrapping for the rand_param_envs. \\
        To use it, please install the environment manually.
    """
    def __init__(self, EnvCls, task= None, **env_kwargs):
        """ Different from this repo protocol, the kwargs will be used directly to build
        one of the rand_param_envs given EnvCls.

        Args:
            EnvCls: the constructor to build one of the rand_param_envs
            task: In this case, all `task` send and revieced in this interface should be a string
            env_kwargs: the kwargs that feed into EnvCls for building the environment instance
        """
        self._wrapped_env = EnvCls(**env_kwargs)
        if not task is None:
            self._wrapped_env.set_task(task)

        # get them via @property method
        self._observation_space = FloatBox(low= self._wrapped_env.observation_space.low, high= self._wrapped_env.observation_space.high)
        self._action_space = FloatBox(low= self._wrapped_env.action_space.low, high= self._wrapped_env.action_space.high)

        # mimicing the usage of GymEnvWrapper
        _, _, _, info = self._wrapped_env.step(self._observation_space.null_value())
        build_info_tuples(info)

    def sample_tasks(self, n_tasks):
        return self._wrapped_env.sample_tasks(n_tasks)
    def get_task(self):
        return self._wrapped_env.get_task()
    def set_task(self, task):
        """ task: a string sampled from this instance """
        # assuming no return value
        self._wrapped_env.set_task(task)

    def log_diagnostics(self, paths, prefix):
        # Did not checked yet, I haven't figure out what to do.
        raise NotImplementedError

    def step(self, action):
        o, r, d, info = self._wrapped_env.step(action)
        o = o.astype(np.float32)
        info = info_to_nt(info)
        return EnvStep(o, r, d, info)
    
    def reset(self):
        return self._wrapped_env.reset().astype(np.float32)