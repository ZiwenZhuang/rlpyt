""" https://github.com/dennisl88/rand_param_envs/tree/4d1529d61ca0d65ed4bd9207b108d4a4662a4da0
"""

from rand_param_envs.base import MetaEnv # just import here force python to check when you are using this file.
from rlpyt.spaces.gym_wrapper import GymSpaceWrapper
from rlpyt.envs.meta_env.base import MultitaskEnv
from rlpyt.envs.gym import info_to_nt
from rlpyt.envs.base import EnvStep, EnvSpaces

class RandParamEnv(MultitaskEnv):
    """ A interface wrapping for the rand_param_envs. \\
        To use it, please install the environment manually.
    """
    def __init__(self, EnvCls, task= None, **env_kwargs):
        """ Different from this repo protocol, the kwargs will be used directly to build
        one of the rand_param_envs given EnvCls.

        Args:
            EnvCls: the constructor to build one of the rand_param_envs
            task: NOTE: the task parameter you sampled from a wrapped environment instance
            env_kwargs: the kwargs that feed into EnvCls for building the environment instance
        """
        self._wrapped_env = EnvCls(**env_kwargs)
        self._wrapped_env.set_task(task)

        self.observation_space = GymSpaceWrapper(
            space=self._wrapped_env.observation_space,
            name="obs"
        )
        self.action_space = GymSpaceWrapper(
            space=self._wrapped_env.action_space,
            name="act"
        )

    def get_task(self):
        return self._wrapped_env.get_task()
    def set_task(self, task):
        # assuming no return value
        self._wrapped_env.set_task(task)
    def log_diagnostics(self, paths, prefix):
        # Did not checked yet, I haven't figure out what to do.
        raise NotImplementedError

    def step(self, action):
        a = self.action_space.revert(action)
        o, r, d, info = self._wrapped_env.step(a)
        obs = self.observation_space.convert(o)
        info = info_to_nt(info)
        return EnvStep(obs, r, d, info)