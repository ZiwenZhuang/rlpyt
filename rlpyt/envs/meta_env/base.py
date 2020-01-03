""" Implementing the interface for multitask environment that can be used in this
code-base. \\
    NOTE: each instance of the environment should maintain only one task
"""
from rlpyt.envs.base import Env

class MultitaskEnv(Env):
    def __init__(self, task= None, **kwargs):
        """ NOTE: environment should be able to be formed directly with input 
        `task` or sample one if `task` is None (depends on the implementation)
        """
        pass

    @staticmethod
    def smaple_tasks(n_tasks):
        """ Return a list of tasks that is valid in this environment. \\
            We don't specify that `task` should be in here, as long as each
            task can be accepted by the `__init__` function of its own.
        """
        raise NotImplementedError

    def set_task(self, task):
        """ Change task of current environment instance, whether to reset
        depends on the implementation of the environment.
        """
        pass

    def get_task(self):
        """ Get the task that the agent is performing in the current env
        instance.
        """
        raise NotImplementedError
