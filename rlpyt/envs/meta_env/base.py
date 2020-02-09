""" Implementing the interface for multitask environment that can be used in this
code-base. \\
    NOTE: each instance of the environment should maintain only one task
"""
from rlpyt.envs.base import Env

class MultitaskEnv(Env):
    ''' The environment that handles different task.
    By task, it means an object accessable for this env instance.
    '''
    def __init__(self, **kwargs):
        pass

    # Maybe none of the method is implemented in the class
    @staticmethod
    def static_sample_tasks(n_tasks):
        """ Return a list of tasks that is valid in this environment. \\
            We don't specify that `task` should be in here, as long as each
            task can be accepted by the `__init__` function of its own.
        """
        raise NotImplementedError
    def sample_tasks(self, n_tasks):
        """ To meet some of the sampling tasks method of those multitask \\
            environment,
            This should be called when there is an instance of the MultitaskEnv.
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
