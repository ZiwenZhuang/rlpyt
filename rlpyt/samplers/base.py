
from rlpyt.samplers.collectors import BaseCollector
from rlpyt.samplers.collections import BatchSpec, TrajInfo
from rlpyt.utils.quick_args import save__init__args


class BaseSampler:
    """Class which interfaces with the Runner, in master process only."""

    alternating = False

    def __init__(
            self,
            EnvCls,
            env_kwargs,
            batch_T,
            batch_B,
            CollectorCls= BaseCollector,
            max_decorrelation_steps=100,
            TrajInfoCls=TrajInfo,
            eval_n_envs=0,  # 0 for no eval setup.
            eval_CollectorCls=None,  # Must supply if doing eval.
            eval_env_kwargs=None,
            eval_max_steps=None,  # int if using evaluation.
            eval_max_trajectories=None,  # Optional earlier cutoff.
            ):
        eval_max_steps = None if eval_max_steps is None else int(eval_max_steps)
        eval_max_trajectories = (None if eval_max_trajectories is None else
            int(eval_max_trajectories))
        save__init__args(locals())
        self.batch_spec = BatchSpec(batch_T, batch_B)
        self.mid_batch_reset = CollectorCls.mid_batch_reset

    def initialize(self, *args, **kwargs):
        raise NotImplementedError

    def obtain_samples(self, itr):
        raise NotImplementedError  # type: Samples

    def evaluate_agent(self, itr):
        raise NotImplementedError

    def shutdown(self):
        pass

    @property
    def batch_size(self):
        return self.batch_spec.size  # For logging at least.

class MultitaskSampler:
    ''' A example for implementing meta-RL sampler.
    This is a wrapper of multiple Sampler, using dictionary (NOTE: key is necessarily not string)
    '''
    def __init__(self,
            SamplerCls,
            tasks_env_kwargs,
            **sampler_kwargs,
            ):
        '''
            param SamplerCls: the constructor to build sampler for a single task
            param tasks_env_kwargs: a dictionary with (task, env_kwargs) pairs
            param sampler_kwargs: the rest of kwargs are what you feed to a single sampler. \
                But recommending not env_kwargs item
        '''
        self.tasks = list(tasks_env_kwargs.keys())
        self.samplers = dict()
        sampler_kwargs.pop("env_kwargs") # make sure there is no such a key

        for task, env_kwargs in tasks_env_kwargs.items():
            self.samplers.update({
                task: SamplerCls({"env_kwargs": env_kwargs}.update(sampler_kwargs))
            })

    def initialize(self, **kwargs):
        '''
            param kwargs: all fed into each single sampler
        '''
        return dict([
            (task, self.samplers[task].initialize(**kwargs)) for task in self.tasks
        ])

    def obtain_samples(self, itr, tasks= None):
        '''
            param task: If provided, it will sample from given tasks
        '''
        return dict([
            (task, self.samplers[task].obtain_samples(itr))
            for task in (self.tasks if not task is None else tasks)
        ])

    def evaluate_agent(self, itr, tasks= None):
        return dict([
            (task, self.samplers[task].evaluate_agent(itr))
            for task in (self.tasks if not task is None else tasks)
        ])

    def shutdown(self):
        ''' Normally, there will be no return value from those samplers, so you don't
        really need to handle them.
        '''
        return dict([
            (task, self.samplers[task].shutdown()) for task in self.tasks
        ])

    @property
    def batch_size(self):
        s = 0
        for sampler in self.samplers.values():
            s += sampler.batch_size
        return s