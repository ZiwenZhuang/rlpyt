
from rlpyt.samplers.collectors import BaseCollector
from rlpyt.samplers.collections import BatchSpec, TrajInfo
from rlpyt.samplers.serial.collectors import SerialContextEvalCollector
from rlpyt.samplers.parallel.cpu.collectors import CpuContextCollector
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

class MultitaskBaseSampler(BaseSampler):
    ''' A example for implementing meta-RL sampler.
    * Due to request from collector `CpuContextCollector`, `infer_context_period` should \
be provided by sampler, This sampler is completely re-written.
    * What is different from original sampler:
        All data (samples_pyt, samples_np, samples..) are added "tasks_" as prefix.
        All returned value from common interface are a dictionary with a common returned value.
    '''
    def __init__(self,
            EnvCls,
            batch_T, # parameter for training collector for each task
            batch_B, # parameter for training collector for each task
            CollectorCls= CpuContextCollector,
            TrajInfoCls=TrajInfo,
            infer_context_period=np.inf,
            tasks_env_kwargs: dict,
            eval_n_envs_per_task: int=0, # How many envs will be evaluated in batch for each task.
            eval_CollectorCls=None, # Must supply if doing eval.
            eval_tasks_env_kwargs: dict=None, # a dictionary of dictionarys.
            eval_max_steps=None,  # int if using evaluation.
            eval_max_trajectories=None,  # Optional earlier cutoff.
            ):
        '''
            param tasks_env_kwargs: a dictionary with (task, env_kwargs) pairs
            param eval_tasks_env_kwargs: a dictionary with (task, env_kwargs) pairs
        '''
        eval_max_steps = None if eval_max_steps is None else int(eval_max_steps)
        eval_max_trajectories = (None if eval_max_trajectories is None else
            int(eval_max_trajectories))
        save__init__args(locals())
        self.mid_batch_reset = CollectorCls.mid_batch_reset
        self.batch_spec = BatchSpec(batch_T, batch_B)
        self.tasks = list(tasks_env_kwargs.keys())
        self.tasks_collectors = dict()
        self.eval_tasks = list(eval_tasks_env_kwargs.keys())
        self.eval_tasks_collectors = dict()

    @property
    def batch_size(self):
        return len(self.tasks) * self.batch_spec.size