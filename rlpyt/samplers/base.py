
from rlpyt.samplers.collectors import BaseCollector
from rlpyt.samplers.collections import BatchSpec, TrajInfo
from rlpyt.samplers.serial.collectors import SerialContextEvalCollector
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
    * This is a wrapper of multiple Sampler, using dictionary (NOTE: key is not necessarily string)
    * And evaluation will not be using instance of SamplerCls, but implmented with MultitaskSampler\
and its collector
    '''
    def __init__(self,
            SamplerCls,
            tasks_env_kwargs: dict,
            eval_tasks_env_kwargs: dict,
            eval_n_envs_per_task: int,   # How many envs will be evaluated in batch for each task.
            **common_sampler_kwargs, # The same as BaseSampler.__init__(...)
            ):
        '''
            param SamplerCls: the constructor to build sampler for a single task
            param tasks_env_kwargs: a dictionary with (task, env_kwargs) pairs
            param sampler_kwargs: the rest of kwargs are what you feed to a single sampler. \
                But recommending not env_kwargs item
        '''
        save__init__args(locals())
        self.tasks = list(tasks_env_kwargs.keys())
        self.samplers = dict()
        self.eval_tasks_collectors = dict()

        common_sampler_kwargs.pop("env_kwargs") # make sure there is no such a key
        common_sampler_kwargs.pop("eval_env_kwargs") # this key is subsituteed by "eval_tasks_env_kwargs"
        common_sampler_kwargs["eval_n_envs"] = 0 # not using nested sampler to evaluate
        for task, env_kwargs in tasks_env_kwargs.items():
            self.samplers.update({
                task: SamplerCls({"env_kwargs": env_kwargs}.update(common_sampler_kwargs))
            })

    def initialize(self, 
            agent,
            affinity=None,
            seed=None,
            bootstrap_value=False,
            traj_info_kwargs=None,
            rank=0,
            world_size=1):
        '''
            param kwargs: all fed into each single sampler
        '''
        B = self.batch_spec.B
        envs = [self.EnvCls(**self.env_kwargs) for _ in range(B)]
        global_B = B * world_size
        env_ranks = list(range(rank * B, (rank + 1) * B))
        agent.initialize(envs[0].spaces, share_memory=False,
            global_B=global_B, env_ranks=env_ranks)
        samples_pyt, samples_np, examples = build_samples_buffer(agent, envs[0],
            self.batch_spec, bootstrap_value, agent_shared=False,
            env_shared=False, subprocess=False)
        if traj_info_kwargs:
            for k, v in traj_info_kwargs.items():
                setattr(self.TrajInfoCls, "_" + k, v)  # Avoid passing at init.
        for task, self.tasks_env_kwargs
            collector = self.CollectorCls(
                rank=0,
                envs=envs,
                samples_np=samples_np,
                batch_T=self.batch_spec.T,
                TrajInfoCls=self.TrajInfoCls,
                agent=agent,
                global_B=global_B,
                env_ranks=env_ranks,  # Might get applied redundantly to agent.
            )

        # We build eval envs for eval tasks
        if self.eval_n_envs_per_task > 0 and len(self.eval_tasks_env_kwargs) > 0:
            for task, eval_task_env_kwargs in self.eval_tasks_env_kwargs.items():
                eval_envs = [self.EnvCls(**eval_task_env_kwargs) for _ in range(self.eval_n_envs_per_task)]
                self.eval_tasks_collectors[task] = SerialContextEvalCollector(
                    envs=eval_envs,
                    agent=agent,
                    TrajInfoCls=self.TrajInfoCls,
                    infer_posterior_period= 
                    max_T=self.eval_max_steps // self.eval_n_envs,
                    max_trajectories=self.eval_max_trajectories,
                )

        return results

    def obtain_samples(self, itr, tasks= None):
        '''
            param task: If provided, it will sample from given tasks
        '''
        return dict([
            (task, self.samplers[task].obtain_samples(itr))
            for task in (self.tasks if not task is None else tasks)
        ])

    def evaluate_agent(self, itr, tasks= None):
        if tasks is None:
            tasks = self.eval_tasks_env_kwargs.keys()
        return dict([
            (task, self.eval_tasks_collectors[task].collect_evaluation(itr))
            for task in tasks
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