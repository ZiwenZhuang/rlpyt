''' A rlpyt buffer for multi-task replay buffer
'''
from rlpyt.utils.quick_args import save__init__args
from rlpyt.replays.base import BaseReplayBuffer

class MultitaskReplayBuffer(BaseReplayBuffer):
    ''' This is just a wrapper managing a list of replay bufffer for each task
    '''
    def __init__(self,
            SingleReplayBufferCls,
            tasks_example,  # A list of tasks that serve as key for each single replay buffer
            **buffer_kwargs, # The argument feed directly to a single replay buffer
            ):
        save__init__args(locals())
        buffer_kwargs.pop("example") # make sure they have no such a argument
        self.replay_buffers = dict([
            (task, SingleReplayBufferCls(example= tasks_example[task], **buffer_kwargs))
            for task in tasks_example.keys()
        ])

    def append_samples(self, tasks_samples):
        ''' Append sample in terms of a given task
        '''
        return dict([
            (task, self.replay_buffers[task].append_samples(samples))
            for task, samples in tasks_samples.items()
        ])

    def sample_batch(self, tasks, batch_B, batch_T= None):
        ''' Sample a batch of trajectories based on given task
        '''
        return dict([
            (task, self.replay_buffers[task].sample_batch(batch_B, batch_T))
            for task in tasks
        ])
