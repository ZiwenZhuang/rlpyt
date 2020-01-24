''' A rlpyt buffer for multi-task replay buffer
'''
from rlpyt.utils.quick_args import save__init__args
from rlpyt.replays.base import BaseReplayBuffer

import numpy as np

class MultitaskReplayBuffer(BaseReplayBuffer):
    ''' This is just a wrapper managing a list of replay bufffer for each task
    '''
    def __init__(self,
            SingleReplayBufferCls,
            tasks_example,  # A list of tasks that serve as key for each single replay buffer
            **buffer_kwargs, # The argument feed directly to a single replay buffer
            ):
        save__init__args(locals())
        if "example" in buffer_kwargs.keys():
            buffer_kwargs.pop("example") # make sure they have no such an argument
        self.tasks_replay_buffers = [
            SingleReplayBufferCls(example= example, **buffer_kwargs)
            for example in tasks_example
        ]

    def append_samples(self, tasks_samples):
        ''' Append sample in terms of a given task, all train tasks samples should be given.
            Assuming there is no return value
        '''
        for idx, samples in enumerate(tasks_samples):
            self.tasks_replay_buffers[idx].append_samples(samples)

    def sample_batch(self, batch_B, batch_T= None):
        ''' Sample a batch of trajectories based on current stored tasks samples.
        '''
        return [
            replay_buffer.sample_batch(batch_B) for replay_buffer in self.tasks_replay_buffers
        ]

class PearlReplayBuffer(MultitaskReplayBuffer):
    ''' This is a small variant of PEARL as it needed
    '''
    def append_samples(self, tasks_examples, update_samples_ratio):
        """
        Args:
            update_samples_ratio: As PEARL source code shows, it only updates certain number of tasks samples.
                For convience, this implementation adopts ratio to update training task samples, but you still
                need to provide all tasks samples.
        """
        num_update_task_samples = int(len(self.tasks_replay_buffers) * update_samples_ratio)
        indices = np.random.choice(len(self.tasks_replay_buffers), num_update_task_samples, replace= False)
        for idx in indices:
            self.tasks_replay_buffers[idx].append_samples(tasks_examples[idx])
