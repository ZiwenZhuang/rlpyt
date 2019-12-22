
import numpy as np

from rlpyt.replays.n_step import BaseNStepReturnBuffer
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.buffer import torchify_buffer

SamplesFromReplay = namedarraytuple("SamplesFromReplay",
    ["agent_inputs", "action", "return_", "done", "done_n", "target_inputs"])


class NStepReturnBuffer(BaseNStepReturnBuffer):

    def extract_batch(self, T_idxs, B_idxs):
        s = self.samples
        target_T_idxs = (T_idxs + self.n_step_return) % self.T
        batch = SamplesFromReplay(
            agent_inputs=AgentInputs(
                observation=self.extract_observation(T_idxs, B_idxs),
                prev_action=s.action[T_idxs - 1, B_idxs],
                prev_reward=s.reward[T_idxs - 1, B_idxs],
            ),
            action=s.action[T_idxs, B_idxs],
            return_=self.samples_return_[T_idxs, B_idxs],
            done=self.samples.done[T_idxs, B_idxs],
            done_n=self.samples_done_n[T_idxs, B_idxs],
            target_inputs=AgentInputs(
                observation=self.extract_observation(target_T_idxs, B_idxs),
                prev_action=s.action[target_T_idxs - 1, B_idxs],
                prev_reward=s.reward[target_T_idxs - 1, B_idxs],
            ),
        )
        # target_... means what happend after self.n_step_return timestep
        # It serve as target for predicting the n-step return.
        t_news = np.where(s.done[T_idxs - 1, B_idxs])[0]
        batch.agent_inputs.prev_action[t_news] = 0
        batch.agent_inputs.prev_reward[t_news] = 0
        return torchify_buffer(batch)

    def extract_observation(self, T_idxs, B_idxs):
        """Generalization anticipating frame-based buffer."""
        return self.samples.observation[T_idxs, B_idxs]
