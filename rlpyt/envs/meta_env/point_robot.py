''' The code is copied and modified from oyster: https://github.com/katerakelly/oyster/blob/master/rlkit/envs/point_robot.py
    at 2020/02/09 by Ziwen Zhuang
'''
import numpy as np
from rlpyt.spaces.float_box import FloatBox
from rlpyt.envs.base import EnvStep, EnvSpaces
from rlpyt.utils.collections import namedarraytuple
from gym import Env
from .base import MultitaskEnv

EnvInfo = namedarraytuple("EnvInfo", ["sparse_reward"])

class PointEnv(Env, MultitaskEnv):
    """
    point robot on a 2-D plane with position control
    each task is represented by a integer (idx)
    tasks (aka goals) are positions on the plane
     - tasks sampled from unit square
     - reward is L2 distance
     - done_threshold is the radius when the state is in the goal radius
    """

    def __init__(self, randomize_tasks=False, n_tasks=2, done_threshold= 0.2):

        if randomize_tasks:
            np.random.seed(1337)
            goals = [[np.random.uniform(-1., 1.), np.random.uniform(-1., 1.)] for _ in range(n_tasks)]
        else:
            # some hand-coded goals for debugging
            goals = [np.array([10, -10]),
                     np.array([10, 10]),
                     np.array([-10, 10]),
                     np.array([-10, -10]),
                     np.array([0, 0]),

                     np.array([7, 2]),
                     np.array([0, 4]),
                     np.array([-6, 9])
                     ]
            goals = [g / 10. for g in goals]
        self.goals = goals
        self.done_threshold = done_threshold

        self.reset_task(0)
        self.observation_space = FloatBox(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = FloatBox(low=-0.1, high=0.1, shape=(2,))

    def reset_task(self, idx):
        ''' reset goal AND reset the agent '''
        self._goal = self.goals[idx]
        self._goal_idx = idx
        self.reset()
    def set_task(self, idx):
        return self.reset_task(idx)
    def get_task(self):
        return self._goal_idx
    def sample_tasks(self, n_tasks):
        return list(np.random.choice(len(self.goals), n_tasks))

    def get_all_task_idx(self):
        return range(len(self.goals))

    def reset_model(self):
        # reset to a random location on the unit square
        self._state = np.random.uniform(-1., 1., size=(2,))
        return self._get_obs()

    def reset(self):
        return self.reset_model()

    def _get_obs(self):
        return np.copy(self._state).astype(np.float32)

    def step(self, action):
        self._state = self._state + action
        x, y = self._state
        x -= self._goal[0]
        y -= self._goal[1]
        reward = - (x ** 2 + y ** 2) ** 0.5
        done = 1 if reward < self.done_threshold else 0
        ob = self._get_obs()
        return EnvStep(ob, reward, done, EnvInfo(np.nan))

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('current state:', self._state)

class SparsePointEnv(PointEnv):
    '''
     - tasks sampled from unit half-circle
     - reward is L2 distance given only within goal radius
     NOTE that `step()` returns the dense reward because this is used during meta-training
     the algorithm should call `sparsify_rewards()` to get the sparse rewards
     '''
    def __init__(self, randomize_tasks=False, n_tasks=2, goal_radius=0.2):
        super().__init__(randomize_tasks, n_tasks)
        self.goal_radius = goal_radius

        if randomize_tasks:
            np.random.seed(1337)
            radius = 1.0
            angles = np.linspace(0, np.pi, num=n_tasks)
            xs = radius * np.cos(angles)
            ys = radius * np.sin(angles)
            goals = np.stack([xs, ys], axis=1)
            np.random.shuffle(goals)
            goals = goals.tolist()

        self.goals = goals
        self.reset_task(0)

    def sparsify_rewards(self, r):
        ''' zero out rewards when outside the goal radius '''
        mask = (r >= -self.goal_radius).astype(np.float32)
        r = r * mask
        return r

    def reset_model(self):
        self._state = np.array([0, 0])
        return self._get_obs()

    def step(self, action):
        ob, reward, done, d = super().step(action)
        sparse_reward = self.sparsify_rewards(reward)
        # make sparse rewards positive
        if reward >= -self.goal_radius:
            sparse_reward += 1
        return EnvStep(ob, reward, done, EnvInfo(sparse_reward))
