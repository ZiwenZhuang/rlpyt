
import numpy as np
import torch
from collections import namedtuple
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallelCPU as DDPC

from rlpyt.agents.base import AgentStep
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.models.mlp import FlattenMlpModel
from rlpyt.models.qpg.mlp import QofMuMlpModel, PiMlpModel
from rlpyt.utils.quick_args import save__init__args
from rlpyt.distributions.gaussian import Gaussian, DistInfoStd
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.logging import logger
from rlpyt.models.utils import update_state_dict
from rlpyt.utils.collections import namedarraytuple


MIN_LOG_STD = -20
MAX_LOG_STD = 2

AgentInfo = namedarraytuple("AgentInfo", ["dist_info"])
Models = namedtuple("Models", ["pi", "q1", "q2", "v"])


class PearlSacAgent(SacAgent):

    def __init__(
            self,
            EncoderCls=FlattenMlpModel,
            ModelCls=PiMlpModel,  # Pi model.
            QModelCls=QofMuMlpModel,
            encoder_model_kwargs=None,
            model_kwargs=None,  # Pi model.
            q_model_kwargs=None,
            v_model_kwargs=None,
            initial_model_state_dict=None,  # All models.
            action_squash=1.,  # Max magnitude (or None).
            pretrain_std=0.75,  # With squash 0.75 is near uniform.
            ):
        """ NOTE: Currently only tested in len(env_space.observation.shape) == 1
        because of the encoder and need to concatenate encoder output and observation
        """
        if encoder_model_kwargs is None:
            encoder_model_kwargs = dict(
                hidden_sizes=[200, 200, 200],
                output_size=1024,
            )
        super().__init__(
            ModelCls=ModelCls, 
            QModelCls=QModelCls,
            model_kwargs=model_kwargs,
            q_model_kwargs= q_model_kwargs,
            v_model_kwargs=v_model_kwargs,
            initial_model_state_dict=initial_model_state_dict,
            action_squash=action_squash,
            pretrain_std=pretrain_std,
        )
        save__init__args(locals())

    def initialize(self, env_spaces, share_memory=False,
            global_B=1, env_ranks=None):
        """ In order to setup encoder model and add encoder.output_size to models
        input dimension, This method is complete rewritten
        """
        self.env_model_kwargs = self.make_env_to_model_kwargs(env_spaces)
        # 
        self.encoder_model = self.EncoderCls(**self.env_model_kwargs,
            **self.encoder_model_kwargs)
        # modify the parameters so that code can be directly reused
        observation_shape = self.env_model_kwargs["observation_shape"]
        self.env_model_kwargs["observation_shape"] = [
            np.prod(observation_shape) + self.encoder_model.otuput_size
        ]
        # build regular model as sac_agent or base agent do
        self.model = self.ModelCls(**self.env_model_kwargs,
            **self.model_kwargs)
        self.q1_model = self.QModelCls(**self.env_model_kwargs, **self.q_model_kwargs)
        self.q2_model = self.QModelCls(**self.env_model_kwargs, **self.q_model_kwargs)
        self.target_q1_model = self.QModelCls(**self.env_model_kwargs,
            **self.q_model_kwargs)
        self.target_q2_model = self.QModelCls(**self.env_model_kwargs,
            **self.q_model_kwargs)
        self.target_q1_model.load_state_dict(self.q1_model.state_dict())
        self.target_q2_model.load_state_dict(self.q2_model.state_dict())
        # share memory if needed
        if share_memory:
            self.encoder_model.share_memory()
            self.model.share_memory()
            self.shared_model = self.model
        self.env_spaces = env_spaces
        self.share_memory = share_memory
        # load state_dict if needed
        if self.initial_model_state_dict is not None:
            self.load_state_dict(self.initial_model_state_dict)
        # and some distribution stuff
        assert len(env_spaces.action.shape) == 1
        self.distribution = Gaussian(
            dim=env_spaces.action.shape[0],
            squash=self.action_squash,
            min_std=np.exp(MIN_LOG_STD),
            max_std=np.exp(MAX_LOG_STD),
        )

    def q(self, observation, prev_action, prev_reward, action):
        # model_inputs = buffer_to((observation, prev_action, prev_reward,
        #     action), device=self.device)
        # q1 = self.q1_model(*model_inputs)
        # q2 = self.q2_model(*model_inputs)
        # return q1.cpu(), q2.cpu()

    def target_q(self, observation, prev_action, prev_reward, action):
        # model_inputs = buffer_to((observation, prev_action, prev_reward,
        #     action), device=self.device)
        # target_q1 = self.target_q1_model(*model_inputs)
        # target_q2 = self.target_q2_model(*model_inputs)
        # return target_q1.cpu(), target_q2.cpu()

    def pi(self, observation, prev_action, prev_reward):
        # model_inputs = buffer_to((observation, prev_action, prev_reward),
        #     device=self.device)
        # mean, log_std = self.model(*model_inputs)
        # dist_info = DistInfoStd(mean=mean, log_std=log_std)
        # action, log_pi = self.distribution.sample_loglikelihood(dist_info)
        # # action = self.distribution.sample(dist_info)
        # # log_pi = self.distribution.log_likelihood(action, dist_info)
        # log_pi, dist_info = buffer_to((log_pi, dist_info), device="cpu")
        # return action, log_pi, dist_info  # Action stays on device for q models.

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        # model_inputs = buffer_to((observation, prev_action, prev_reward),
        #     device=self.device)
        # mean, log_std = self.model(*model_inputs)
        # dist_info = DistInfoStd(mean=mean, log_std=log_std)
        # action = self.distribution.sample(dist_info)
        # agent_info = AgentInfo(dist_info=dist_info)
        # action, agent_info = buffer_to((action, agent_info), device="cpu")(action=a
        # from rlpyt.agents.qpg.sac_agent import SacAgentction, agent_info=agent_info)

    def reset(self):

    def infer_posterior(self, context):

    def sample_mode(self, itr):
        self.encoder_model.eval()
        super().sample_mode(itr)

    def train_mode(self, itr):
        super().train_mode(itr)
        self.encoder_model.train()

    def eval_mode(self, itr):
        super().eval_mode(itr)
        self.encoder_model.eval()

    def state_dict(self):
        return dict(
            model=self.model.state_dict(),  # Pi model.
            encoder_model=self.encoder_model.state_dict(),
            q1_model=self.q1_model.state_dict(),
            q2_model=self.q2_model.state_dict(),
            target_q1_model=self.target_q1_model.state_dict(),
            target_q2_model=self.target_q2_model.state_dict(),
        )

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.encoder_model.load_state_dict(state_dict["encoder_model"])
        self.q1_model.load_state_dict(state_dict["q1_model"])
        self.q2_model.load_state_dict(state_dict["q2_model"])
        self.target_q1_model.load_state_dict(state_dict["target_q1_model"])
        self.target_q2_model.load_state_dict(state_dict["target_q2_model"])
