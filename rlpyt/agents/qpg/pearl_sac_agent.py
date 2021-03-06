
import numpy as np
import torch
from collections import namedtuple
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallelCPU as DDPC

from rlpyt.agents.base import AgentStep, BaseAgent
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.models.qpg.latent_mlp import ContextInferModel, LatentQofMuMlpModel, LatentPiMlpModel
from rlpyt.utils.quick_args import save__init__args
from rlpyt.distributions.gaussian import Gaussian, DistInfoStd
from rlpyt.utils.buffer import buffer_to
from exptools.logging import logger
from rlpyt.models.utils import update_state_dict
from rlpyt.samplers.collections import Context
from exptools.collections import namedarraytuple


MIN_LOG_STD = -20
MAX_LOG_STD = 2

AgentInfo = namedarraytuple("AgentInfo", ["dist_info"])
Models = namedtuple("Models", ["pi", "q1", "q2", "v"])


class PearlSacAgent(SacAgent):

    def __init__(
            self,
            EncoderCls=ContextInferModel,
            ModelCls=LatentPiMlpModel,  # Pi model.
            QModelCls=LatentQofMuMlpModel,
            latent_size=32,
            encoder_model_kwargs=None, # see __init__() for example
            model_kwargs=None,  # Pi model.
            q_model_kwargs=None, # see __init__() for example
            v_model_kwargs=None, # see __init__() for example
            initial_model_state_dict=None,  # All models.
            action_squash=1.,  # Max magnitude (or None).
            pretrain_std=0.75,  # With squash 0.75 is near uniform.
            ):
        """ NOTE: Currently only tested in len(env_space.observation.shape) == 1
        because of the encoder and need to concatenate encoder output and observation
        """
        if encoder_model_kwargs is None:
            # This surve as an example of making encoder_model_kwargs
            encoder_model_kwargs = dict(
                hidden_sizes=[200, 200, 200],
                use_information_bottleneck= True,
            )
        encoder_model_kwargs["output_size"] = latent_size
        save__init__args(locals())
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
        self.model_kwargs.update(dict(latent_size= latent_size))
        self.q_model_kwargs.update(dict(latent_size= latent_size))
        self.v_model_kwargs.update(dict(latent_size= latent_size)) # NOTE: This is model is not used in PEARL implementation

    def initialize(self, env_spaces, share_memory=False,
            global_B=1, env_ranks=None):
        """ In order to setup encoder model and add encoder.output_size to models
        input dimension, This method is complete rewritten
        """
        self.env_model_kwargs = self.make_env_to_model_kwargs(env_spaces)
        # 
        self.encoder_model = self.EncoderCls(**self.env_model_kwargs,
            **self.encoder_model_kwargs)
        # build regular model as sac_agent or base agent do
        self.model = self.ModelCls(**self.env_model_kwargs, **self.model_kwargs)
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
        self.z_distribution = Gaussian(
            dim= self.encoder_model.output_size,
            min_std=np.exp(MIN_LOG_STD),
            max_std=np.exp(MAX_LOG_STD),
        )
    
    def to_device(self, cuda_idx=None):
        super(PearlSacAgent, self).to_device(cuda_idx= cuda_idx)
        self.encoder_model.to(self.device)

    def q(self, observation, prev_action, prev_reward, action, latent_z=None):
        if latent_z is None:
            latent_z = self.zs
            # assuming latent_z is broadcastable for now
        model_inputs = buffer_to((observation, prev_action, prev_reward,
            action, latent_z), device=self.device)
        q1 = self.q1_model(*model_inputs)
        q2 = self.q2_model(*model_inputs)
        return q1.cpu(), q2.cpu()

    def target_q(self, observation, prev_action, prev_reward, action, latent_z=None):
        if latent_z is None:
            latent_z = self.zs
            # assuming latent_z is broadcastable for now
        model_inputs = buffer_to((observation, prev_action, prev_reward,
            action, latent_z), device=self.device)
        target_q1 = self.target_q1_model(*model_inputs)
        target_q2 = self.target_q2_model(*model_inputs)
        return target_q1.cpu(), target_q2.cpu()

    def pi(self, observation, prev_action, prev_reward, latent_z=None):
        if latent_z is None:
            latent_z = self.zs
            # assuming latent_z is broadcastable for now
        model_inputs = buffer_to((observation, prev_action, prev_reward, latent_z),
            device=self.device)
        mean, log_std = self.model(*model_inputs)
        dist_info = DistInfoStd(mean=mean, log_std=log_std)
        action, log_pi = self.distribution.sample_loglikelihood(dist_info)
        # action = self.distribution.sample(dist_info)
        # log_pi = self.distribution.log_likelihood(action, dist_info)
        log_pi, dist_info = buffer_to((log_pi, dist_info), device="cpu")
        return action, log_pi, dist_info  # Action stays on device for q models.

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        latent_z = self.zs
        # assuming latent_z is broadcastable for now
        model_inputs = buffer_to((observation, prev_action, prev_reward, latent_z),
            device=self.device)
        mean, log_std = self.model(*model_inputs)
        dist_info = DistInfoStd(mean=mean, log_std=log_std)
        action = self.distribution.sample(dist_info)
        agent_info = AgentInfo(dist_info=dist_info)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)
    
    def reset(self, batch_size= 1):
        """ Clear batch-wise latent zs value \\
            Write to `self.z_means` and `self.z_logstds`
        """
        self.z_means = torch.zeros(batch_size, self.latent_size)
        self.z_logstds = torch.ones(batch_size, self.latent_size)
        self.sample_zs()

    def infer_posterior(self, context: Context):
        """ NOTE: You should be sure that the batch-size should equal to
            action batch-size, which also means the number of tasks. \\
            Calculate batch-wise latent z distribution and sample it. \\
            Write to `self.z_means` and `self.z_vars` who has only leading dim (B,) instead of (T,B) \\
            And it will sample zs in batch and write to `self.zs`
        """
        # assuming arrays in context has leading dimension (T, B)
        context_inputs = buffer_to(context, device= self.device)
        self.z_means = self.encoder_model(*context_inputs)
        if self.encoder_model_kwargs["use_information_bottleneck"]:
            self.z_means, self.z_logstds = self.z_means
        self.sample_zs()

    def sample_zs(self):
        # NOTE: this self.zs only has leading dimension (B,) instead of (T, B)
        if self.encoder_model_kwargs["use_information_bottleneck"]:
            posteriors = [
                self.z_distribution.sample(DistInfoStd(mean=m, log_std=l)) \
                    for m, l in zip(torch.unbind(self.z_means), torch.unbind(self.z_logstds))
            ]
            self.zs = torch.stack(posteriors)
        else:
            self.zs = self.z_means
        return self.zs
    
    def compute_latent_KL(self):
        # Compute the kl divergence between latent z distribution
        z_means = self.z_means
        z_logstds = self.z_logstds
        z_zeros = torch.zeros_like(z_means)
        z_ones = torch.ones_like(z_logstds)
        # NOTE: consider kl divergence is not commutative, you might need to check this later
        kl_div = self.z_distribution.kl(
            old_dist_info = DistInfoStd(mean= z_zeros, log_std= z_ones),
            new_dist_info = DistInfoStd(mean= z_means, log_std= z_logstds)
        )
        return kl_div

    def detach_z(self):
        self.z = self.z.detach()

    def sample_mode(self, itr):
        # To prevent keep printing because the corespoinding sampler might switch
        # its mode over and over again.
        BaseAgent.sample_mode(self, itr)
        self.encoder_model.eval()
        self.q1_model.eval()
        self.q2_model.eval()
        # no printing logs here, which prints in sac_agent
        std = None if itr >= self.min_itr_learn else self.pretrain_std
        self.distribution.set_std(std)  # If None: std from policy dist_info.

    def train_mode(self, itr):
        super().train_mode(itr)
        self.encoder_model.train()

    def eval_mode(self, itr):
        super().eval_mode(itr)
        self.encoder_model.eval()

    def encoder_model_parameters(self):
        return self.encoder_model.parameters()

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
