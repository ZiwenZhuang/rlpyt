""" implmenting with adding a latent variable as input (called z)
    Different from mlp.py in the same level, latent_z is assumed with
leading dim (B,) instead of (T,B)
"""
import numpy as np
import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel

from rlpyt.distributions.gaussian import product_of_gaussian

class ContextInferModel(torch.nn.Module):
    def __init__(
            self, 
            observation_shape,
            hidden_sizes,
            action_size,
            output_size, # latent variable
            use_information_bottleneck= False
            ):
        """ NOTE: if not use_information_bottleneck, the return of this model
        will be only one array.
        """
        super().__init__()
        self._use_information_bottleneck = use_information_bottleneck
        self.output_size = output_size
        self._obs_ndim = len(observation_shape)
        self.mlp = MlpModel(
            input_size=int(np.prod(observation_shape)*2 + action_size + 1 + 1),
            hidden_sizes=hidden_sizes,
            output_size=output_size * (2 if use_information_bottleneck else 1)
        )

    def forward(self, observation, action, reward, next_observation, done):
        """ NOTE: the returned tensor will have no T dimension
        Defined input sequences according to context. \\
        Maybe see rlpyt.samplers.parallel.cpu.collectors.Context
        """
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        context_input = torch.cat([
            observation.view(T * B, -1),
            action.view(T * B, -1),
            reward.view(T * B, -1),
            next_observation.view(T * B, -1),
            done.view(T * B, -1)
        ], dim=1)
        params = self.mlp(context_input)
        params = restore_leading_dims(params, lead_dim, T, B)
        if self._use_information_bottleneck:
            mus = params[..., :self.action_size]
            logstds = params[..., self.action_size:]
            z_params = [
                product_of_gaussian(m, l) for m, l in \
                    zip(torch.unbind(mus, dim=1), torch.unbind(logstds, dim=1))
                # The batch dimension is dim=1 not dim=0
            ]
            z_means = torch.stack([p[0] for p in z_params])
            z_logstds = torch.stack([p[1] for p in z_params])
        else:
            return torch.mean(params, dim=0) # along time dimension

class LatentMuMlpModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            hidden_sizes,
            action_size,
            latent_size,
            output_max=1,
            ):
        super().__init__()
        self._output_max = output_max
        self._obs_ndim = len(observation_shape)
        self.mlp = MlpModel(
            input_size=int(np.prod(observation_shape) + latent_size),
            hidden_sizes=hidden_sizes,
            output_size=action_size,
        )

    def forward(self, observation, prev_action, prev_reward, latent_z):
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        assert B == latent_z.shape[0], "Please check the batch_size of the latent space of the agent"
        mu_intput = torch.cat([
            observation.view(T * B, -1),
            torch.stack(T * [latent_z.view(B, -1)], dim=0)
        ], dim=1)
        mu = self._output_max * torch.tanh(self.mlp(mu_input))
        mu = restore_leading_dims(mu, lead_dim, T, B)
        return mu


class LatentPiMlpModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            hidden_sizes,
            action_size,
            latent_size,
            ):
        super().__init__()
        self._obs_ndim = len(observation_shape)
        self._action_size = action_size
        self.mlp = MlpModel(
            input_size=int(np.prod(observation_shape) + latent_size),
            hidden_sizes=hidden_sizes,
            output_size=action_size * 2,
        )

    def forward(self, observation, prev_action, prev_reward, latent_z):
        lead_dim, T, B, _ = infer_leading_dims(observation,
            self._obs_ndim)
        assert B == latent_z.shape[0], "Please check the batch_size of the latent space of the agent"
        pi_input = torch.cat([
            observation.view(T * B, -1),
            torch.stack(T * [latent_z.view(B, -1)], dim=0)
        ], dim=1)
        output = self.mlp(pi_input)
        mu, log_std = output[:, :self._action_size], output[:, self._action_size:]
        mu, log_std = restore_leading_dims((mu, log_std), lead_dim, T, B)
        return mu, log_std


class LatentQofMuMlpModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            hidden_sizes,
            action_size,
            latent_size,
            ):
        super().__init__()
        self._obs_ndim = len(observation_shape)
        self.mlp = MlpModel(
            input_size=int(np.prod(observation_shape)) + action_size + latent_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
        )

    def forward(self, observation, prev_action, prev_reward, action, latent_z):
        lead_dim, T, B, _ = infer_leading_dims(observation,
            self._obs_ndim)
        assert B == latent_z.shape[0], "Please check the batch_size of the latent space of the agent"
        q_input = torch.cat([
            observation.view(T * B, -1),
            action.view(T * B, -1),
            torch.stack(T * [latent_z.view(B, -1)], dim=0)
        ], dim=1)
        q = self.mlp(q_input).squeeze(-1)
        q = restore_leading_dims(q, lead_dim, T, B)
        return q


class LatentVMlpModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            hidden_sizes,
            action_size=None,  # Unused but accept kwarg.
            latent_size=None,   # You have to provide it, put at last is preventing *args
            ):
        assert latent_size is not None
        super().__init__()
        self._obs_ndim = len(observation_shape)
        self.mlp = MlpModel(
            input_size=int(np.prod(observation_shape) + latent_size),
            hidden_sizes=hidden_sizes,
            output_size=1,
        )

    def forward(self, observation, prev_action, prev_reward, latent_z):
        lead_dim, T, B, _ = infer_leading_dims(observation,
            self._obs_ndim)
        assert B == latent_z.shape[0], "Please check the batch_size of the latent space of the agent"
        v_input = torch.cat([
            observation.view(T * B, -1),
            torch.stack(T * [latent_z.view(B, -1)], dim=0)
        ], dim=1)
        v = self.mlp(v_input).squeeze(-1)
        v = restore_leading_dims(v, lead_dim, T, B)
        return v
