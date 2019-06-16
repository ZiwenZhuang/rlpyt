
import torch
import torch.nn.functional as F

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.conv2d import Conv2dModel
from rlpyt.models.mlp import MlpModel
from rlpyt.models.dqn.dueling import DistributionalDuelingHeadModel


class DistributionalHeadModel(torch.nn.Module):

    def __init__(self, input_size, layer_sizes, output_size, n_atoms):
        super().__init__()
        self.mlp = MlpModel(input_size, layer_sizes, output_size * n_atoms)
        self._output_size = output_size
        self._n_atoms = n_atoms

    def forward(self, input):
        return self.mlp(input).view(-1, self._output_size, self._n_atoms)


class AtariCatDqnModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            output_size,
            n_atoms=51,
            fc_sizes=512,
            dueling=False,
            use_maxpool=False,
            channels=None,  # None uses default.
            kernel_sizes=None,
            strides=None,
            paddings=None,
            ):
        super().__init__()
        self.dueling = dueling
        c, h, w = image_shape
        self.conv = Conv2dModel(
            in_channels=c,
            channels=channels or [32, 64, 64],
            kernel_sizes=kernel_sizes or [8, 4, 3],
            strides=strides or [4, 2, 1],
            paddings=paddings or [0, 1, 1],
            use_maxpool=use_maxpool,
        )
        conv_out_size = self.conv.conv_out_size(h, w)
        if dueling:
            self.head = DistributionalDuelingHeadModel(conv_out_size, fc_sizes,
                output_size=output_size, n_atoms=n_atoms)
        else:
            self.head = DistributionalHeadModel(conv_out_size, fc_sizes,
                output_size=output_size, n_atoms=n_atoms)

    def forward(self, observation, prev_action, prev_reward):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""
        img = observation.type(torch.float)  # Expect torch.uint8 inputs
        img = img.mul_(1. / 255)  # From [0-255] to [0-1], in place.

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        img_shape, T, B, has_T, has_B = infer_leading_dims(img, 3)

        conv_out = self.conv(img.view(T * B, *img_shape))  # Fold if T dimension.
        p = self.head(conv_out.view(T * B, -1))
        p = F.softmax(p, dim=-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        p = restore_leading_dims(p, T, B, has_T, has_B)
        return p
