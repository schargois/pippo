import torch
import torch.nn as nn
import Blocks as Blocks
from ProgNet import *
from Column_Generator import *
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces

class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        

        policy_column_generator = Column_generator_LSTM(
            input_size=feature_dim,
            hidden_size=128,
            num_of_classes=last_layer_dim_pi,
            num_LSTM_layer=2,
            num_dens_Layer=0,
            dropout=0.2,
        )
        value_column_generator = Column_generator_LSTM(
            input_size=feature_dim,
            hidden_size=128,
            num_of_classes=last_layer_dim_vf,
            num_LSTM_layer=2,
            num_dens_Layer=0,
            dropout=0.2,
        )

        # Policy network
        self.policy_net = ProgNet(
            policy_column_generator
        )
        # Value network
        self.value_net = ProgNet(
            value_column_generator
        )

    def add_policy_column(self, device: torch.device, column = None, last=False) -> None:
        """
        Add a new column to the network.
        :param device: device to use for the new column
        """
        self.policy_column = self.policy_net.addColumn(device, column)
        if not last:
            self.policy_net.freezeColumn(self.policy_column)
    
    def add_value_column(self, device: torch.device, column = None, last=False) -> None:
        """
        Add a new column to the network.
        :param device: device to use for the new column
        """
        self.value_column = self.value_net.addColumn(device, column)
        if not last:
            self.value_net.freezeColumn(self.value_column)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (torch.Tensor, torch.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(self.policy_column, features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(self.value_column, features)
    
class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        policy_columns: Optional[List[ProgColumn]] = None,
        value_columns: Optional[List[ProgColumn]] = None,
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        self.policy_columns = policy_columns
        self.value_columns = value_columns
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        


    def _build_mlp_extractor(self) -> None:
        model = CustomNetwork(self.features_dim)
        if self.policy_columns is not None:
            for column in self.policy_columns:
                model.add_policy_column(self.device, column)
        model.add_policy_column(self.device, last=True)
        if self.value_columns is not None:
            for column in self.value_columns:
                model.add_value_column(self.device, column)
        model.add_value_column(self.device, last=True)
        self.mlp_extractor = model