import numpy as np
import torch
import torch.nn as nn
import Blocks as Blocks
from ProgNet import *
from Column_Generator import *
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces
from torch.distributions.normal import Normal
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from functools import partial

from custom_dist import (
  CustomDistribution,
  make_proba_distribution,
)

from stable_baselines3.common.distributions import Distribution

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
        last_layer_dim_pi: int = 128,
        last_layer_dim_vf: int = 128,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.output_index_hardcode = -1

        print(f"feature_dim: {feature_dim}")
        print(f"last_layer_dim_pi: {last_layer_dim_pi}")
        print(f"last_layer_dim_vf: {last_layer_dim_vf}")

        policy_column_generator = Column_generator_LSTM(
            input_size=feature_dim,
            hidden_size=128,
            num_of_classes=last_layer_dim_pi,
            num_LSTM_layer=2,
            num_dens_Layer=0,
            dropout=0.0,
        )
        value_column_generator = Column_generator_LSTM(
            input_size=feature_dim,
            hidden_size=128,
            num_of_classes=last_layer_dim_vf,
            num_LSTM_layer=2,
            num_dens_Layer=0,
            dropout=0.0,
        )

        # Policy network
        self.policy_net = ProgNet(policy_column_generator)
        # Value network
        self.value_net = ProgNet(value_column_generator)

    def add_policy_column(self, device: torch.device, column=None, last=False) -> None:
        """
        Add a new column to the network.
        :param device: device to use for the new column
        """
        self.policy_column = self.policy_net.addColumn(device, column)
        return self.policy_column

    def add_value_column(self, device: torch.device, column=None, last=False) -> None:
        """
        Add a new column to the network.
        :param device: device to use for the new column
        """
        self.value_column = self.value_net.addColumn(device, column)
        return self.value_column

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (torch.Tensor, torch.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        if self.output_index_hardcode != -1:
            return self.policy_net(self.output_index_hardcode, features)
        return self.policy_net(self.policy_column, features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        if self.output_index_hardcode != -1:
            # Hardcode the output index to the last layer of the value network
            # print(f"hardcode the output index to {self.output_index_hardcode}")
            return self.value_net(self.output_index_hardcode, features)
        return self.value_net(self.value_column, features)
        # if self.output_index_hardcode != -1:
        #     return self.policy_net(self.output_index_hardcode, features)
        # return self.policy_net(self.policy_column, features)

class ProgNetWrapper(ProgNet):
    def __init__(self, colGen=None):
        super().__init__(colGen)
        self.output_index_hardcode = -1
    
    def forward(self, x):
        if self.output_index_hardcode != -1:
            return super().forward(self.output_index_hardcode, x)
        return super().forward(self.numCols-1, x)
    

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        mlp_policy_columns: Optional[List[ProgColumn]] = None,
        mlp_value_columns: Optional[List[ProgColumn]] = None,
        action_net_columns: Optional[List[ProgColumn]] = None,
        value_net_columns: Optional[List[ProgColumn]] = None,
        new_column: bool = True,
        output_index_hardcode: int = -1,
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        self.mlp_policy_columns = mlp_policy_columns
        self.mlp_value_columns = mlp_value_columns
        self.action_net_columns = action_net_columns
        self.value_net_columns = value_net_columns
        self.new_column = new_column
        self.output_index_hardcode = output_index_hardcode
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )


    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        self.action_dist = make_proba_distribution(self.action_space)
        self._build_mlp_extractor()

        feature_dim = self.mlp_extractor.latent_dim_pi
        output_dim = self.action_space.shape[0]
        action_column_generator = Column_generator_LSTM(
            input_size=feature_dim,
            hidden_size=128,
            num_of_classes=output_dim,
            num_LSTM_layer=2,
            num_dens_Layer=0,
            dropout=0.0,
        )

        self.action_dist.mean_actions = ProgNetWrapper(action_column_generator)
        if self.action_net_columns is not None:
            for col_dict in self.action_net_columns:
                id = self.action_dist.mean_actions.addColumn(self.device)
                column = self.action_dist.mean_actions.getColumn(id)
                column.load_state_dict(col_dict)
                column.freeze()
        self.action_dist.mean_actions.addColumn(self.device)

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        self.action_net, self.log_std = self.action_dist.proba_distribution_net(
            latent_dim=latent_dim_pi, log_std_init=self.log_std_init
        )

        value_column_generator = Column_generator_LSTM(
            input_size=feature_dim,
            hidden_size=128,
            num_of_classes=1,
            num_LSTM_layer=2,
            num_dens_Layer=0,
            dropout=0.0,
        )

        self.value_net = ProgNetWrapper(value_column_generator)
        if self.value_net_columns is not None:
            for col_dict in self.value_net_columns:
                id = self.value_net.addColumn(self.device)
                column = self.value_net.getColumn(id)
                column.load_state_dict(col_dict)
                column.freeze()
        self.value_net.addColumn(self.device)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)

        return self.action_dist.proba_distribution(mean_actions, self.log_std)
    
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)
        if self.mlp_policy_columns is not None:
            for col_dict in self.mlp_policy_columns:
                id = self.mlp_extractor.add_policy_column(self.device)
                column = self.mlp_extractor.policy_net.getColumn(id)
                column.load_state_dict(col_dict)
                column.freeze()
        if self.new_column:
            self.mlp_extractor.add_policy_column(self.device)
        if self.mlp_value_columns is not None:
            for col_dict in self.mlp_value_columns:
                id = self.mlp_extractor.add_value_column(self.device)
                column = self.mlp_extractor.value_net.getColumn(id)
                column.load_state_dict(col_dict)
                column.freeze()
        if self.new_column:
            self.mlp_extractor.add_value_column(self.device)
        if self.output_index_hardcode != -1:
            self.mlp_extractor.output_index_hardcode = self.output_index_hardcode
