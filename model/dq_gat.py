import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from model.agent_state import VocabularyStateType
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from gymnasium import spaces
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ResNetEncoder(nn.Module):
    """CNN-based feature extractor using ResNet-18"""
    def __init__(self, output_dim=512):
        super(ResNetEncoder, self).__init__()
        resnet = resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1]) 
        self.fc = nn.Linear(resnet.fc.in_features, output_dim)  

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x 

class MLPNodeEncoder(nn.Module):
    """Encodes the node motion states into latent vector"""
    def __init__(self, input_dim=10, hidden_dim=128, output_dim=128):
        super(MLPNodeEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.mlp(x)  

class MultiHeadSelfAttentionGAT(nn.Module):
    """Implements GAT using MultiHead Self-Attention"""
    def __init__(self, input_dim=640, hidden_dim=256, output_dim=256, num_heads=4):
        super(MultiHeadSelfAttentionGAT, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        self.W_q = nn.Linear(input_dim, hidden_dim * num_heads, bias=False)
        self.W_k = nn.Linear(input_dim, hidden_dim * num_heads, bias=False)
        self.W_v = nn.Linear(input_dim, hidden_dim * num_heads, bias=False)
        self.W_o = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        """
        Inputs:
            - x: (batch_size, num_agents, input_dim)
        Outputs:
            - (batch_size, num_agents, output_dim)
        """
        batch_size, num_agents, _ = x.shape

        Q = self.W_q(x).view(batch_size, num_agents, self.num_heads, self.hidden_dim)
        K = self.W_k(x).view(batch_size, num_agents, self.num_heads, self.hidden_dim)
        V = self.W_v(x).view(batch_size, num_agents, self.num_heads, self.hidden_dim)

        attention_scores = torch.einsum("bqhd,bkhd->bhqk", Q, K) / (self.hidden_dim ** 0.5) 
        attention_weights = F.softmax(attention_scores, dim=-1)  
        attention_output = torch.einsum("bhqk,bkhd->bqhd", attention_weights, V) 
        x_out = attention_output.mean(dim=2) 
        x_out = self.W_o(x_out)  

        return x_out
    
class DQGAT(nn.Module):
    def __init__(self, bev_output_dim=512, node_embed_dim=128):
        super(DQGAT, self).__init__()
        self.cnn_encoder = ResNetEncoder(output_dim=bev_output_dim) 
        # self.mlp_encoder = MLPNodeEncoder(input_dim=node_embed_dim, output_dim=node_embed_dim)  
        self.mlp_encoder = MLPNodeEncoder(input_dim=12, output_dim=node_embed_dim)
        self.gat_encoder = nn.Sequential(
            MultiHeadSelfAttentionGAT(input_dim=bev_output_dim + node_embed_dim, output_dim=512),
            MultiHeadSelfAttentionGAT(input_dim=512, output_dim=256),
        )
        # self.token_embedding = nn.Embedding(VocabularyStateType.PAD_TOKEN.vocal_size, node_embed_dim)

    def forward(self, bev_image, agent_feats):
        """
        Inputs:
            - bev_image: (batch_size, 3, 224, 224)      [Semantic BEV Image]
            - agent_feats: (batch_size, num_agents, 10) [Motion State Features]
        Outputs:
            - ego_node_feature: (batch_size, 256)       [Final feature representing ego-vehicle interactions]
        """
        batch_size, num_agents, _ = agent_feats.shape
        
        # Extract the BEV and agent features
        bev_embedding = self.cnn_encoder(bev_image)                            # (batch_size, 512)
        # node_features = self.mlp_encoder(self.token_embedding(agent_feats))  # (batch_size, num_agents, 128)
        node_features = self.mlp_encoder(agent_feats)

        # Expand bev_embedding to match agent count and concatenate
        bev_expanded = bev_embedding.unsqueeze(1).expand(-1, num_agents, -1)  
        fused_features = torch.cat([node_features, bev_expanded], dim=-1)  

        scene_graph_embedding = self.gat_encoder(fused_features) 
        ego_node_feature = scene_graph_embedding[:, 0, :] 

        return ego_node_feature


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
        last_layer_dim_pi: int = 256,
        last_layer_dim_vf: int = 256,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)

class CustomDictFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=1)
        # DQ-GAT
        self.DQGAT = DQGAT()

    def forward(self, observations: dict) -> torch.Tensor:
        return self.DQGAT.forward(observations['bev_image'], observations['agent_feats'])
    
class CustomActorCriticPolicy(MultiInputActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=CustomDictFeatureExtractor,
            *args,
            **kwargs,
        )
        self.share_features_extractor = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(feature_dim=256)