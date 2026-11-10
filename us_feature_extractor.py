import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class UltrasoundCNNExtractor(BaseFeaturesExtractor):
    """
    Ultrasound CNN Feature Extractor for 2D/3D B-mode ultrasound observations.
    Accepts Gymnasium Dict observation space containing 'image' and auxiliary vectors
    (e.g. force, pose, contact), or a single Box observation space.
    """
    def __init__(self, observation_space: gym.Space, cnn_output_dim: int = 256):
        self.is_dict = isinstance(observation_space, gym.spaces.Dict)
        if self.is_dict:
            img_space = observation_space.spaces["image"]
        else:
            img_space = observation_space

        # Determine channels and spatial dimensions
        shape = img_space.shape
        if len(shape) == 2:
            n_input_channels = 1
            h, w = shape
        elif len(shape) == 3:
            if shape[0] in [1, 2, 3, 4]:
                n_input_channels = shape[0]
                h, w = shape[1], shape[2]
            else:
                h, w = shape[0], shape[1]
                n_input_channels = shape[2]
        else:
            n_input_channels = 1
            h, w = 128, 128

        super().__init__(observation_space, features_dim=cnn_output_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            dummy = th.zeros(1, n_input_channels, h, w)
            n_flatten = self.cnn(dummy).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, cnn_output_dim),
            nn.ReLU()
        )

        # Handle auxiliary vector keys if Dict observation
        self.vec_keys = []
        vec_dim = 0
        if self.is_dict:
            for k, space in observation_space.spaces.items():
                if k != "image":
                    self.vec_keys.append(k)
                    vec_dim += int(space.shape[0]) if len(space.shape) > 0 else 1

        self.has_vectors = len(self.vec_keys) > 0
        total_dim = cnn_output_dim + vec_dim
        self._features_dim = total_dim

    def forward(self, observations) -> th.Tensor:
        if self.is_dict:
            img = observations["image"]
        else:
            img = observations

        # Ensure (B, C, H, W) float32 in [0, 1]
        if img.dim() == 3:
            img = img.unsqueeze(1)
        elif img.dim() == 4 and img.shape[1] not in [1, 2, 3, 4] and img.shape[-1] in [1, 2, 3, 4]:
            img = img.permute(0, 3, 1, 2)

        if img.dtype == th.uint8 or img.max() > 1.0:
            img = img.float() / 255.0
        else:
            img = img.float()

        cnn_feats = self.linear(self.cnn(img))

        if self.is_dict and self.has_vectors:
            vec_list = [cnn_feats]
            for k in self.vec_keys:
                v = observations[k].float()
                if v.dim() == 1:
                    v = v.unsqueeze(1)
                vec_list.append(v)
            return th.cat(vec_list, dim=1)

        return cnn_feats


class GoalConditionedExtractor(BaseFeaturesExtractor):
    """
    Dual-branch Goal-Conditioned Feature Extractor.
    Branch 1: NatureCNN on B-mode ultrasound image -> cnn_output_dim (default: 128)
    Branch 2: 2-layer MLP on goal/kinematic vectors -> vec_output_dim (default: 128)
    Total embedding dimension = cnn_output_dim + vec_output_dim (default: 256).
    """
    def __init__(self, observation_space: gym.spaces.Dict, cnn_output_dim: int = 128, vec_output_dim: int = 128):
        super().__init__(observation_space, features_dim=cnn_output_dim + vec_output_dim)

        img_space = observation_space.spaces["image"]
        shape = img_space.shape
        if len(shape) == 2:
            n_input_channels = 1
            h, w = shape
        elif len(shape) == 3:
            if shape[0] in [1, 2, 3, 4]:
                n_input_channels = shape[0]
                h, w = shape[1], shape[2]
            else:
                h, w = shape[0], shape[1]
                n_input_channels = shape[2]
        else:
            n_input_channels = 1
            h, w = 128, 128

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            dummy = th.zeros(1, n_input_channels, h, w)
            n_flatten = self.cnn(dummy).shape[1]

        self.cnn_linear = nn.Sequential(
            nn.Linear(n_flatten, cnn_output_dim),
            nn.ReLU()
        )

        self.vec_keys = []
        total_vec_dim = 0
        for k, space in observation_space.spaces.items():
            if k != "image":
                self.vec_keys.append(k)
                total_vec_dim += int(space.shape[0]) if len(space.shape) > 0 else 1

        self.vec_mlp = nn.Sequential(
            nn.Linear(max(total_vec_dim, 1), vec_output_dim),
            nn.ReLU(),
            nn.Linear(vec_output_dim, vec_output_dim),
            nn.ReLU()
        )

    def forward(self, observations: dict) -> th.Tensor:
        img = observations["image"]
        if img.dim() == 3:
            img = img.unsqueeze(1)
        elif img.dim() == 4 and img.shape[1] not in [1, 2, 3, 4] and img.shape[-1] in [1, 2, 3, 4]:
            img = img.permute(0, 3, 1, 2)

        if img.dtype == th.uint8 or img.max() > 1.0:
            img = img.float() / 255.0
        else:
            img = img.float()

        vis_feats = self.cnn_linear(self.cnn(img))

        vec_list = []
        for k in self.vec_keys:
            v = observations[k].float()
            if v.dim() == 1:
                v = v.unsqueeze(1)
            vec_list.append(v)

        if vec_list:
            vec_input = th.cat(vec_list, dim=1)
        else:
            vec_input = th.zeros(vis_feats.shape[0], 1, device=vis_feats.device)

        goal_feats = self.vec_mlp(vec_input)
        return th.cat([vis_feats, goal_feats], dim=1)


class OccupancyConditionedExtractor(BaseFeaturesExtractor):
    """
    Occupancy-conditioned extractor for Task 2 (3D Volumetric Reconstruction).
    Extracts image / coverage grid features combined with kinematic features.
    """
    def __init__(self, observation_space: gym.spaces.Dict, cnn_output_dim: int = 128, vec_output_dim: int = 128):
        super().__init__(observation_space, features_dim=cnn_output_dim + vec_output_dim)

        main_key = "image" if "image" in observation_space.spaces else list(observation_space.spaces.keys())[0]
        space = observation_space.spaces[main_key]
        shape = space.shape
        n_input_channels = 1 if len(shape) == 2 else shape[0]
        h = shape[-2] if len(shape) >= 2 else 40
        w = shape[-1] if len(shape) >= 2 else 40

        self.main_key = main_key
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with th.no_grad():
            dummy = th.zeros(1, n_input_channels, h, w)
            n_flatten = self.cnn(dummy).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, cnn_output_dim),
            nn.ReLU()
        )

        self.vec_keys = [k for k in observation_space.spaces.keys() if k != main_key]
        total_vec_dim = sum(int(observation_space.spaces[k].shape[0]) if len(observation_space.spaces[k].shape) > 0 else 1 for k in self.vec_keys)

        self.vec_mlp = nn.Sequential(
            nn.Linear(max(total_vec_dim, 1), vec_output_dim),
            nn.ReLU()
        )

    def forward(self, observations: dict) -> th.Tensor:
        x = observations[self.main_key].float()
        if x.dim() == 3:
            x = x.unsqueeze(1)
        cnn_feats = self.linear(self.cnn(x))

        vec_list = [observations[k].float() for k in self.vec_keys]
        if vec_list:
            vec_tensor = th.cat([v.unsqueeze(1) if v.dim() == 1 else v for v in vec_list], dim=1)
        else:
            vec_tensor = th.zeros(cnn_feats.shape[0], 1, device=cnn_feats.device)

        vec_feats = self.vec_mlp(vec_tensor)
        return th.cat([cnn_feats, vec_feats], dim=1)
