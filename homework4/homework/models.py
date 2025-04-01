from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        #Code Start
        
        class ResidualMLPBlock(nn.Module):
            """MLP block with residual connection"""
            def __init__(self, input_dim, output_dim, dropout=0.0):
                super().__init__()
                # Only add residual if dimensions match
                self.use_residual = (input_dim == output_dim)
                
                self.linear1 = nn.Linear(input_dim, output_dim)
                self.linear2 = nn.Linear(output_dim, output_dim)
                self.activation = nn.ReLU()
                self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
                
                # Optional projection for residual if dimensions don't match
                if input_dim != output_dim:
                    self.residual_proj = nn.Linear(input_dim, output_dim)
                else:
                    self.residual_proj = nn.Identity()

            def forward(self, x):
                residual = self.residual_proj(x)
                
                out = self.linear1(x)
                out = self.activation(out)
                out = self.dropout(out)
                
                out = self.linear2(out)
                
                # Add residual connection
                out = out + residual
                out = self.activation(out)
                
                return out

         # Input size: 2 tracks * n_track points * 2 coordinates
        input_size = 2 * n_track * 2
        
        # Build MLP blocks
        mlp_layers = []
        c1 = input_size
        
        dropout = 0
        block =4
        for _ in range(block): 
            c2 = c1 // 2
            mlp_layers.append(ResidualMLPBlock(c1, c2, dropout))
            c1 = c2
        
        # Final output layer
        mlp_layers.append(nn.Linear(c1, n_waypoints * 2))
        
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        #Code start 
        batch_size = track_left.shape[0]
        # Flatten the track points
        x = torch.cat([track_left, track_right], dim=1)  # (b, 2*n_track, 2)
        x = x.view(batch_size, -1)  # (b, 2*n_track*2)
        
        # Pass through MLP
        out = self.mlp(x)  # (b, n_waypoints*2)
        waypoints= out.view(batch_size, self.n_waypoints, 2)

        return waypoints
        raise NotImplementedError


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        self.query_embed = nn.Embedding(n_waypoints, d_model)

        #Code start 
        self.input_proj = nn.Linear(2, d_model)

        nhead=4
        dim_feedforward=256
        dropout=0
        num_layers=3
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        #Code start 
        batch_size = track_left.shape[0]

        # Combine and project track points (B, 2*n_track, d_model)
        track_points = torch.cat([track_left, track_right], dim=1)
        track_points = self.input_proj(track_points)  # (B, 2*n_track, d_model)

        # Get query embeddings (n_waypoints, d_model) -> (B, n_waypoints, d_model)
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)

        # Transformer decoder (cross-attention: queries attend to track_points)
        waypoint_features = self.transformer_decoder(
            tgt=queries,
            memory=track_points
        )  # (B, n_waypoints, d_model)

        # Project to output coordinates
        waypoints = self.output_proj(waypoint_features)  # (B, n_waypoints, 2)
        return waypoints
        raise NotImplementedError


class ResNetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        
        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                torch.nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        residual = self.shortcut(x)
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return torch.nn.functional.relu(out)

class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()
        self.n_waypoints = n_waypoints

        # Normalization parameters
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)
        
        # ResNet-like backbone
        self.in_channels = 16
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(16)
        
        # ResNet blocks
        self.layer1 = self._make_layer(16, 2, stride=2)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        self.layer4 = self._make_layer(128, 2, stride=2)
        self.layer5 = self._make_layer(256, 2, stride=2)
        
        # Global average pooling
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, n_waypoints * 2)
        )

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResNetBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return torch.nn.Sequential(*layers)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        # Normalize input
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        
        # Feature extraction
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        # Pooling and FC
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        waypoints = self.fc(x)
        
        # Reshape output
        return waypoints.view(-1, self.n_waypoints, 2)

MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
