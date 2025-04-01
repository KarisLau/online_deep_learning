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


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        #Code start 
        self.embed_dim = 128
        self.num_heads = 8
        self.num_layers = 4

        class CNNBackbone(torch.nn.Module):
            class Block(torch.nn.Module):
                def __init__(self, in_channels, out_channels, stride):
                    super().__init__()
                    kernel_size = 3
                    padding = (kernel_size-1)//2

                    self.c1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
                    self.n1 = torch.nn.GroupNorm(1, out_channels)
                    self.c2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
                    self.n2 = torch.nn.GroupNorm(1, out_channels)
                    self.relu1 = torch.nn.ReLU()
                    self.relu2 = torch.nn.ReLU()

                    self.skip = torch.nn.Conv2d(in_channels, out_channels, 1, stride, 0) if in_channels != out_channels else torch.nn.Identity()
                    # self.pool = nn.MaxPool2d(2, 2)



                def forward(self, x0):
                    x = self.relu1(self.n1(self.c1(x0)))
                    x = self.relu2(self.n2(self.c2(x)))
                    # x = self.pool(x)
                    return self.skip(x0) + x 
            
            # Define the CNN layers
            
            channels_l0 = 32
            cnn_layers = [
                torch.nn.Conv2d(n_waypoints, channels_l0, kernel_size=3, stride=2, padding=1),
                torch.nn.ReLU(),
            ]
            c1 = channels_l0
            n_blocks = 3
            for _ in range(n_blocks):
                c2 = c1 * 2
                cnn_layers.append(Block(c1, c2, stride=2))
                c1 = c2
            # cnn_layers.append(torch.nn.Conv2d(c1, c1, kernel_size=1))
            cnn_layers.append(Block(c1, c1, stride=2))
            self.network = torch.nn.Sequential(*cnn_layers)

        self.backbone = CNNBackbone()
        class TransformerLayer(torch.nn.Module):
            def __init__(self, embed_dim, num_heads):
                super().__init__()

                self.self_att = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
                self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(embed_dim, 4 * embed_dim), torch.nn.ReLU(), torch.nn.Linear(4 * embed_dim, embed_dim)
                )
                self.in_norm = torch.nn.LayerNorm(embed_dim)
                self.mlp_norm = torch.nn.LayerNorm(embed_dim)

            def forward(self, x):
                x_norm = self.in_norm(x)
                x = x + self.self_att(x_norm, x_norm, x_norm)[0]
                x = x + self.mlp(self.mlp_norm(x))
                return x

        class Transformer(torch.nn.Module):
            def __init__(self, embed_dim, num_heads, num_layers):
                super().__init__()
                self.network = torch.nn.Sequential(*[TransformerLayer(embed_dim, num_heads) for _ in range(num_layers)])

            def forward(self, x):
                return self.network(x)
        
        self.transformer = Transformer(self.embed_dim, self.num_heads, self.num_layers)
        self.fc = torch.nn.Linear(self.embed_dim, 2 * n_waypoints)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        #code start
        features = self.backbone(x)
        b, c, h, w = features.shape
        features = features.view(b, c, h * w).permute(2, 0, 1)
        features = self.transformer(features)
        features = features.permute(1, 0, 2)
        waypoints = self.fc(features)
        waypoints = waypoints.view(b, self.n_waypoints, 2)

        if 1 ==2:  #POE Code 1
            # Reshape features for the transformer
            batch_size, channels, height, width = features.size()
            features = features.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)  # shape (B, H*W, C)

            # Pass through the transformer
            transformer_output = self.transformer(features)  # shape (B, H*W, embed_dim)

            # Use the last output from the transformer
            last_output = transformer_output[:, -1, :]  # shape (B, embed_dim)

            # Predict waypoints
            waypoints_output = self.fc(last_output)  # shape (B, n_waypoints * 2)

            # Reshape to get the waypoints
            waypoints_output = waypoints_output.view(-1, self.n_waypoints, 2)  # shape (B, n_waypoints, 2)
        return waypoints

        raise NotImplementedError


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
