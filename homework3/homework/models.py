from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class Classifier(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))
    
        # TODO: implement
 
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
        
        channels_l0 = 64
        cnn_layers = [
            torch.nn.Conv2d(in_channels, channels_l0, kernel_size=11, stride=2, padding=5),
            torch.nn.ReLU(),
        ]
        c1 = channels_l0
        n_blocks = 3
        for _ in range(n_blocks):
            c2 = c1 * 2
            cnn_layers.append(Block(c1, c2, stride=2))
            c1 = c2
        cnn_layers.append(torch.nn.Conv2d(c1, num_classes, kernel_size=1))
        self.network = torch.nn.Sequential(*cnn_layers)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # TODO: replace with actual forward pass
         # Forward through the network
        z = self.network(z)

        # Fully connected layer to produce logits (B, num_classes)
        logits = z.mean(dim=-1).mean(dim=-1)


        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)


class Detector(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # TODO: implement
         # Feature extractor (shared layers)
         # Define the convolutional layers for feature extraction

        # Define the DownBlock for downsampling
        class DownBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super(DownBlock, self).__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
                self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=2)

            def forward(self, x):
                x = self.conv(x)
                return self.downsample(x)

        # Define the UpBlock for upsampling
        class UpBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super(UpBlock, self).__init__()
                stride = 2
                self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=stride)
                self.conv = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
                self.skip = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0) if in_channels != out_channels else torch.nn.Identity()

            def forward(self, x, skip_connection):
                x0 = x
                x = self.upconv(x)
                x = self.conv(x)
                skip = self.skip(skip_connection)
                return skip + x
                #return self.skip(x0) + x 

        self.approach = 1
        if self.approach ==1 : #not using up and down network
            # Downsampling layers
            self.down_conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)  # (B, 16, 48, 64)
            self.down_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1) # (B, 32, 24, 32)
            
            # Up-sampling layers
            self.up_conv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)    # (B, 16, 48, 64)
            self.up_conv2 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)     # (B, 16, 96, 128)
            
            # Final layers for logits and depth
            self.logits_conv = nn.Conv2d(16, 3, kernel_size=1)                     # (B, 3, 96, 128)
            self.depth_conv = nn.Conv2d(16, 1, kernel_size=1)                      # (B, 1, 96, 128)

            # Activation
            self.relu = nn.ReLU()
        elif self.approach ==2:
            # Feature extractor (shared layers)
            self.down1 = DownBlock(in_channels, 32)  # First down block outputs 32 channels
            self.down2 = DownBlock(32, 64)            # Second down block outputs 64 channels

            # Up-sampling layers
            self.up1 = UpBlock(64, 32)  # Up sampling from 64 channels to 32 channels
            self.up2 = UpBlock(32, 16)   # Up sampling from 32 channels to 16 channels

            # Final layers for logits and depth
            self.logits_conv = nn.Conv2d(16, num_classes, kernel_size=1)
            self.depth_conv = nn.Conv2d(16, 1, kernel_size=1)


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # TODO: replace with actual forward pass
        
        if self.approach ==1: 
            # Down-sampling path
            z1 = self.relu(self.down_conv1(z))  # Down1
            z2 = self.relu(self.down_conv2(z1)) # Down2
            
            # Up-sampling path
            z3 = self.up_conv1(z2)  # Up1
            z4 = self.relu(z3 + z1)  # Skip connection with Down1
            z5 = self.up_conv2(z4)  # Up2
            
            # Logits and Depth output
            logits = self.logits_conv(z5)  # (B, 3, 96, 128)
            depth = self.depth_conv(z5)     # (B, 1, 96, 128)

            raw_depth = depth.squeeze(1)  # (B, 96, 128)
        elif self.approach ==2:
                # Down-sampling path
            z1 = self.down1(z)  # Down1, outputs 32 channels
            z2 = self.down2(z1) # Down2, outputs 64 channels

            # Up-sampling path
            z3 = self.up1(z2, z1)  # Up1 with skip connection from Down1
            z4 = self.up2(z3, z1)  # Up2 with skip connection from Down1 (or z3 depending on your design)

            # Logits and Depth output
            logits = self.logits_conv(z4)  # (B, num_classes, 96, 128)
            depth = self.depth_conv(z4)     # (B, 1, 96, 128)

            raw_depth = depth.squeeze(1)  # (B, 96, 128)

        return logits, raw_depth

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)

        # Optional additional post-processing for depth only if needed
        depth = raw_depth

        return pred, depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
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
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = "mps"
            else:
                print("CUDA not available, using CPU")
                device = "cpu"
            
            m.load_state_dict(torch.load(model_path, map_location="device"))
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
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")
    #sample_batch = torch.rand(batch_size, 3, 64, 64).to(device) #Classfication
    '''Detector'''
#     Your `forward` function receives a `(B, 3, 96, 128)` image tensor as an input and should return both:
# - `(B, 3, 96, 128)` logits for the 3 classes
# - `(B, 96, 128)` tensor of depths.

    sample_batch = torch.rand(batch_size, 3, 96, 128).to(device) #Classfication
    print(f"Input shape: {sample_batch.shape}")

    # model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    # output = model(sample_batch)
    # print(f"Output shape: {output.shape}")
    model = load_model("detector", in_channels=3, num_classes=3).to(device)
    logit, depth = model(sample_batch)
    pred,depth = model.predict(sample_batch)
    print(f'logit shape: {logit.shape}')
    print(f'depth shape: {depth.shape}')
    print(f'pred shape: {pred.shape}')
    
    
    


if __name__ == "__main__":
    debug_model(10)
