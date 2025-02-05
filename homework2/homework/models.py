"""
Implement the following models for classification.

Feel free to modify the arguments for each of model's __init__ function.
This will be useful for tuning model hyperparameters such as hidden_dim, num_layers, etc,
but remember that the grader will assume the default constructor!
"""

from pathlib import Path

import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss
        Hint: simple one-liner

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        return nn.functional.cross_entropy(logits, target)
        raise NotImplementedError("ClassificationLoss.forward() is not implemented")


class LinearClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
    ):
        """
        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        super().__init__()
        self.input_size = 3 * h * w  # Assuming RGB images
        self.linear_model = torch.nn.Linear(self.input_size, num_classes)
        self.loss = torch.nn.MSELoss()

        # raise NotImplementedError("LinearClassifier.__init__() is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        #reduce x dimension
        flatten_x = x.view(x.size(0),-1)
        #pass to linear layers
        return self.linear_model(flatten_x)
        # raise NotImplementedError("LinearClassifier.forward() is not implemented")


class MLPClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
    ):
        """
        An MLP with a single hidden layer

        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        super().__init__()
        self.input_size = 3 * h * w  # Assuming RGB images
        mid_class_d = w*2
        self.model = nn.Sequential(nn.Linear(self.input_size,mid_class_d)  #first layer
                                #    ,nn.BatchNorm1d(mid_class_d)
                                   ,nn.ReLU(),      #activation function
                                #    nn.BatchNorm1d(mid_class_d),
                                   nn.Linear(mid_class_d,num_classes)) #second layer
        ''' This MLP will take an input image, flatten it, 
        and pass it through a hidden layer with an activation function, followed by an output layer to produce logits for classification.'''
        # raise NotImplementedError("MLPClassifier.__init__() is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
         # Flatten the input tensor
        x = x.view(x.size(0), -1)  # (b, 3*H*W)
        
        # Pass through the output layer
        return self.model(x)
        # raise NotImplementedError("MLPClassifier.forward() is not implemented")


class MLPClassifierDeep(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        num_layers: int = 4,
    ):
        """
        An MLP with multiple hidden layers

        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int

        Hint - you can add more arguments to the constructor such as:
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layers
        
        you can dynamically create the hidden layers based on the specified number of 
        layers and their dimensions. 
        """
        super().__init__()
        # Calculate the input size after flattening
        self.input_size = 3 * h * w  # Assuming RGB images

        hidden_dim = [128*2/2**(i) for i in range(1,num_layers) if 128*2/2**(i)>=num_classes]
        # Create a list to hold the layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(self.input_size, int(hidden_dim[0])))
        # layers.append(nn.BatchNorm1d(int(hidden_dim[0])))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(1,num_layers - 1):
            layers.append(nn.Linear(int(hidden_dim[i-1]), int(hidden_dim[i])))
            # layers.append(nn.BatchNorm1d(int(hidden_dim[i])))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(int(hidden_dim[-1]), num_classes))

        # Combine all layers into a Sequential module
        self.model = nn.Sequential(*layers)

        # raise NotImplementedError("MLPClassifierDeep.__init__() is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Flatten the input tensor
        x = x.view(x.size(0), -1)  # (b, 3*H*W)
        
        # Forward pass through the model
        
        return self.model(x)
        # raise NotImplementedError("MLPClassifierDeep.forward() is not implemented")

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.functional.relu(out)
        return out

class MLPClassifierDeepResidual(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        # hidden_dim: int = 128,
        num_layers: int = 4,
    ):
        """
        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int

        Hint - you can add more arguments to the constructor such as:
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layers
        """
        super().__init__()
        # Calculate the input size after flattening
        self.layers = nn.ModuleList([
            BasicBlock(3, 16),
            BasicBlock(16, 32),
            BasicBlock(32, 64),
            BasicBlock(64, 64)  # Last block can keep the same number of channels
        ])
        self.fc = nn.Linear(64, num_classes)

        # Calculate output size after the last convolutional layer
        self.output_height = h // 2  # Adjust based on the number of downsampling operations
        self.output_width = w // 2    # Adjust based on the number of downsampling operations
        

        # raise NotImplementedError("MLPClassifierDeepResidual.__init__() is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Flatten the input tensor
        # x = x.view(x.size(0), -1)  # (b, 3*H*W)
        # residual = x 
        '''trial 1
        for layer in self.layers:
            x = nn.functional.relu(layer(x))  # Apply each layer and ReLU activation
            x += residual  # Add the residual connection
            residual = x  # Update residual for the next layer
        '''

        '''trail 2
                # Pass through layers with residual connections
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x))  # Apply each layer and ReLU activation
            
            # Ensure the residual is projected to the right size for addition
            if i < len(self.layers) - 1:  # Skip projection for the final layer
                if x.size(1) != residual.size(1):
                     residual = self.residual_layer(residual)  # Project residual to match dimensions
                
                x += residual  # Add the residual connection
                residual = x  # Update residual for the next layer'''
        
        '''trial 3
        x = x.view(x.size(0), -1) 
        if x.size(1) != self.layers[0].in_features:
            raise ValueError(f"Expected input size {self.layers[0].in_features}, got {x.size(1)} instead.")
        identity = x
        
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = nn.functional.relu(x)
                        # Apply residual connection
            if i < len(self.layers) - 2:  # Skip the output layer
                identity = self.projection_layers[i](identity)  # Adjust identity to match dimensions
                x += identity  # Add identity for residual connection
                # identity = x  # Update identity for the next layer
                x = nn.functional.relu(x)  # Apply ReLU after adding residual
            
        x = self.layers[-1](x)'''  # Output layer
        
        '''trial 4
        identity = x  # Original input for the first layer
        for i in range(len(self.layers) - 1):
            print(i)
            x = self.layers[i](x)
            x = nn.functional.relu(x)

            # Apply residual connection
            if i < len(self.layers) - 2:  # Skip the output layer
                # Project identity to match the output dimension if necessary
                if identity.size(1) != x.size(1):
                    identity = nn.functional.relu(self.layers[i](identity))  # Project the identity

                x += identity  # Add identity for the residual connection
                x = nn.functional.relu(x)  # Apply ReLU after adding residual
                
                # Update identity for the next layer
                identity = x  # Set identity to the current output x
            
        x = self.layers[-1](x)'''  # Output layer
        
        '''trial 5
        identity = x  # Original input for the first layer
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = nn.ReLU(x)

            # Apply residual connection
            if i < len(self.layers) - 2:  # Skip the output layer
                # Project identity to match the output dimension if necessary
                if identity.size(1) != x.size(1):
                    identity = nn.ReLU(self.layers[i](identity))  # Project the identity

                x += identity  # Add identity for the residual connection
                x = nn.ReLU(x)  # Apply ReLU after adding residual
                
                # Update identity for the next layer
                identity = x  # Set identity to the current output x
            
        x = self.layers[-1](x)'''  # Output layer
        
        for layer in self.layers:
            x = layer(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))  # Global Average Pooling
        x = x.view(x.size(0), -1)  # Flatten
        logits = self.fc(x)
        return logits
        
                
        
        # Forward pass through the model
        # return self.output_layer(x)
        # raise NotImplementedError("MLPClassifierDeepResidual.forward() is not implemented")


model_factory = {
    "linear": LinearClassifier,
    "mlp": MLPClassifier,
    "mlp_deep": MLPClassifierDeep,
    "mlp_deep_residual": MLPClassifierDeepResidual,
}


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def save_model(model):
    """
    Use this function to save your model in train.py
    """
    for n, m in model_factory.items():
        if isinstance(model, m):
            return torch.save(model.state_dict(), Path(__file__).resolve().parent / f"{n}.th")
    raise ValueError(f"Model type '{str(type(model))}' not supported")


def load_model(model_name: str, with_weights: bool = False, **model_kwargs):
    """
    Called by the grader to load a pre-trained model by name
    """
    r = model_factory[model_name](**model_kwargs)
    if with_weights:
        model_path = Path(__file__).resolve().parent / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        try:
            r.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # Limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(r)
    if model_size_mb > 10:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")
    print(f"Model size: {model_size_mb:.2f} MB")

    return r
