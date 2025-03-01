import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from fire import Fire
# import classification_data.train
from models import Classifier
import numpy as np
from datasets.classification_dataset import load_data

def train():
    ## Let's setup the dataloaders
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = load_data('./classification_data/train', transform_pipeline='aug', return_dataloader=False)
    valid_dataset = load_data('./classification_data/val', transform_pipeline='aug', return_dataloader=False)
    
    size = (64,64)
    writer = SummaryWriter()
    writer.add_graph(Classifier(in_channels=3,num_classes=6), torch.zeros(1, 3, *size))
    writer.add_images("train_images", torch.stack([train_dataset[i][0] for i in range(32)]))
    writer.flush()

    net = Classifier(in_channels=3,num_classes=6)
    
    net.to(device)
    
    optim = torch.optim.AdamW(net.parameters(), lr=0.005, weight_decay=1e-4)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=256, num_workers=8)

    global_step = 0
    for epoch in range(10):

        net.train()
        train_accuracy = []
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            output = net(data)
            loss = torch.nn.functional.cross_entropy(output, label)
            train_accuracy.extend((output.argmax(dim=-1) == label).cpu().detach().float().numpy())

            optim.zero_grad()
            loss.backward()
            optim.step()

            writer.add_scalar("train/loss", loss.item(), global_step=global_step)
            global_step += 1

        writer.add_scalar("train/accuracy", np.mean(train_accuracy), epoch)

        net.eval()
        valid_accuracy = []
        for data, label in valid_loader:
            data, label = data.to(device), label.to(device)
            with torch.inference_mode():
                output = net(data)

            valid_accuracy.extend((output.argmax(dim=-1) == label).cpu().detach().float().numpy())

        writer.add_scalar("valid/accuracy", np.mean(valid_accuracy), epoch)

        writer.flush()

        ## Early stopping
        if epoch % 10 == 0:
            torch.save(net.state_dict(), f"model_{epoch}.pth")


if __name__ == "__main__":
    Fire(train)
