import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
# from fire import Fire
# import classification_data.train
from models import load_model, save_model
import numpy as np
from datasets.classification_dataset import load_data

#tensorboard --logdir runs --bind_all --reuse_port True

def train(models = 'classifier',epochs = 10, batch_size = 256, lr = 0.005, weight_decay = 1e-4):
    ## Let's setup the dataloaders
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")
    

    train_dataset = load_data('./classification_data/train', transform_pipeline='aug', return_dataloader=False)
    valid_dataset = load_data('./classification_data/val', transform_pipeline='aug', return_dataloader=False)
    
    size = (64,64)
    model = load_model(models,with_weights=False) #.to(device)
    writer = SummaryWriter()
    writer.add_graph(model, torch.zeros(1, 3, *size))
    writer.add_images("train_images", torch.stack([train_dataset[i][0] for i in range(32)]))
    writer.flush()

    net = model
    
    # optim = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    # valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)

    # global_step = 0
    # for epoch in range(epochs):

    #     net.train()
    #     train_accuracy = []
    #     for data, label in train_loader:
    #         data, label = data.to(device), label.to(device)
    #         output = net(data)
    #         loss = torch.nn.functional.cross_entropy(output, label)
    #         train_accuracy.extend((output.argmax(dim=-1) == label).cpu().detach().float().numpy())

    #         optim.zero_grad()
    #         loss.backward()
    #         optim.step()

    #         writer.add_scalar("train/loss", loss.item(), global_step=global_step)
    #         global_step += 1

    #     writer.add_scalar("train/accuracy", np.mean(train_accuracy), epoch)

    #     net.eval()
    #     valid_accuracy = []
    #     with torch.inference_mode(): 
    #         for data, label in valid_loader:
    #             data, label = data.to(device), label.to(device)
    #             with torch.inference_mode():
    #                 logits, depth_preds = net(data)

    #             valid_accuracy.extend((logits.argmax(dim=1) == label).cpu().detach().float().numpy())

        
    #     writer.add_scalar("valid/accuracy", np.mean(valid_accuracy), epoch)

    #     writer.flush()

    #     ## Early stopping
    #     if epoch % 10 == 0:
    #         torch.save(net.state_dict(), f"model_{epoch}.pth")
    
    # # models.
    #     save_model(net)

if __name__ == "__main__":
    train()
