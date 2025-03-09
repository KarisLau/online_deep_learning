import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from fire import Fire
# import classification_data.train
from models import load_model, save_model
import numpy as np
from datasets.classification_dataset import load_data

#tensorboard --logdir runs --bind_all --reuse_port True

def train(models = 'detector',epochs = 10, batch_size = 256, lr = 0.005, weight_decay = 1e-4,seg_loss_weight =0.5):
    ## Let's setup the dataloaders
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = load_data('./classification_data/train', transform_pipeline='aug', return_dataloader=False)
    valid_dataset = load_data('./classification_data/val', transform_pipeline='aug', return_dataloader=False)
    
    size = (96, 128)
    model = load_model(models,with_weights=False).to(device)
    writer = SummaryWriter()
    writer.add_graph(model, torch.zeros(1, 3, *size))
    writer.add_images("train_images", torch.stack([train_dataset[i][0] for i in range(2)]))
    writer.flush()

    net = model
    
    optim = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)

    global_step = 0
    
    for epoch in range(epochs):

        net.train()
        seg_accuracy = []
        depth_accuracy = []
        for data, label, depth in train_loader:
            data, label, depth = data.to(device), label.to(device), depth.to(device)
            logits, depth_preds = net(data)
            # loss = torch.nn.functional.cross_entropy(output, label)
               # Compute losses
            seg_loss = torch.nn.functional.cross_entropy(logits, label)  # labels should be of shape (B, 96, 128)
            depth_loss = torch.nn.functional.mse_loss(depth_preds, depth)      # depths should be of shape (B, 96, 128)
            seg_accuracy.extend((logits.argmax(dim=-1) == label).cpu().detach().float().numpy())
            depth_accuracy.extend((depth_preds.argmax(dim=-1) == label).cpu().detach().float().numpy())

            optim.zero_grad()
        
            # Combine losses
            # total_loss = seg_loss + depth_loss  # You can also use weights like: total_loss = seg_loss + 0.5 * depth_loss
            
            # Backward pass and optimization
             # Combine losses
            total_loss = seg_loss_weight * seg_loss + (1-seg_loss_weight) * depth_loss

            # Backward pass and optimization
            optim.zero_grad()
            total_loss.backward()
            optim.step()

            # Accuracy calculations
            seg_accuracy.extend((logits.argmax(dim=1) == label).cpu().detach().float().numpy())
            depth_accuracy.extend((depth_preds.squeeze(1).argmax(dim=1) == depth).cpu().detach().float().numpy())

            # Log losses
            writer.add_scalar("driver_train/seg_loss", seg_loss.item(), global_step=global_step)
            writer.add_scalar("driver_train/depth_loss", depth_loss.item(), global_step=global_step)
            writer.add_scalar("driver_train/total_loss", total_loss.item(), global_step=global_step)
            global_step += 1

            # Log accuracies
            writer.add_scalar("driver_train/seg_accuracy", np.mean(seg_accuracy), epoch)
            writer.add_scalar("driver_train/depth_accuracy", np.mean(depth_accuracy), epoch)

        # Validation phase
        net.eval()
        valid_accuracy = []
        with torch.inference_mode(): 
            for data, label in valid_loader:
                data, label = data.to(device), label.to(device)
                with torch.inference_mode():
                    logits, depth_preds = net(data)

                valid_accuracy.extend((logits.argmax(dim=1) == label).cpu().detach().float().numpy())

        writer.add_scalar("driver_valid/accuracy", np.mean(valid_accuracy), epoch)

        writer.flush()


    #     ## Early stopping
    #     if epoch % 10 == 0:
    #         torch.save(net.state_dict(), f"model_{epoch}.pth")
    
    # models.
        save_model(net)

if __name__ == "__main__":
    print(train())
