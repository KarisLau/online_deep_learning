import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
# from fire import Fire
# import classification_data.train
from homework.models import load_model, save_model
import numpy as np
from homework.datasets.road_dataset import load_data
import matplotlib.pyplot as plt
import time
import pickle

#tensorboard --logdir runs --bind_all --reuse_port True

def train(models = 'detector',epochs = 10, batch_size = 256, lr = 0.005, weight_decay = 1e-4,seg_loss_weight =0.5):
    ## Let's setup the dataloaders
    timestamps = time.time()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")
    

    train_dataset = load_data('./drive_data/train', transform_pipeline='default', return_dataloader=False)
    valid_dataset = load_data('./drive_data/val', transform_pipeline='default', return_dataloader=False)
    
    size = (96, 128)
    model = load_model(models,with_weights=False) #.to(device)
    writer = SummaryWriter()
    writer.add_graph(model, torch.zeros(1, 3, *size))
    #writer.add_images("train_images", torch.stack([train_dataset[i][0] for i in range(32)]))
    # writer.flush()

    net = model
    net.to(device)
    
    optim = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)

    global_step = 0
    train_accuracies = []
    valid_accuracies = []

    
    for epoch in range(epochs):

        net.train()
        train_accuracy = []
        seg_accuracy = []
        depth_accuracy = []
        total_loss = 0.0
        # output_loss = 0.0
        # dpeth_loss = 0.0
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            logits, depth_preds = net(data)
            # loss = torch.nn.functional.cross_entropy(output, label)
               # Compute losses
            seg_loss = torch.nn.functional.cross_entropy(logits, label)  # labels should be of shape (B, 96, 128)
            depth_loss = torch.nn.functional.mse_loss(depth_preds, depth_preds)      # depths should be of shape (B, 96, 128)
            loss = seg_loss_weight * seg_loss + (1-seg_loss_weight) * depth_loss
            seg_accuracy.extend((logits.argmax(dim=-1) == label).cpu().detach().float().numpy())
            depth_accuracy.extend((depth_preds.argmax(dim=-1) == label).cpu().detach().float().numpy())
            total_loss += loss.item()

            optim.zero_grad()
            loss.backward()
            optim.step()


            '''classification code 
            loss = torch.nn.functional.cross_entropy(output, label)
            train_accuracy.extend((output.argmax(dim=-1) == label).cpu().detach().float().numpy())
            total_loss += loss.item()

            optim.zero_grad()
            loss.backward()
            optim.step()

            writer.add_scalar("train/loss", loss.item(), global_step=global_step)
            global_step += 1'''

            writer.add_scalar("detection_train/seg_loss", seg_loss.item(), global_step=global_step)
            writer.add_scalar("detection_train/depth_loss", depth_loss.item(), global_step=global_step)
            writer.add_scalar("detection_train/loss", loss.item(), global_step=global_step)
            global_step += 1

        writer.add_scalar("detection_train/accuracy", np.mean(train_accuracy), epoch)
        avg_train_loss = total_loss / len(train_loader)
        avg_train_accuracy = np.mean(train_accuracy)
        train_accuracies.append(avg_train_accuracy)

        net.eval()
        valid_accuracy = []
        with torch.inference_mode(): 
            for data, label in valid_loader:
                data, label = data.to(device), label.to(device)
                with torch.inference_mode():
                    logits = net(data)

                valid_accuracy.extend((logits.argmax(dim=-1) == label).cpu().detach().float().numpy())

        
        writer.add_scalar("detection_valid/accuracy", np.mean(valid_accuracy), epoch)
        avg_valid_accuracy = np.mean(valid_accuracy)
        valid_accuracies.append(avg_valid_accuracy)

        writer.flush()

        ''''Update the live plot
        if 1 ==2: 
            line1.set_xdata(np.arange(1, epoch + 2))
            line1.set_ydata(train_accuracies)
            line2.set_xdata(np.arange(1, epoch + 2))
            line2.set_ydata(valid_accuracies)
            
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.1)  # Pause to update the plot'''
        print(f"Epoch [{epoch + 1}/{epochs}] - Train Accuracy: {avg_train_accuracy:.4f}, Valid Accuracy: {avg_valid_accuracy:.4f}")

        ## Early stopping
        if epoch % 10 == 0:
            torch.save(net.state_dict(), f"model_{epoch}.pth")
            
    
    # models.
        # plt.ioff()  # Turn off interactive mode
        # plt.show()  # Show the final plot
        save_model(net)
        # Save accuracies and timestamps to a pickle file
        with open(f'detection_accuracies{timestamps}.pkl', 'wb') as f:
            pickle.dump({'train_accuracies': train_accuracies,
                        'valid_accuracies': valid_accuracies,
                        'timestamps': timestamps}, f)

if __name__ == "__main__":
    train(models="detection",
    epochs=30,
    lr=1e-3)
