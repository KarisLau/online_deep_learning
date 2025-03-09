import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from models import load_model, save_model
import numpy as np
from datasets.classification_dataset import load_data
from metrics import AccuracyMetric

#tensorboard --logdir runs --bind_all --reuse_port True

def train(models = 'classifier',epochs = 20, batch_size = 256, lr = 1e-3, weight_decay = 1e-4):
    init_acc = 0 
    ## Let's setup the dataloaders
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")
    

    train_dataset = load_data('./classification_data/train', transform_pipeline='aug', return_dataloader=False,shuffle=True)
    valid_dataset = load_data('./classification_data/val', transform_pipeline='default', return_dataloader=False,shuffle=False)
    
    size = (64,64)
    model = load_model(models,with_weights=False) #.to(device)
    writer = SummaryWriter('logs')
    writer.add_graph(model, torch.zeros(1, 3, *size))
    writer.add_images("train_images", torch.stack([train_dataset[i][0] for i in range(32)]))
    # writer.flush()

    net = model
    net.to(device)
    train_am = AccuracyMetric()
    valid_am = AccuracyMetric()
    
    optim = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)

    global_step = 0
    train_accuracies = []
    valid_accuracies = []

    for epoch in range(epochs):
        net.train()
        train_accuracy = []
        total_loss = 0.0
        train_am.reset()
        valid_am.reset()
        
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            output = net(data)
            loss = torch.nn.functional.cross_entropy(output, label)
            train_accuracy.extend((output.argmax(dim=-1) == label).cpu().detach().float().numpy())
            total_loss += loss.item()

            optim.zero_grad()
            loss.backward()
            optim.step()

            pred= net.predict(data)
            train_am.add(pred, label)
            writer.add_scalar("train/loss", loss.item(), global_step=global_step)
            global_step += 1

        writer.add_scalar("train/accuracy", np.mean(train_accuracy), epoch)
        train_am_m = train_am.compute()
        writer.add_scalar('train/M_accruacy', train_am_m['accuracy'], epoch)
        train_M = train_am_m['accuracy']

        avg_train_loss = total_loss / len(train_loader)
        avg_train_accuracy = np.mean(train_accuracy)
        train_accuracies.append(avg_train_accuracy)

        net.eval()
        valid_accuracy = []
        
        for data, label in valid_loader:
            data, label = data.to(device), label.to(device)
            with torch.inference_mode(): 
                logits = net(data)
                valid_accuracy.extend((logits.argmax(dim=-1) == label).cpu().detach().float().numpy())
                pred= net.predict(data)
                valid_am.add(pred, label)

        valid_am_m = valid_am.compute()
        writer.add_scalar('train/M_accruacy', valid_am_m['accuracy'], epoch)
        valid_M = valid_am_m['accuracy']
        writer.add_scalar("valid/accuracy", np.mean(valid_accuracy), epoch)
        avg_valid_accuracy = np.mean(valid_accuracy)
        valid_accuracies.append(avg_valid_accuracy)

        writer.flush()

        print(f"Epoch [{epoch + 1}/{epochs}] - Train Accuracy: {train_M:.4f}, Valid Accuracy: {valid_M:.4f}")

        ## Early stopping
        # if epoch % 10 == 0:
        #     torch.save(net.state_dict(), f"model_{epoch}.pth")
    
        if init_acc < valid_M: #save model if val accuracy is highest
            save_model(net)
        
    
if __name__ == "__main__":
    train(
    models="classifier",
    epochs=20,
    lr=1e-3,
)
