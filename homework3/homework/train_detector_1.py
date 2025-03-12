import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from models import load_model, save_model
import numpy as np
from datasets.road_dataset import load_data
from metrics import DetectionMetric
from torchvision import transforms
# homework/datasets/road_dataset.py

def train(models = 'detector',epochs = 210, batch_size = 256, lr = 1e-3/2, weight_decay = 1e-4,seg_weight = 5):
    init_acc = 0 
    ## Let's setup the dataloaders
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")
    

    train_dataset = load_data('./drive_data/train',shuffle=True, return_dataloader=False)
    train_dataset.transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=(96,128), scale=(0.8, 1.0),antialias=True),  # Random crop with sclare 80% -> 100% with smoothing
        transforms.ColorJitter(0.9, 0.9, 0.9, 0.1), # Random color jitter with brightness, contrast, saturation and hue
        transforms.ToTensor()
    ])
    valid_dataset = load_data('./drive_data/val',shuffle=False, return_dataloader=False)
    
    size = (96,128)
    model = load_model(models,with_weights=False) #.to(device)
    writer = SummaryWriter()
    writer.add_graph(model, torch.zeros(1, 3, *size))
    
    net = model
    net.to(device)
    train_dm = DetectionMetric()
    val_dm = DetectionMetric()

    optim = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)

    global_step = 0

    for epoch in range(epochs):
        net.train()
        # train_accuracy = []
        total_loss = 0.0
        train_dm.reset()
        val_dm.reset()
        
        # Training phase
        # for images, depths, tracks in train_loader:
        #     images, depths, tracks = images.to(device), depths.to(device), tracks.to(device)
        for batch in train_loader:
            images = batch['image'].to(device)
            depths = batch['depth'].to(device)
            tracks = batch['track'].to(device)    
            # if global_step == 0:
            #     print(f'train images: {images.shape}, depths: {depths.shape}, tracks: {tracks.shape}')
            optim.zero_grad()
            
            # Forward pass
            logits, depth_pred = model(images)
            seg_loss = torch.nn.functional.cross_entropy(logits, tracks)
            depth_loss = torch.nn.functional.mse_loss(depth_pred, depths)
            loss = seg_loss*seg_weight + depth_loss
            
            
            loss.backward()
            optim.step()
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            # if global_step == 0:
            #     print(f'Train preds: {preds.shape}, tracks: {tracks.shape}, depth_pred: {depth_pred.shape}, depths: {depths.shape}')
            train_dm.add(preds, tracks, depth_pred, depths)
            writer.add_scalar("train/loss", loss.item(), global_step=global_step)
            global_step += 1
        
        train_dm_m = train_dm.compute()
        writer.add_scalar('train/M_accruacy', train_dm_m['accuracy'], epoch)
        writer.add_scalar('train/M_iou', train_dm_m['iou'], epoch)
        # print(f"Epoch {epoch+1}/{epochs}, Train Acc: {train_dm_m['accuracy']:.4f}, Train IoU: {train_dm_m['iou']:.4f}")
        writer.flush()

        net.eval()
        with torch.inference_mode():
            for batch in valid_loader:
                images = batch['image'].to(device)
                depths = batch['depth'].to(device)
                tracks = batch['track'].to(device)    
            # for images, depths, tracks in valid_loader:
            #     images, depths, tracks = images.to(device), depths.to(device), tracks.to(device)
                # if global_step == 0:
                #     print(f'Val images: {images.shape}, depths: {depths.shape}, tracks: {tracks.shape}')
                logits, depth_pred = net(images)
                preds = logits.argmax(dim=1)
                # if global_step == 0:
                #     print(f'Val preds: {preds.shape}, tracks: {tracks.shape}, depth_pred: {depth_pred.shape}, depths: {depths.shape}')
                val_dm.add(preds, tracks, depth_pred, depths)
        val_dm_m = val_dm.compute()
        writer.add_scalar('val/M_accruacy', val_dm_m['accuracy'], epoch)
        writer.add_scalar('val/M_iou', val_dm_m['iou'], epoch)
        writer.add_scalar('val/depth_er', val_dm_m['abs_depth_error'], epoch)
        writer.add_scalar('val/TruePositive', val_dm_m['tp_depth_error'], epoch)
        writer.flush()
        # print(f"Epoch {epoch+1}/{epochs}, T Acc: {train_dm_m['accuracy']:.4f},  V Acc: {val_dm_m['accuracy']:.4f}, V IoU: {val_dm_m['iou']:.4f}, V abs depth: {val_dm_m['abs_depth_error']:.4f}, V TP : {val_dm_m['tp_depth_error']:.4f}")
        if val_dm_m['iou'] > init_acc:
            save_model(net)
            init_acc = val_dm_m['iou']
            print(f"Epoch {epoch+1} is saved, T Acc: {train_dm_m['accuracy']:.4f},  V Acc: {val_dm_m['accuracy']:.4f}, V IoU: {val_dm_m['iou']:.4f}, V abs depth: {val_dm_m['abs_depth_error']:.4f}, V TP : {val_dm_m['tp_depth_error']:.4f}")
    
        
        

if __name__ == "__main__":
    train()
