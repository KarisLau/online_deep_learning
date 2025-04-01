"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

print("Time to train")

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from homework.models import load_model, save_model
import numpy as np
from homework.datasets.road_dataset import load_data
from homework.metrics import PlannerMetric
from torchvision import transforms
# homework/datasets/road_dataset.py

def train(models = 'linear_planner',epochs = 250, batch_size = 256, lr = 1e-3/2, weight_decay = 1e-4,transform_pipeline = 'default'):
    init_lat = 999
    init_long = 999
    
    ## Let's setup the dataloaders
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")
    

    train_dataset = load_data('./drive_data/train',shuffle=True, return_dataloader=False,transform_pipeline =transform_pipeline)
    # train_dataset.transform = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomResizedCrop(size=(96,128), scale=(0.8, 1.0),antialias=True),  # Random crop with sclare 80% -> 100% with smoothing
    #     transforms.ColorJitter(0.9, 0.9, 0.9, 0.1), # Random color jitter with brightness, contrast, saturation and hue
    #     transforms.ToTensor()
    # ])
    valid_dataset = load_data('./drive_data/val',shuffle=False, return_dataloader=False,transform_pipeline='default')
    
    # size = (96,128)
    model = load_model(models,with_weights=False) #.to(device)
    writer = SummaryWriter()
    # writer.add_graph(model, torch.zeros(1, 3, *size))
    
    net = model
    net.to(device)
    train_dm = PlannerMetric()
    val_dm = PlannerMetric()

    optim = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers)

    global_step = 0
    # print(seg_weight)

    for epoch in range(epochs):
        net.train()
        # train_accuracy = []
        total_loss = 0.0
        train_dm.reset()
        val_dm.reset()
        # Training phase
        
        for batch in train_loader:
            track_left = batch['track_left'].to(device)
            track_right = batch['track_right'].to(device)
            waypoints = batch['waypoints'].to(device)
            waypoints_mask = batch['waypoints_mask'].to(device)
            
            optim.zero_grad()
            preds = net(track_left, track_right)
            
            
            # Compute loss
            loss = abs(preds - waypoints).mean()
            loss.backward()
            optim.step()
            total_loss += loss.item()
            train_dm.add(preds, waypoints, waypoints_mask)
            writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

        #     # "l1_error": float(l1_error),
        #     # "longitudinal_error": float(longitudinal_error),
        #     # "lateral_error": float(lateral_error),
        #     # "num_samples": self.total,
        
        train_dm_m = train_dm.compute()
        writer.add_scalar('train/l1_error', train_dm_m['l1_error'], epoch)
        writer.add_scalar('train/longitudinal_error', train_dm_m['longitudinal_error'], epoch)
        writer.add_scalar('train/lateral_error', train_dm_m['lateral_error'], epoch)
        writer.add_scalar('train/num_samples', train_dm_m['num_samples'], epoch)
        #print(f"Epoch {epoch+1}/{epochs}, Train Long_Err: {train_dm_m['longitudinal_error']:.4f}, Train Lat Err: {train_dm_m['lateral_error']:.4f}")
        writer.flush()

        net.eval()
        with torch.inference_mode():
            for batch in valid_loader:
                track_left = batch['track_left'].to(device)
                track_right = batch['track_right'].to(device)
                waypoints = batch['waypoints'].to(device)
                waypoints_mask = batch['waypoints_mask'].to(device)
                preds = net(track_left, track_right)
                val_dm.add(preds, waypoints, waypoints_mask)
                
        val_dm_m = val_dm.compute()
        writer.add_scalar('val/l1_error', val_dm_m['l1_error'], epoch)
        writer.add_scalar('val/longitudinal_error', val_dm_m['longitudinal_error'], epoch)
        writer.add_scalar('val/lateral_error', val_dm_m['lateral_error'], epoch)
        writer.add_scalar('val/num_samples', val_dm_m['num_samples'], epoch)
        writer.flush()
        #- < 0.2 Longitudinal error
        # - < 0.6 Lateral error
        print(f"Epoch {epoch+1}/{epochs}, V Long error: {val_dm_m['longitudinal_error']:.4f}, V Lat Error : {val_dm_m['lateral_error']:.4f}")
        
        if val_dm_m['longitudinal_error'] < init_long and val_dm_m['lateral_error'] < init_lat:
            init_long = val_dm_m['longitudinal_error']
            init_lat = val_dm_m['lateral_error']
        
            save_model(net)
        
            print(f"Epoch {epoch+1} is saved, V Long error: {val_dm_m['longitudinal_error']:.4f}, V Lat Error : {val_dm_m['lateral_error']:.4f}")
        
        

if __name__ == "__main__":
    train()
