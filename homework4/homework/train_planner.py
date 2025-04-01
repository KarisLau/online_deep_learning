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

def train(models="transformer_planner",
        transform_pipeline="state_only",
        num_workers=4,
        lr=1e-3,
        batch_size=128,
        epochs=100,weight_decay = 1e-2):
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
    
    if models == 'cnn_planner':
        transform_pipeline = 'default'
        
    train_dataset = load_data('./drive_data/train',shuffle=True, return_dataloader=False,transform_pipeline =transform_pipeline)
    
    valid_dataset = load_data('./drive_data/val',shuffle=False, return_dataloader=False,transform_pipeline='default')
    
    model = load_model(models,with_weights=False) 
    writer = SummaryWriter()
    
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
        total_loss = 0.0
        train_dm.reset()
        val_dm.reset()
        # Training phase
        
        for batch in train_loader:
            track_left = batch['track_left'].to(device)
            track_right = batch['track_right'].to(device)
            waypoints = batch['waypoints'].to(device)
            waypoints_mask = batch['waypoints_mask'].to(device)
            if models == 'cnn_planner':
                image = batch['image'].to(device)
            
            optim.zero_grad()
            if models == 'cnn_planner':
                preds = net(image)
            else: 
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
        print(f"Epoch {epoch+1}/{epochs}, Train Long_Err: {train_dm_m['longitudinal_error']:.4f}, Train Lat Err: {train_dm_m['lateral_error']:.4f}")
        writer.flush()

        net.eval()
        with torch.inference_mode():
            for batch in valid_loader:
                track_left = batch['track_left'].to(device)
                track_right = batch['track_right'].to(device)
                waypoints = batch['waypoints'].to(device)
                waypoints_mask = batch['waypoints_mask'].to(device)
                image = batch['image'].to(device)
                if models == 'cnn_planner':
                    preds = net(image)
                else:
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
        
        now_long_err = val_dm_m['longitudinal_error']
        now_lat_err = val_dm_m['lateral_error']

        if models!='cnn_planner' and now_long_err <0.2 and now_lat_err <0.6: 
          if val_dm_m['longitudinal_error'] < init_long or val_dm_m['lateral_error'] < init_lat:
              init_long = min(init_long, val_dm_m['longitudinal_error'])
              init_lat = min(val_dm_m['lateral_error'], init_lat)
              save_model(net)
              print(f"Epoch {epoch+1} is saved, V Long error: {val_dm_m['longitudinal_error']:.4f}, V Lat Error : {val_dm_m['lateral_error']:.4f}")
        
        if models == 'cnn_planner' and now_long_err <0.4 and now_lat_err <0.45: 
          if val_dm_m['longitudinal_error'] < init_long or val_dm_m['lateral_error'] < init_lat:
              init_long = min(init_long, val_dm_m['longitudinal_error'])
              init_lat = min(val_dm_m['lateral_error'], init_lat)
              save_model(net)
              print(f"Epoch {epoch+1} is saved, V Long error: {val_dm_m['longitudinal_error']:.4f}, V Lat Error : {val_dm_m['lateral_error']:.4f}")
        
        
        

if __name__ == "__main__":
    train()
