import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from models import load_model, save_model
import numpy as np
from datasets.road_dataset import load_data
from metrics import DetectionMetric,AccuracyMetric
from torchvision import transforms


#tensorboard --logdir runs --bind_all --reuse_port True

def train(models = 'detector',epochs = 10, batch_size = 256, lr = 0.005, weight_decay = 1e-4,seg_loss_weight =0.5):
    ## Let's setup the dxataloaders
    init_acc = 0
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    train_dataset = load_data('./drive_data/train', transform_pipeline='default', return_dataloader=False,shuffle=True)
    valid_dataset = load_data('./drive_data/val', transform_pipeline='default', return_dataloader=False,shuffle=False)

    size = (96, 128)
    model = load_model(models,with_weights=False) #.to(device)
    writer = SummaryWriter()
    writer.add_graph(model, torch.zeros(1, 3, *size))

    # writer.flush()

    net = model
    net.to(device)

    optim = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)

    global_step = 0

    # Create instances of the metric classes
    train_accuracy_metric = AccuracyMetric()
    val_accuracy_metric = DetectionMetric(num_classes=3)

    for epoch in range(epochs):
        train_accuracy_metric.reset()
        val_accuracy_metric.reset()
        net.train()
        
        total_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            # transformed_images = []
            depths = batch["depth"].to(device)
            tracks = batch["track"].to(device)
          
            # Forward pass
            logits,raw_depth = net(images)
            
            # Compute the losses
            loss_segmentation = torch.nn.functional.cross_entropy(logits, tracks)
            loss_depth = torch.nn.functional.mse_loss(raw_depth, depths)      # depths should be of shape (B, 96, 128)
         
            # Total loss
            loss = loss_segmentation*seg_loss_weight + loss_depth*(1-seg_loss_weight)
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Compute the metrics
            pred,raw_depth = net.predict(images)
            train_accuracy_metric.add(pred, tracks)
            # train_detection_metric.update(logits, tracks)

            # Update the total loss
            total_loss += loss.item()

            # Update the global step
            global_step += 1

            # Log the metrics
            writer.add_scalar("detec_train/Loss", loss.item(), global_step)
            
        train_acc_metrics = train_accuracy_metric.compute()
        
        writer.add_scalar('detec_train/accruacy', train_acc_metrics["accuracy"], epoch)
        # Reset metrics for the next epoch
        train_accuracy_metric.reset()
        t_acc =train_acc_metrics["accuracy"]
        writer.flush()

        net.eval()
        with torch.inference_mode(): 
            for batch in valid_loader:

                images = batch["image"].to(device)
                depths = batch["depth"].to(device)
                tracks = batch["track"].to(device)
                
                logits,raw_depth = net(images,depths)
                preds = torch.argmax(logits, dim=1)
                # Compute the metrics
                val_accuracy_metric.add(preds, tracks, raw_depth, depths)
                
                depth_error = (raw_depth - depths).abs()
                tp_mask = ((preds == tracks) & (tracks > 0)).float()
                tp_depth_error = depth_error * tp_mask
                tp_depth_error_sum += tp_depth_error.sum().item()
                tp_depth_error_n += tp_mask.sum().item()

            
            # Compute validation metrics
            print(f'{tp_depth_error_sum =}')
            print(f'{tp_depth_error_n =}')
            val_results = val_accuracy_metric.compute()
            v_acc,v_iou,v_depth,v_depth_lane = val_results['accuracy'],val_results['iou'],val_results['abs_depth_error'],val_results['tp_depth_error']

            writer.add_scalar('Validation/IoU', val_results['iou'], epoch)
            writer.add_scalar('Validation/Accuracy', val_results['accuracy'], epoch)
            writer.add_scalar('Validation/AvgDepthError', val_results['abs_depth_error'], epoch)
            writer.add_scalar('Validation/TPDepthError', val_results['tp_depth_error'], epoch)
            val_accuracy_metric.reset()
        writer.flush()

    
        print(f"Epoch [{epoch + 1}/{epochs}] - Train Accuracy: {t_acc:.4f}, Valid Accuracy: {v_acc:.4f}, Valid Iou: {v_iou:.4f}, Valid Depth Error: {v_depth:.4f}, Valid Depth Error Lane: {v_depth_lane:.4f}")

        ## Early stopping
        # if epoch % 10 == 0:
        #     torch.save(net.state_dict(), f"model_{epoch}.pth")
            
        
        if init_acc < v_acc:
            save_model(net)
        
if __name__ == "__main__":
    train(models="detector",
    epochs=3,
    lr=1e-3,
    batch_size= 256)
