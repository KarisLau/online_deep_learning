import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
# from fire import Fire
# import classification_data.train
from models import load_model, save_model
import numpy as np
from datasets.road_dataset import load_data
import matplotlib.pyplot as plt
import time
import pickle
from metrics import DetectionMetric, AccuracyMetric

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


    train_dataset = load_data(train_dataset, transform_pipeline='default', return_dataloader=False)
    valid_dataset = load_data(valid_dataset, transform_pipeline='default', return_dataloader=False)

    train_dataset
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
    train_accuracies = []
    valid_accuracies = []

    # Create instances of the metric classes
    train_accuracy_metric = DetectionMetric(num_classes=3)
    val_accuracy_metric = DetectionMetric(num_classes=3)

    for epoch in range(epochs):
        train_accuracy_metric.reset()
        val_accuracy_metric.reset()

        net.train()
        train_accuracy = []
        seg_accuracy = []
        depth_accuracy = []
        total_loss = 0.0
        for batch in train_loader:
            images = batch["image"]
            depths = batch["depth"]
            tracks = batch["track"]
            images, tracks, depths = images.to(device), tracks.to(device),depths.to(device)

            # Zero the gradients
            optim.zero_grad()

            # Forward pass
            logits,raw_depth = net(images)
            
            # Compute the losses
            loss_segmentation = torch.nn.functional.cross_entropy(logits, tracks)
            loss_depth = torch.nn.functional.mse_loss(raw_depth, depths)      # depths should be of shape (B, 96, 128)
            
            # criterion_segmentation(logits, tracks)  # Assuming `tracks` are the ground truth labels
            # loss_depth = criterion_depth(raw_depth, depths)

            # Total loss
            loss = loss_segmentation*seg_loss_weight + loss_depth*(1-seg_loss_weight)

            # Backward pass
            loss.backward()
            optim.step()

            # Compute the metrics
            pred,raw_depth = net.predict(images)
            train_accuracy_metric.add(pred, tracks, raw_depth, depths)
            # train_detection_metric.update(logits, tracks)

            # Update the total loss
            total_loss += loss.item()

            # Update the global step
            global_step += 1

            # Log the metrics
            writer.add_scalar("detec_train/Loss", loss.item(), global_step)
            
        train_acc_metrics = train_accuracy_metric.compute()
        # train_detection_metrics = train_detection_metric.compute()
        writer.add_scalar('detec_train/accruacy', train_acc_metrics["accuracy"], epoch)
        writer.add_scalar("detec_train/iou", train_acc_metrics["iou"], epoch)
        writer.add_scalar("detect_train/depth_error", train_acc_metrics["abs_depth_error"], epoch)
        writer.add_scalar("detect_train/depth_error_lane", train_acc_metrics["tp_depth_error"], epoch)
        t_acc,t_iou,t_depth,t_depth_lane = train_acc_metrics['accuracy'],train_acc_metrics['iou'],train_acc_metrics['abs_depth_error'],train_acc_metrics['tp_depth_error']
        
        writer.flush()

        net.eval()
        valid_accuracy = []
        with torch.inference_mode(): 
            for batch in valid_loader:
                images = batch["image"]
                tracks = batch["track"]
                depths = batch["depth"]
                images, tracks, depths = images.to(device), tracks.to(device),depths.to(device)

                pred,raw_depth = net.predict(images)

                # Compute the metrics
                val_accuracy_metric.add(pred, tracks, raw_depth, depths)
                # Update the total loss
                total_loss += loss.item()

                # Update the global step
                global_step += 1

                # Log the metrics
                writer.add_scalar("detec_val/Loss", loss.item(), global_step)
        
        val_acc_metrics = val_accuracy_metric.compute()
        v_acc,v_iou,v_depth,v_depth_lane = val_acc_metrics['accuracy'],val_acc_metrics['iou'],val_acc_metrics['abs_depth_error'],val_acc_metrics['tp_depth_error']
        writer.add_scalar('detec_val/accruacy', val_acc_metrics["accuracy"], epoch)
        writer.add_scalar("detec_val/iou", val_acc_metrics["iou"], epoch)
        writer.add_scalar("detect_val/depth_error", val_acc_metrics["abs_depth_error"], epoch)
        writer.add_scalar("detect_val/depth_error_lane", val_acc_metrics["tp_depth_error"], epoch)
        
        
        # writer.add_scalar("valid/accuracy", np.mean(valid_accuracy), epoch)
        writer.flush()

    
        print(f"Epoch [{epoch + 1}/{epochs}] - Train Accuracy: {t_acc:.4f}, Valid Accuracy: {v_acc:.4f}, Valid Iou: {v_iou:.4f}, Valid Depth Error: {v_depth:.4f}, Valid Depth Error Lane: {v_depth_lane:.4f}")

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
    train(models="detector",
    epochs=30,
    lr=1e-3)
