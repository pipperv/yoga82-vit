import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

def train(model, dataloader, optimizer, criterion, config_dict):
    # Initialize WandB
    wandb.init(project='yoga82-vit', name='run')
    loss_values = []
    avg_loss_values = []
    model.train()

    # Training loop
    epochs = config_dict["epochs"]
    device = config_dict["device"]
    for epoch in range(epochs):
        losses_per_epoch = []
        for batch in tqdm(dataloader):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            # Your training logic here
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            wandb.log({'Train Loss per Step': loss.item()})
            loss_values.append(loss.item())
            losses_per_epoch.append(loss.item())
            loss.backward()
            optimizer.step()
            # Log metrics to WandB
            
        avg_loss = np.mean(losses_per_epoch)
        wandb.log({'Epoch': epoch, 'Train Avg Loss per Epoch': avg_loss.item()})
        avg_loss_values.append(avg_loss)

        # Save the model checkpoint after each epoch
        checkpoint_path = f"model_checkpoint_epoch_{epoch + 1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, checkpoint_path)

        print(f"Epoch {epoch + 1} - Loss: {avg_loss.item()} - Model checkpoint saved to {checkpoint_path}")

    # Optionally, save the model
    wandb.save('your_model.pth')

    print("Training completed.")