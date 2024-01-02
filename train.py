import numpy as np
import torch
import wandb
from tqdm import tqdm

def train(model, train_dataloader, test_dataloader, optimizer, criterion, scheduler, config_dict, run_name='run', init_epoch=0):
    # Initialize WandB
    wandb.init(project='yoga82-vit', name=run_name)
    loss_values = []
    avg_loss_values = []
    model.train()

    # Training loop
    epochs = config_dict["epochs"]
    device = config_dict["device"]
    for epoch in range(init_epoch, epochs):
        print(f'Current Epoch: {epoch+1}')
        losses_per_epoch = []
        for batch in tqdm(train_dataloader):
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
            scheduler.step()
        # Log metrics to WandB
            
        avg_loss = np.mean(losses_per_epoch)
        wandb.log({'Epoch': epoch+1, 'Train Avg Loss per Epoch': avg_loss.item()})
        avg_loss_values.append(avg_loss)

        # Save the model checkpoint after each epoch
        checkpoint_path = f"model_checkpoint_epoch_{epoch + 1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss.item(),
        }, checkpoint_path)

        print(f"Epoch {epoch + 1} - Loss: {avg_loss.item()} - Model checkpoint saved to {checkpoint_path}")

        if (epoch + 1) % 1 == 0:
            model.eval()
            with torch.no_grad():
                losses_per_epoch = []
                for batch in tqdm(test_dataloader):
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    losses_per_epoch.append(loss.item())
                avg_loss = np.mean(losses_per_epoch)
                wandb.log({'Test Avg Loss per Epoch': avg_loss.item()})
                print(f"Epoch {epoch + 1} - Test Loss: {avg_loss.item()}")
            model.train()

    # Optionally, save the model
    wandb.save('your_model.pth')

    print("Training completed.")