import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pix2pix_dataset import Pix2PixDataset
from GANNetwork import Generator, Discriminator
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
# 引入tensorboard
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).astype(np.uint8)
    return image


def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        # Convert tensors to images
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # Concatenate the images horizontally
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(
            f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)


def train_one_epoch(generator, discriminator, dataloader, optimizer_generator,
                    optimizer_discriminator, criterion_rec, criterion_adv, device, epoch, num_epochs,reconstr_weight=20.0):
    """
    Train the model for one epoch.
    """
    generator.train()
    discriminator.train()
    running_loss_generator = 0.0
    running_loss_discriminator = 0.0
    loop = tqdm(enumerate(dataloader), total=len(dataloader), position=0,
                ncols=120, desc=f"Training Epoch [{epoch + 1}/{num_epochs}]")
    for i, (image_rgb, image_semantic) in loop:
        # Move data to the device
        image_rgb = image_rgb.to(device)
        image_semantic = image_semantic.to(device)

        # Zero the gradients
        optimizer_generator.zero_grad()

        # Forward pass
        # Train the discriminator

        optimizer_discriminator.zero_grad()

        pred_real = discriminator(torch.cat([image_rgb, image_semantic], dim=1))
        loss_real = criterion_adv(pred_real, torch.ones_like(pred_real))
        
        fake_semantic = generator(image_rgb)
        pred_fake = discriminator(torch.cat([image_rgb, fake_semantic], dim=1))
        loss_fake = criterion_adv(pred_fake, torch.zeros_like(pred_fake))

        loss_discriminator = (loss_real + loss_fake) / 2

        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Train the generator
        optimizer_generator.zero_grad()

        fake_semantic = generator(image_rgb)
        pred_fake = discriminator(torch.cat([image_rgb, fake_semantic], dim=1))
        loss_generator_adv = criterion_adv(pred_fake, torch.ones_like(pred_fake))
        loss_generator_rec = criterion_rec(fake_semantic, image_semantic)

        loss_generator = (loss_generator_adv + reconstr_weight*loss_generator_rec)/(reconstr_weight+1)

        loss_generator.backward()
        optimizer_generator.step()

        # Save sample images every 5 epochs
        if epoch % 5 == 0 and i == 0:
            save_images(image_rgb, image_semantic, fake_semantic,
                        'train_results', epoch, num_images=10)

        # Update running loss
        running_loss_generator += loss_generator.item()
        running_loss_discriminator += loss_discriminator.item()

        # Print loss information
        # print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')
        # loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}]\n')
        loop.set_postfix(generator_loss=loss_generator.item(),discriminator_loss=loss_discriminator.item())

    avg_train_loss_generator = running_loss_generator / len(dataloader)
    avg_train_loss_discriminator = running_loss_discriminator / len(dataloader)
    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Generator Loss: {avg_train_loss_generator :.4f} Discriminator Loss: {avg_train_loss_discriminator:.4f}')
    return avg_train_loss_generator, avg_train_loss_discriminator


def validate(model, dataloader, criterion_rec,criterion_adv, device, epoch, num_epochs):
    """
    Validate the model on the validation dataset.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the validation data.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the validation on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        loop = tqdm(enumerate(dataloader), total=len(dataloader), position=0,
                    ncols=120, desc=f"Validation Epoch [{epoch + 1}/{num_epochs}]")
        for i, (image_rgb, image_semantic) in loop:
            # Move data to the device
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            # Forward pass
            outputs = model(image_rgb)

            # Compute the loss
            loss = criterion_rec(outputs, image_semantic)
            val_loss += loss.item()

            # Save sample images every 5 epochs
            if epoch % 5 == 0 and i == 0:
                save_images(image_rgb, image_semantic, outputs,
                            'val_results', epoch, num_images=10)

    # Calculate average validation loss
    avg_val_loss = val_loss / len(dataloader)
    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Validation Generator Loss: {avg_val_loss:.4f}')
    return avg_val_loss


def main():
    """
    Main function to set up the training and validation processes.
    """

    best_val_loss = 1000
    # Set device to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 添加一个时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = {
        'train_generator': SummaryWriter(f'logs/{timestamp}/train_generator'),
        'train_discriminator': SummaryWriter(f'logs/{timestamp}/train_discriminator'),
        'val': SummaryWriter(f'logs/{timestamp}/val'),
        "model_generator": SummaryWriter(f'logs/{timestamp}/model_generator'),
        "model_discriminator": SummaryWriter(f'logs/{timestamp}/model_discriminator'),
    }
    # Initialize datasets and dataloaders
    train_dataset = Pix2PixDataset(list_file='train_list.txt')
    val_dataset = Pix2PixDataset(list_file='val_list.txt')

    train_loader = DataLoader(
        train_dataset, batch_size=100, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=100,
                            shuffle=False, num_workers=4)

    # Initialize model, loss function, and optimizer
    generator = Generator().to(device)
    writer['model_generator'].add_graph(
        generator, torch.randn(1, 3, 256, 256).to(device))
    discriminator = Discriminator().to(device)
    writer['model_discriminator'].add_graph(
        discriminator, torch.randn(1, 6, 256, 256).to(device))
    # 鼓励生成器生成真实图像
    criterion_rec = nn.L1Loss()
    # 鼓励提升生成器欺骗鉴别器能力
    criterion_adv = nn.BCEWithLogitsLoss()
    optimizer_generator = optim.Adam(
        generator.parameters(), lr=0.001, betas=(0.5, 0.999))
    optimizer_discriminator = optim.Adam(
        discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))

    # Add a learning rate scheduler for decay
    scheduler_geneator = StepLR(optimizer_generator, step_size=200, gamma=0.2)
    scheduler_discriminator = StepLR(
        optimizer_discriminator, step_size=200, gamma=0.2)

    # Training loop
    num_epochs = 800
    for epoch in range(num_epochs):
        avg_train_loss_generator,avg_train_loss_discriminator = train_one_epoch(generator, discriminator, train_loader,
                                         optimizer_generator, optimizer_discriminator,
                                         criterion_rec, criterion_adv, device, epoch, num_epochs)
        avg_val_loss = validate(
            generator, val_loader, criterion_rec, criterion_adv, device, epoch, num_epochs)

        # Log the losses to TensorBoard
        writer['train_generator'].add_scalar(
            'Generator Loss', avg_train_loss_generator, epoch)
        writer['train_discriminator'].add_scalar(
            'Discriminator Loss', avg_train_loss_discriminator, epoch)
        writer['val'].add_scalar('Generator Loss', avg_val_loss, epoch)
        # writer['lr'].add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        # Step the scheduler after each epoch
        scheduler_geneator.step()
        scheduler_discriminator.step()

        # Save model checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0 and avg_val_loss < best_val_loss:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(generator.state_dict(),
                       f'checkpoints/pix2pix_generator_epoch_{epoch + 1}.pth')
            best_val_loss = avg_val_loss


if __name__ == '__main__':
    main()
