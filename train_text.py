"""
Script to train Unet Diffusion model

Configuration:
    'data_paths': Path to data folder with .npy files
    'alternate_embeddings_path': Path to json file with data labels as keys and list 
        of alternate text phrases
    'image_size': Width/height of images (assumed to be square)
    'channels': Number of channels per image. 1 for gray, 3 or RGB
    'number_examples_per_label': Number of examples in training set per file in data_paths
    'diffusion_timesteps': Number of forward and reverse noise timesteps in diffusion process
    'learning_rate': Learning rate for AdamW
    'epochs': Number of training epochs
    'batch_size': Batch size for training
    'model_save_path': Base path of where to save model
    'load_model_path': Path to load model and resume training. If invalid path, starts new model
    'start_epoch': Which epoch to resume training. If starts new model, will set to 0

Output:
    Trained PyTorch model will be saved every .5 epochs. Will print loss and epoch every 100 batches
"""

from pathlib import Path
import os

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from helpers import (
    get_embedding_dataset,
    get_sampling_functions,
)
from model_text import Unet

# Data paths
data_paths = "guessed_quickdraw_data"
alternate_embeddings_path = "alternate_embeddings.json"

# Dataset parameters
image_size = 28
channels = 1
number_examples_per_label = 45000

# Model/Training Hyperparameters
diffusion_timesteps = 300
learning_rate = 0.001
epochs = 10
batch_size = 400

# Model save base path
model_save_path = "models/bigger_embed_space"

# Load model to resume training. Empty string for new model
load_model_path = "models/bigger_embed_space_epoch_1.pth"
start_epoch = 4

_, q_sample = get_sampling_functions(timesteps=diffusion_timesteps)


def p_losses(denoise_model, x_start, x_emb, t, noise=None, loss_type="l1"):
    """
    Function to make inference and calculate loss between true and predicted noise

    Args:
        denoise_model (nn.Module): Unet model to run inference on
        x_start (torch.Tensor): Images in batch with shape (B, C, W, H)
        x_emb (torch.Tensor): Text embeddings for images in batch as (B, 768)
        t (torch.Tensor): Tensor of ints of diffusion timestep to train on (B,)
        noise (torch.Tensor, optional): Noise added to images
        loss_type (str, optional): Which loss to use. Options are l1, l2, huber

    Returns:
        torch.Tensor of training loss
    """

    if noise is None:
        # Generate noise if there isn't any
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)  # Image with noise
    predicted_noise = denoise_model(
        x_noisy, x_emb, t
    )  # Predict how much noise was added

    # Regression style loss between predicted amounts of noise
    if loss_type == "l1":
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == "l2":
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


dataset = get_embedding_dataset(
    data_paths, alternate_embeddings_path, number_examples_per_label
)

dataset_size = len(dataset)

print(
    f"Dataset has {dataset_size} and {len(dataset)/number_examples_per_label} categories"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

results_folder = Path("./results")
results_folder.mkdir(exist_ok=True)

if os.path.exists(load_model_path):
    model = torch.load(load_model_path).to(device)
else:
    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(
            1,
            2,
            4,
        ),
    )
    start_epoch = 0

model.to(device)

optimizer = Adam(model.parameters(), lr=learning_rate)
steps_per_epoch = dataset_size / batch_size
half_epoch = steps_per_epoch // 2

for epoch in range(start_epoch, epochs):
    for step, (batch_img, batch_emb) in enumerate(dataloader):
        optimizer.zero_grad()

        batch_size = batch_img.shape[0]
        batch_img = batch_img.to(device)
        batch_emb = batch_emb.to(device)

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        t = torch.randint(0, diffusion_timesteps, (batch_size,), device=device).long()

        loss = p_losses(model, batch_img, batch_emb, t, loss_type="huber")

        if step % 100 == 0:
            print(f"Loss: {loss.item()}, Epoch: {epoch + (step/steps_per_epoch)}")

        if step == half_epoch:
            torch.save(model, f"{model_save_path}_half_{epoch}.pth")

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} done")
    torch.save(model, f"{model_save_path}_{epoch}.pth")

torch.save(model, f"{model_save_path}_complete.pth")
