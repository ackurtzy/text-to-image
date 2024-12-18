"""
Script to run inference on a diffusion model and plot results

Configuration:
    `model_path`: Path to PyTorch diffusion model
    `image_label_paths`: Path to json list with all trainied label paths
    `random_sample`: Whether to randomly pick categories to plot
    `phrases': If random sample is false, will use these phrases as text conditioning
    `diffusion_timesteps`: Number of forward and reverse noise timesteps in diffusion process
    'image_size': Width/height of images (assumed to be square). Must be same as model trained on
    'channels': Number of channels per image. 1 for gray, 3 or RGB, Must be same as model trained on
    'give_save_path`: Path of where to save diffusion process gif of first plotted image

Output:
    Plot of 9 images, either random if random_sample is true, or based on phrases. Also displays gif of
    first image being created through diffusion process. Saves this gif to `gif_save_path`
"""

import json
import random

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

from helpers import encode_text, get_sampling_functions

# Paths
model_path = "models/bigger_embed_space_epoch_3.pth"
image_label_paths = "all_categories.json"
gif_save_path = "results/diffusion_process.gif"

# Set text conditioning
random_sample = True
phrases = ["submarine", "plane"]

# Other parameters
diffusion_timesteps = 300
image_size = 28
channels = 1

sample, _ = get_sampling_functions(timesteps=diffusion_timesteps)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load(model_path).to(device)

if random_sample:
    with open(image_label_paths, "r") as file:
        full_phrase_list = json.load(file)

    random.shuffle(full_phrase_list)
else:
    full_phrase_list = phrases * ((9 // len(phrases)) + 1)


text_embeddings = torch.stack(
    [encode_text(phrase) for phrase in full_phrase_list[:9]], dim=0
).to(device)

# sample 64 images
samples = sample(
    model,
    text_embeddings,
    image_size=image_size,
    batch_size=9,
    channels=channels,
)

figs, axs = plt.subplots(3, 3)

for index, ax in enumerate(axs.flatten()):
    ax.imshow(samples[-1][index].reshape(image_size, image_size, channels), cmap="gray")
    ax.set_title(full_phrase_list[index])

plt.tight_layout()

gif_index = 0

fig = plt.figure()
ims = []
for i in range(diffusion_timesteps):
    im = plt.imshow(
        samples[i][gif_index].reshape(image_size, image_size, channels),
        cmap="gray",
        animated=True,
    )
    ims.append([im])

animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
animate.save(gif_save_path)
plt.show()
