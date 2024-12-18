"""
Library of helpers for defining and training diffusion model
"""

import os
import json
import random
from inspect import isfunction

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from transformers import BertModel, BertTokenizer
from einops import rearrange
from tqdm.auto import tqdm


def exists(variable):
    """
    Checks if a variable is None. If not none returns true else false

    Args:
        variable: Any variable

    Returns:
        boolean: True if x not None false if None
    """
    return variable is not None


def default(value, defualt):
    """
    If value exists, return it. Else return default or default's callable return

    Args:
        value: Any variable
        default: Any variable or callable

    Returns:
        value if exists, default or default() otherwise
    """

    if exists(value):
        return value
    return defualt() if isfunction(defualt) else defualt


class Residual(nn.Module):
    """
    Module that adds the input to the output of a function applied to the input.

    Attributes:
        fn (callable): A function or a neural network module that takes `x` as input
            and returns an output of the same shape as `x`.
    """

    def __init__(self, fn):
        """
        Initializes residual connection

        Args:
            fn (callable): A function or a neural network module that takes `x` as input
            and returns an output of the same shape as `x`.
        """
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        """
        Forward pass for the Residual module.

        Args:
            x (torch.Tensor): Input tensor to the residual block.
            *args: Additional positional arguments passed to `fn`.
            **kwargs: Additional keyword arguments passed to `fn`.

        Returns:
            torch.Tensor: The result of fn(x, *args, **kwargs) + x.
        """
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):  # Scale up image by factor of 2 using nearest neighbor
    """
    Return upsampling PyTorch sequence consisting of 2 scaling and a convolutional layer

    Sequential with a scaling factor of 2 using nearest neighbor and a convolutional layer

    Args:
        dim (int): Number of input channels
        dim_out (int, optional): Number of output channels if different than input

    Returns:
        nn.Sequential: A PyTorch sequential module containing:
            - `nn.Upsample`: Scales the input feature map by a factor of 2 using
              nearest-neighbor interpolation.
            - `nn.Conv2d`: A convolutional layer with kernel size 3, padding 1, and
              input/output channels defined by `dim` and `dim_out`.
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(
            dim, default(dim_out, dim), 3, padding=1
        ),  # If no dim_out, use dim and keep same number of dimensions
    )


class Rearrange(nn.Module):
    """
    A custom PyTorch module for performing tensor rearrangement using `einops.rearrange`.

    This module applies a specified rearrangement pattern to the input tensor. The
    rearrangement is defined using a pattern string and optional keyword arguments.

    Attributes:
        pattern (str): The rearrangement pattern string compatible with `einops.rearrange`.
        **kwargs: Additional keyword arguments passed to `einops.rearrange`, such as
            dimensions or specific transformations.
    """

    def __init__(self, pattern, **kwargs):
        """
        Initializes the Rearrange module with a pattern and optional arguments.

        Args:
            pattern (str): The pattern string for rearranging the input tensor.
            **kwargs: Additional keyword arguments passed to `einops.rearrange`.
        """
        super().__init__()
        self.pattern = pattern
        self.kwargs = kwargs

    def forward(self, x):
        """
        Applies the rearrangement pattern to the input tensor.

        Args:
            x (torch.Tensor): The input tensor to be rearranged.

        Returns:
            torch.Tensor: The rearranged tensor according to the specified pattern.
        """
        return rearrange(x, self.pattern, **self.kwargs)


def Downsample(dim, dim_out=None):
    """
    Return downsampling PyTorch sequence consisting of 2 scaling and a convolutional layer

    Downsamples without strided convolutions or pooling by stacking 2x2 chunks of the image
    into channels, then doing 1x1 convolution

    Args:
        dim (int): Number of input channels
        dim_out (int, optional): Number of output channels if different than input

    Returns:
        nn.Sequential: A PyTorch sequential module containing:
            - `Rearrange`: Custom rearrange module to stack 2x2 chunks of image
            - `nn.Conv2d`: A convolutional layer with kernel size 1 and
              input/output channels defined by `dim` and `dim_out`.
    """
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, dim_out if dim_out else dim, 1),
    )


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Makes a cosine-based beta schedule for diffusion models.

    This schedule is for the noise variance level. Cosine beta schedule
    makes noise more gradual by calculating a cumulative product of alpha terms.

    Args:
        timesteps (int): The total number of timesteps for the diffusion process.
        s (float, optional): A small offset to prevent the cosine argument from reaching 0.
            Defaults to 0.008.

    Returns:
        torch.Tensor: A 1D tensor of shape `(timesteps,)` containing the beta values
        for each timestep, clipped to the range `[0.0001, 0.9999]`.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)

    # Cummulative product of alpha terms
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def encode_text(text, pretrained_model_name="bert-base-uncased"):
    """
    Create text embeddings for a word or phrase using BERT text encoding model.

    Args:
        text (str, list of str, or dict): Words, list of words, or dict with lists of phrases.
        pretrained_model_name (str, optional): BERT embeddings model to use.

    Returns:
        phrase_embedding (torch.Tensor or dict): If input is str, returns (768,) tensor.
            If input is list, returns dict with strings as keys and embeddings as values.
            If input is dict, returns dict with keys and stacked tensors of embeddings as values.
    """
    bert = BertModel.from_pretrained(pretrained_model_name)
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
    if isinstance(text, str):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = bert(**inputs)
        # cls_embeddings = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token embeddings
        token_embeddings = outputs.last_hidden_state.squeeze()[1:-1]

        # Compute the mean across all token embeddings
        phrase_embedding = token_embeddings.mean(dim=0).squeeze()
    elif isinstance(text, list):
        phrase_embedding = {}
        for label in text:
            inputs = tokenizer(
                label, return_tensors="pt", padding=True, truncation=True
            )
            outputs = bert(**inputs)
            # cls_embeddings = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token embeddings
            token_embeddings = outputs.last_hidden_state.squeeze()[1:-1]

            # Compute the mean across all token embeddings
            embedding = token_embeddings.mean(dim=0).squeeze()

            phrase_embedding[label] = embedding
    elif isinstance(text, dict):
        phrase_embedding = {}
        for key, val in text.items():
            label_embed_list = []
            for phrase in val:
                inputs = tokenizer(
                    phrase, return_tensors="pt", padding=True, truncation=True
                )
                outputs = bert(**inputs)
                # cls_embeddings = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token embeddings
                token_embeddings = outputs.last_hidden_state.squeeze()[1:-1]

                # Compute the mean across all token embeddings
                embedding = token_embeddings.mean(dim=0).squeeze()
                label_embed_list.append(embedding)

            phrase_embedding[key] = label_embed_list

    return phrase_embedding


def get_embedding_dataset(
    data_or_folder_path, alternate_embedding_path, samples_per_type
):
    """
    Returns a PyTorch dataset from Quick, Draw! .npy files with text embeddings.

    Args:
        data_or_folder_path (str or list): Paths to get drawings from
        alternate_embedding_path (str): Path to json file with alternate embeddings
        samples_per_type (int): Maximum number of samples per category

    Returns:
        QuickDrawDatasetText object that returns numpy image and text embedding
    """
    if isinstance(data_or_folder_path, list):
        sketch_paths = data_or_folder_path
    else:
        sketch_paths = []
        for index, name in enumerate(os.listdir(data_or_folder_path)):
            if name[-4:] == ".npy":
                full_path = os.path.join(data_or_folder_path, name)
                sketch_paths.append(full_path)

    sketches = np.load(sketch_paths[0], encoding="latin1", allow_pickle=True)[
        :samples_per_type, :
    ]
    labels = [sketch_paths[0].split("/")[-1].replace(".npy", "")] * sketches.shape[0]
    if len(sketch_paths) > 1:
        for index in range(1, len(sketch_paths)):
            new_sketches = np.load(
                sketch_paths[index], encoding="latin1", allow_pickle=True
            )[:samples_per_type, :]
            sketches = np.vstack((sketches, new_sketches))
            labels = labels + (
                [sketch_paths[index].split("/")[-1].replace(".npy", "")]
                * new_sketches.shape[0]
            )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # shape (channels, height, width), divide by 255
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda t: (t * 2) - 1),  # Scale -1 to 1
        ]
    )

    with open(alternate_embedding_path, "r") as file:
        alternate_embeddings = json.load(file)

    embedding_dict = encode_text(alternate_embeddings)

    class QuickDrawDatasetText(Dataset):
        """
        Dataset for Quick, Draw! data with alternate text embeddings
        """

        def __init__(self, data, embedding_dict, text, transform=None):
            """
            Intializes dataset

            Args:
                data (torch.Tensor): Image data for dataset in form (samples, 784)
                embedding_dict (dict of list of tensors): Dict with key for each label category
                    containing a list of tensors with all embeddings for the category
                text (list of str): Category labels corresponding to data of length samples
            """
            self.data = data
            self.transform = transform
            self.embedding_dict = embedding_dict
            self.text = text

        def __len__(self):
            """
            Get the number of datapoints
            """
            return len(self.data)

        def __getitem__(self, idx):
            """
            Returns data from dataset at index idx after applying transformation

            Args:
                idx (int): Index to return data

            Returns:
                sketch as (28, 28) torch.Tensor, text embedding as (768,) torch.Tensor
            """
            # Load and reshape the sketch
            sketch = (
                self.data[idx].reshape(28, 28).astype(np.uint8)
            )  # Ensure 8-bit image format

            # Apply transformations if defined
            if self.transform:
                sketch = self.transform(sketch)

            all_embeddings = self.embedding_dict[self.text[idx]]

            return sketch, random.choice(all_embeddings)

    return QuickDrawDatasetText(sketches, embedding_dict, labels, transform=transform)


def get_sampling_functions(timesteps):
    """
    Get functions for diffusion process based on the number of timesteps

    Args:
        timesteps (int): Number of noising/denoising timesteps in diffusion process

    Returns:
        sample (function): Function that performs denoising process by sampling from diffusion model
        q_sample (function): Simulate the forward diffusion process to get the noisy image at timestep
            t given the start image
    """
    # define beta schedule
    betas = cosine_beta_schedule(timesteps=timesteps)

    # define alphas
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(
        alphas, axis=0
    )  # Calculate cumulative alphas, which is how much of the signal was retained
    alphas_cumprod_prev = F.pad(
        alphas_cumprod[:-1], (1, 0), value=1.0
    )  # Pushing all back by inserting a 1 at index 0
    sqrt_recip_alphas = torch.sqrt(
        1.0 / alphas
    )  # Calculate the 1/sqrt(alpha) or 1/sqrt(1 - B) which is used to normalize the noise

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)  # used in noising equations
    sqrt_one_minus_alphas_cumprod = torch.sqrt(
        1.0 - alphas_cumprod
    )  # used in noising equation

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = (
        betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )  # variance used to compute in gaussian distribute to get one step more noisy

    def extract(
        a, t, x_shape
    ):  # a is a precomputed tensor of shape (timesteps,), t is a tensor of timesteps shaped (batch_size,)
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(
            t.device
        )  # Reshape it for broadcasting with shape (batch size, 1, ... as many 1s as to match x_shape)

    # Forward diffusion function using the precomputed values:
    # This function simulates the noisy image x_t at timestep t given the original image x_start.
    def q_sample(x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)  # generate random noise

        # Get calculations for the batches
        sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        # sampling equation: sqrt(alpha_cum) * x_start + sqrt(1 - alpha_cum) * random noise
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample(
        model, x, text_embeddings, t, t_index
    ):  # model, x=image batch tensor, t = tensor of timesteps, t_index = current timestep in process
        betas_t = extract(betas, t, x.shape)  # get the betas for the batch
        sqrt_one_minus_alphas_cumprod_t = extract(
            sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        # This is denoising equation from notes
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, text_embeddings, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Add gaussian noise
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def p_sample_loop(model, text_embeddings, shape):
        device = next(model.parameters()).device

        b = shape[0]
        # start from pure noise (for each text_embedingsexample in the batch)
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(
            reversed(range(0, timesteps)),
            desc="sampling loop time step",
            total=timesteps,
        ):  # tqdm adds a progress bar
            img = p_sample(
                model,
                img,
                text_embeddings,
                torch.full((b,), i, device=device, dtype=torch.long),
                i,
            )  # t is a tensor of is
            imgs.append(img.cpu().numpy())
        return imgs

    @torch.no_grad()
    def sample(model, text_embeddings, image_size, batch_size=16, channels=3):
        return p_sample_loop(
            model,
            text_embeddings,
            shape=(batch_size, channels, image_size, image_size),
        )

    return sample, q_sample
