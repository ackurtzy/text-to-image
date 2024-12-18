"""
Classes that build up to a Unet text to image diffusion model
"""

import math
from functools import partial
from einops import rearrange, reduce
import torch
from torch import nn, einsum
import torch.nn.functional as F

from helpers import exists, default, Residual, Downsample, Upsample


class SinusoidalPositionEmbeddings(nn.Module):
    """
    PyTorch nn.Module to create sinusoidal position embeddings
    """

    # Creates sinusoidal positional embeddings
    def __init__(self, dim):
        """
        Initialize position embeddings class

        Args:
            dim (int): Dimension of position embeddings
        """
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Calculate the sinusoidal position embeddings.

        Sine waves make up first half of dim, cosines the second half. They have varying frequencies
        in each dimension to convey positional information.

        Args:
            time (torch.Tensor): Vector of positions to encode. Typically range 1 - token index

        Returns:
            embeddings (torch.Tensor): Position embeddings for each position in time tensor. Size (time.shape[0], dim)
        """
        # time is a vector range 1 - token index, first half of embedding dims sine, second half cos
        device = time.device
        half_dim = self.dim // 2

        # Log step size: ln(10000) / d
        embeddings = math.log(10000) / (half_dim - 1)

        # -ln(10000)/2 * i
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        # None unsqueezes the vector by adding a new axis with
        # dim 1: time: (3,) -> (3,1), then broadcasted to (3,3) and embeddings: (3,) -> (1, 3), then broadcast to (3, 3)
        # Finally elemntwise multiplication
        embeddings = time[:, None] * embeddings[None, :]

        # Take sin and cos, then stack them
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class WeightStandardizedConv2d(nn.Conv2d):
    """
    Weight standardization wrapper for nn.Conv2d. It has been shown to work well with
    group normalization which is also used in the Unet
    """

    def forward(self, x):
        """
        Modified forward function that first normalizes the weights by subtracting
        mean and dividing by variance

        Args:
            x (torch.Tensor): Data to run forward pass on

        Returns:
            torch.Tensor that is result of conv2d with normalized weights
        """

        # For normalization, there aren't learnable parameters, but gradients still flow through them
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = (
            self.weight
        )  # Shape (o: output channels, i: input channels, h: height, w: width)
        mean = reduce(
            weight, "o ... -> o 1 1 1", "mean"
        )  # Keeps the output channel, reduces by taking means across input, height, and widths

        # Keeps the output channel, reduces by taking means across input, height, and widths
        # partial creates a partial function by prefilling parameters of input function. In this case torch.var is filled with unbiased = False
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    """
    Block with Conv2d, group normalization, and activation function
    """

    def __init__(self, dim, dim_out, groups=8):
        """
        Intialize a Block

        Args:
            dim (int): Number of input channels
            dim_out (int): Number of output channels
            groups (int, optional): Number of groups in group norm
        """
        super().__init__()
        self.proj = WeightStandardizedConv2d(
            dim, dim_out, 3, padding=1
        )  # Weight standardized Conv2d
        self.norm = nn.GroupNorm(
            groups, dim_out
        )  # Group Norm with learnable scaling and bias
        self.act = nn.SiLU()  # More drawn out curve than GeLU

    def forward(self, x, scale_shift=None):  # scale_shift = (scale, shift)
        """
        Forward pass through block

        Args:
            x (torch.Tensor): Data to run forward pass on
            scale_shift (tuple, optional): Conditioning to add the block in form (scale, shift),
                where scale, shift are torch.Tensor that are broadcastable with x. Typically it is
                [batch_size, dim_out, 1, 1] to scale/shift each channel independently.

        Returns:
            x (torch.Tensor): Output of block operation
        """
        x = self.proj(x)  # Apply conv2d
        x = self.norm(x)  # Normalize

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)  # SiLU
        return x


class ResnetBlock(nn.Module):  # stack of Conv2d blocks with residual connection
    """
    Block of ResNet Unet as detailed in https://arxiv.org/abs/1512.03385

    ResNet block consists of an MLP on the time/text embeddings if there are any,
    2 Block classes, and a residual connection that projects dimensions if necessary.
    """

    def __init__(self, dim, dim_out, *, emb_dim=None, groups=8):
        """
        Intializes the ResNet block

        Args:
            dim (int): Number of input channels
            dim_out (int): Number of output channels
            time_emb_dim (int, optional): Size of embedding dimension
            groups (int, optional): Number of groups for group norm
        """
        super().__init__()
        self.mlp = (  # MLP to turn positional embeddings into scale_shift transformations
            nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, dim_out * 2))
            if exists(emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)  # Dim in, dim out
        self.block2 = Block(dim_out, dim_out, groups=groups)  # Dim in, dim out
        self.res_conv = (
            nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        )  # residual connection, projects if needed to match dim. Identity(x) returns x

    def forward(self, x, embedding=None):
        """
        Forward pass of resnet block

        Args:
            x (torch.Tensor): Data to run forward pass on
            embedding (torch.Tensor): time/text embeddings

        Returns:
            torch.Tensor that is output of operations
        """
        # Positional embeddings aren't directly added, but rather a scale and shift is computed from then which then augments the features
        scale_shift = None
        if exists(self.mlp) and exists(embedding):
            embedding = self.mlp(embedding)
            embedding = rearrange(embedding, "b c -> b c 1 1")
            scale_shift = embedding.chunk(
                2, dim=1
            )  # split into 2 chunks, so first half determine the scale, second half the shift for each feature

        h = self.block1(
            x, scale_shift=scale_shift
        )  # First block with optional position embedding
        h = self.block2(h)  # second block
        return h + self.res_conv(x)  # Add second block output with residual connection


class Attention(nn.Module):
    """
    Implements a self-attention mechanism for 2D input tensors (e.g., feature maps).

    Parameters:
        dim (int): Number of input channels in the feature map.
        heads (int): Number of attention heads. Default is 4.
        dim_head (int): Dimensionality of each attention head. Default is 32.

    Attributes:
        scale (float): Scaling factor to normalize the dot-product attention scores.
        heads (int): Number of attention heads.
        to_qkv (nn.Conv2d): A 1x1 convolution that projects the input into queries, keys, and values.
        Output channels = 3 * (heads * dim_head), since queries, keys, and values each need their own projection.
        to_out (nn.Conv2d): A 1x1 convolution that projects the concatenated attention output back to the input dimension.
    """

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        # Scaling factor for the queries to stabilize gradients during training.
        # The scale is the inverse square root of the dimension of each attention head.
        self.scale = dim_head**-0.5

        # Number of attention heads
        self.heads = heads

        # Compute the total hidden dimension as heads * dim_head
        hidden_dim = dim_head * heads

        # 1x1 convolution to project the input tensor into queries, keys, and values
        # Produces 3 sets of projections (query, key, value), each of size hidden_dim
        # Each pixel is a token
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        # 1x1 convolution to project the output of attention back to the input dimension
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        """
        Forward pass for the attention mechanism.

        Args:
          x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
          torch.Tensor: Output tensor of shape (batch_size, channels, height, width).
        """
        # Extract dimensions of the input tensor
        b, c, h, w = x.shape  # batch_size, channels, height, width

        # Project the input tensor into queries, keys, and values
        # `to_qkv` outputs a tensor of shape (b, 3 * hidden_dim, h, w)
        # Use `chunk(3, dim=1)` to split it into 3 tensors along the channel dimension
        qkv = self.to_qkv(x).chunk(3, dim=1)  # query, key, value tensors

        # Rearrange queries, keys, and values for multi-head attention
        # Each tensor is reshaped to (batch_size, heads, dim_head, height * width), so that the tokens are flattened
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        # Scale the queries by the normalization factor
        q = q * self.scale

        # Compute attention scores (similarity) between queries and keys
        # Using Einstein summation for efficient batch matrix multiplication
        # sim: (batch_size, heads, height * width, height * width)
        # multiply ith query and jth key - element-wise multiply the tensors, then sum in d dimension
        sim = einsum("b h d i, b h d j -> b h i j", q, k)

        # Normalize the attention scores by subtracting the maximum score for numerical stability
        # This step ensures that the subsequent softmax operation is stable
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()

        # Apply softmax to compute attention weights
        # attn: (batch_size, heads, height * width, height * width)
        attn = sim.softmax(dim=-1)

        # Compute the weighted sum of values based on attention weights
        # out: (batch_size, heads, height * width, dim_head)
        # Learnings from attention for each pixel
        out = einsum("b h i j, b h d j -> b h i d", attn, v)

        # Rearrange the output tensor back to the spatial dimensions
        # Shape: (batch_size, heads * dim_head, height, width)
        # Equavalent to stacking the attention outputs
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)

        # Project the concatenated attention output back to the original number of channels
        # Final output shape: (batch_size, channels, height, width)
        # Give it the flexibility to rearrange
        return self.to_out(out)


class LinearAttention(nn.Module):
    """
    Implements a linearized self-attention mechanism for 2D input tensors (e.g., feature maps).

    Parameters:
        dim (int): Number of input channels in the feature map.
        heads (int): Number of attention heads. Default is 4.
        im_head (int): Dimensionality of each attention head. Default is 32.

    Attributes:
        scale (float): Scaling factor to normalize the query values.
        heads (int): Number of attention heads.
        to_out (nn.Sequential): A 1x1 convolution followed by a group normalization layer to project
        to_qkv (nn.Conv2d): A 1x1 convolution to project the input into queries, keys, and values.
            the output back to the original input dimension with added normalization.
    """

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        # Scaling factor for queries to stabilize gradients
        self.scale = dim_head**-0.5

        # Number of attention heads
        self.heads = heads

        # Compute the hidden dimension as heads * dim_head
        hidden_dim = dim_head * heads

        # 1x1 convolution to project the input into queries, keys, and values
        # Produces 3 sets of projections (query, key, value), each of size hidden_dim
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        # Output projection followed by group normalization
        # GroupNorm(1, dim) normalizes across spatial dimensions for each channel
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), nn.GroupNorm(1, dim))

    def forward(self, x):
        """
        Forward pass for linear attention.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, channels, height, width).
        """
        # Extract input dimensions
        b, c, h, w = x.shape  # batch_size, channels, height, width

        # Project input into queries, keys, and values
        # `to_qkv` outputs a tensor of shape (b, 3 * hidden_dim, h, w)
        # Use `chunk(3, dim=1)` to split into 3 tensors along the channel dimension
        qkv = self.to_qkv(x).chunk(3, dim=1)  # query, key, value tensors

        # Rearrange each tensor for multi-head attention
        # Each tensor is reshaped to (batch_size, heads, dim_head, height * width)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        # Normalize the queries with a softmax operation along the spatial dimension
        # dim=-2 refers to the "head channel" (height * width)
        q = q.softmax(dim=-2)

        # Normalize the keys with a softmax operation along the "key" dimension
        # dim=-1 refers to the spatial dimension (height * width)
        k = k.softmax(dim=-1)

        # Scale the queries by the normalization factor
        q = q * self.scale

        # Compute the context using an Einstein summation
        # context: (batch_size, heads, dim_head, dim_head)
        # This step computes the weighted sum of the values using the normalized keys
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        # Use the context to compute the output feature maps
        # out: (batch_size, heads, dim_head, height * width)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)

        # Rearrange the output tensor back to the spatial dimensions
        # Shape: (batch_size, heads * dim_head, height, width)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)

        # Project the output back to the input dimension and normalize it
        # Final output shape: (batch_size, channels, height, width)
        return self.to_out(out)


class PreNorm(nn.Module):
    """
    Group normalization wrapper to apply before attention mechanism
    """

    def __init__(self, dim, fn):
        """
        Intialize the PreNorm wrapper

        Args:
            dim (int): Number of channels
            fn (callable): Function to apply after normalization
        """
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        """
        Runs forward pass of by applying normalization then function

        Args:
            x (torch.Tensor): Data to run forward pass on

        Returns:
            torch.Tensor output of function
        """
        x = self.norm(x)
        return self.fn(x)


class Unet(nn.Module):
    """
    Unet class for a diffusion model

    Parameters:
        channels (int): Number of input channels
        self_condition (boolean): Whether to add a conditioning image
        init_conv (nn.Conv2d): Convolution to apply at beginning of network
        time_mlp (nn.Sequential): Creates sinusoidal position embeddings, then
            applies linear layer and activation function
        embedding_mlp (nn.Sequential): Performers linear layer on text embeddings,
            then applies activation funciton
        combine_conditional (nn.Linear): Linear layer on concatenated time_mlp and
            embedding_mlp output
        downs (nn.ModuleList): List of downsampling layers
        ups (nn.ModuleList): List of upsampling layers
        mid_block1 (partially parameterized ResNet block): First middle of Unet block
        mid_attn (Attention): Attention to apply at center of network
        mid_block2 (partially parameterized ResNet block): Second middle of Unet block
        out_dim (int): Number of output channels
        final_res_block (partially parameterized ResNet block): Last resnet block in network
        final_conv (nn.Conv2d): Final pointwise convolutional layer to apply
    """

    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(
            1,
            2,
            4,
            8,
        ),
        text_embedding_dim=768,
        channels=3,
        self_condition=False,
        resnet_block_groups=4,
    ):
        """
        Intialize the Unet

        Args:
            dim (int): Base number of channels for Unet features
            init_dim (int, optional): Number of channels of initial features after first conv layer.
                Defaults to dim
            out_dim (int, optional): Number of output channles
            dim_mults (tuple of ints, optional): Scaling factors for channels in downsampling and upsampling
            text_embedding_dim (int, optional): Size of text embedding
            channels (int, optional): Number of image input channels
            self_condition (boolean, optional): Whether to add a conditioning image
            resnet_block_groups (int, optional): Number of groups for group norm within resnet


        """
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)

        # Pointwise convolution to rearrange and mix channels
        self.init_conv = nn.Conv2d(
            input_channels, init_dim, 1, padding=0
        )  # changed to 1 and 0 from 7,3

        # Creates list of
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(
            zip(dims[:-1], dims[1:])
        )  # input and output dimensions for each layer of U-net

        # Prefill ResNetBlock with the number of block groups
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings - 4x bigger for richer representation
        time_dim = 768

        # Create high dimensional representation of the positional embeddings with non linear layer
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim), nn.Linear(dim, time_dim), nn.GELU()
        )

        # Reducing embedding dimensionality
        self.embedding_mlp = nn.Sequential(
            nn.Linear(text_embedding_dim, time_dim), nn.GELU()
        )

        # Combine the positional embeddings and text embeddings
        self.combine_conditional = nn.Linear(time_dim * 2, time_dim)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(
            in_out
        ):  # Append blocks with 2 ResNet blocks, attention, downsampling, Conv2d
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        (
                            Downsample(dim_in, dim_out)
                            if not is_last
                            else nn.Conv2d(dim_in, dim_out, 3, padding=1)
                        ),
                    ]
                )
            )

        # Last dim size is at middle of U-Net
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(
            PreNorm(mid_dim, Attention(mid_dim))
        )  # Attention with pre attention group norm added back to residual
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(
            reversed(in_out)
        ):  # Append blocks with 2 ResNet blocks, attention, downsampling, Conv2d
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(
                            dim_out + dim_in, dim_out, time_emb_dim=time_dim
                        ),  # Skip connections
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        (
                            Upsample(dim_out, dim_in)
                            if not is_last
                            else nn.Conv2d(dim_out, dim_in, 3, padding=1)
                        ),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)  # Final pointwise convolution

    # You can condition by giving it the previous prediction.
    # Ex, if denoising from x_5 to x_4, could give it x_6
    def forward(self, x, text_emb, time, x_self_cond=None):
        """
        Defines forward pass of the Unet

        You can condition by giving it the previous prediction. Ex, if denoising from
        x_5 to x_4, could give it x_6

        Args:
            x (torch.Tensor): Input data to run forward pass on (B, C, W, H)
            text_emb (torch.Tensor): Text embeddings of size (B, text_embedding_dim)
            time (torch.Tensor): Which time position to denoise (B,)
            x_self_cond (torch.Tensor): Conditioning tensor of shape (B, C, W, H)
        """
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        position_embeddings = self.time_mlp(time)
        text_embeddings = self.embedding_mlp(text_emb.detach())

        stacked_embeddings = torch.cat((position_embeddings, text_embeddings), dim=-1)

        conditional_embeddings = self.combine_conditional(stacked_embeddings)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, conditional_embeddings)
            h.append(x)

            x = block2(x, conditional_embeddings)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, conditional_embeddings)
        x = self.mid_attn(x)
        x = self.mid_block2(x, conditional_embeddings)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, conditional_embeddings)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, conditional_embeddings)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, conditional_embeddings)
        return self.final_conv(x)
