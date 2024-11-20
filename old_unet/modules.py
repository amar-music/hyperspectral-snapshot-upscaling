# Based on Saharia et al. (2021). https://arxiv.org/abs/2104.07636
# Adapted from https://github.com/aditya-nutakki/SR3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from math import log
import cv2 as cv


# Create paired Dataset
class PairedDataset(Dataset):
    def __init__(self, train_folder, val_folder, transform=None):
        self.train_folder = train_folder
        self.val_folder = val_folder
        self.transform = transform
        self.train_files = sorted(os.listdir(train_folder))
        self.val_files = sorted(os.listdir(val_folder))
        self.paired_files = [(train_file, val_file) for train_file, val_file in zip(self.train_files, self.val_files) if train_file == val_file]

    def __len__(self):
        return len(self.paired_files)

    def __getitem__(self, idx):
        train_file, val_file = self.paired_files[idx]
        train_image = np.load(os.path.join(self.train_folder, train_file))
        val_image = np.load(os.path.join(self.val_folder, val_file))
        
        # Interpolate train image to (480, 912, 24)
        train_image = cv.resize(train_image, (912, 480), interpolation=cv.INTER_LINEAR)

        # Make both images square (480, 480)
        train_image = train_image[:, 216:696]
        val_image = val_image[:, 216:696]

        # Change to channel first format
        train_image = np.moveaxis(train_image, -1, 0)
        val_image = np.moveaxis(val_image, -1, 0)
        
        
        if self.transform:
            train_image = self.transform(train_image)
            val_image = self.transform(val_image)
        
        return train_image, val_image

# # Create paired Dataset
# class PairedDataset(Dataset):
#     def __init__(self, train_folder, val_folder, transform=None):
#         self.train_folder = train_folder
#         self.val_folder = val_folder
#         self.transform = transform
#         self.train_files = sorted(os.listdir(train_folder))
#         self.val_files = sorted(os.listdir(val_folder))
#         self.paired_files = [(train_file, val_file) for train_file, val_file in zip(self.train_files, self.val_files) if train_file == val_file]

#     def __len__(self):
#         return len(self.paired_files)

#     def __getitem__(self, idx):
#         train_file, val_file = self.paired_files[idx]
#         train_image = np.load(os.path.join(self.train_folder, train_file))
#         val_image = np.load(os.path.join(self.val_folder, val_file))
        
#         # Interpolate train image to (480, 912, 24)
#         train_image = cv.resize(train_image, (912, 480), interpolation=cv.INTER_LINEAR)
        
#         if self.transform:
#             train_image = self.transform(train_image)
#             val_image = self.transform(val_image)
        
#         return train_image, val_image




# Define U-Net architecture with self-attention
class UNet(nn.Module):
    def __init__(self, input_channels=24, output_channels=24, time_steps=512):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.time_steps = time_steps

        # Encoder blocks for downsampling
        self.e1 = encoder_block(self.input_channels, 64, time_steps=self.time_steps)
        self.e2 = encoder_block(64, 128, time_steps=self.time_steps)
        self.e3 = encoder_block(128, 256, time_steps=self.time_steps)
        # self.da3 = AttnBlock(256) # Attention blocks to enhance focus on features
        # self.da3 = nn.MultiheadAttention(embed_dim=256, num_heads=4) # Attention blocks to enhance focus on features
        self.e4 = encoder_block(256, 512, time_steps=self.time_steps)
        # self.da4 = AttnBlock(512)
        # self.da4 = nn.MultiheadAttention(embed_dim=512, num_heads=4)


        # Bottleneck connecting encoder and decoder
        self.b = conv_block(512, 1024, time_steps=self.time_steps) # bottleneck
        # self.ba1 = AttnBlock(1024) # Further refine features using attention
        self.ba1 = nn.MultiheadAttention(embed_dim=1024, num_heads=4) # Further refine features using attention

        # Decoder blocks for upsampling
        self.d1 = decoder_block(1024, 512, time_steps=self.time_steps)
        # self.ua1 = AttnBlock(512)
        # self.ua1 = nn.MultiheadAttention(embed_dim=512, num_heads=4)
        self.d2 = decoder_block(512, 256, time_steps=self.time_steps)
        # self.ua2 = AttnBlock(256)
        # self.ua2 = nn.MultiheadAttention(embed_dim=256, num_heads=4)
        self.d3 = decoder_block(256, 128, time_steps=self.time_steps)
        self.d4 = decoder_block(128, 64, time_steps=self.time_steps)

        # Output layer
        self.outputs = nn.Conv2d(64, self.output_channels, kernel_size=1, padding=0)


    # Define forward pass (process that computes output from input)
    def forward(self, inputs, t=None):

        # Downsampling block (s represents skip connections, p represents downsampled feature maps)
        s1, p1 = self.e1(inputs, t)
        s2, p2 = self.e2(p1, t)
        s3, p3 = self.e3(p2, t)
        # p3 = self.da3(p3) # Attention block
        s4, p4 = self.e4(p3, t)
        # p4 = self.da4(p4)

        # Bottleneck
        b = self.b(p4, t)
        # b = self.ba1(b)

        # Upsampling block
        d1 = self.d1(b, s4, t)
        # d1 = self.ua1(d1)
        d2 = self.d2(d1, s3, t)
        # d2 = self.ua2(d2)
        d3 = self.d3(d2, s2, t)
        d4 = self.d4(d3, s1, t)

        # Output layer
        outputs = self.outputs(d4)
        return outputs




# class AttnBlock(nn.Module):
#     def __init__(self, embedding_dims, num_heads = 4) -> None:
#         super().__init__()
        
#         self.embedding_dims = embedding_dims
#         self.ln = nn.LayerNorm(embedding_dims)

#         self.mhsa = MultiHeadSelfAttention(embedding_dims = embedding_dims, num_heads = num_heads)

#         self.ff = nn.Sequential(
#             nn.LayerNorm(self.embedding_dims),
#             nn.Linear(self.embedding_dims, self.embedding_dims),
#             nn.GELU(),
#             nn.Linear(self.embedding_dims, self.embedding_dims),
#         )

#     def forward(self, x):
#         bs, c, sz, _ = x.shape
#         x = x.view(-1, self.embedding_dims, sz * sz).swapaxes(1, 2) # is of the shape (bs, sz**2, self.embedding_dims)
#         print(x.shape)
#         x_ln = self.ln(x)
#         _, attention_value = self.mhsa(x_ln, x_ln, x_ln)
#         attention_value = attention_value + x
#         attention_value = self.ff(attention_value) + attention_value
#         return attention_value.swapaxes(2, 1).view(-1, c, sz, sz)


# class MultiHeadSelfAttention(nn.Module):
#     def __init__(self, embedding_dims, num_heads = 4) -> None:
#         super().__init__()
#         self.embedding_dims = embedding_dims
#         self.num_heads = num_heads

#         assert self.embedding_dims % self.num_heads == 0, f"{self.embedding_dims} not divisible by {self.num_heads}"
#         self.head_dim = self.embedding_dims // self.num_heads

#         self.wq = nn.Linear(self.head_dim, self.head_dim)
#         self.wk = nn.Linear(self.head_dim, self.head_dim)
#         self.wv = nn.Linear(self.head_dim, self.head_dim)

#         self.wo = nn.Linear(self.embedding_dims, self.embedding_dims)

#     def attention(self, q, k, v):
#         # no need for a mask
#         attn_weights = F.softmax((q @ k.transpose(-1, -2))/self.head_dim**0.5, dim = -1)
#         return attn_weights, attn_weights @ v        

#     def forward(self, q, k, v):
#         bs, img_sz, c = q.shape

#         q = q.view(bs, img_sz, self.num_heads, self.head_dim).transpose(1, 2)
#         k = k.view(bs, img_sz, self.num_heads, self.head_dim).transpose(1, 2)
#         v = v.view(bs, img_sz, self.num_heads, self.head_dim).transpose(1, 2)
#         # q, k, v of the shape (bs, self.num_heads, img_sz**2, self.head_dim)

#         q = self.wq(q)
#         k = self.wk(k)
#         v = self.wv(v)

#         attn_weights, o = self.attention(q, k, v) # of shape (bs, num_heads, img_sz**2, c)
        
#         o = o.transpose(1, 2).contiguous().view(bs, img_sz, self.embedding_dims)
#         o = self.wo(o)

#         return attn_weights, o






# Encoder Block for downsampling
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c, time_steps, activation = "relu"):
        super().__init__()

        # Convolutional block: convolution followed by batch normalization and activation
        self.conv = conv_block(in_c, out_c, time_steps = time_steps, activation = activation, embedding_dims = out_c)

        # Max pooling layer with a kernel size of (2, 2) to downsample the input
        self.pool = nn.MaxPool2d((2, 2))


    def forward(self, inputs, time = None):
        x = self.conv(inputs, time)
        p = self.pool(x)
        return x, p


# Decoder Block for upsampling
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, time_steps, activation = "relu"):
        super().__init__()

        # Upsampling layer with a kernel size of (2, 2) to upsample the input
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)

        # Convolutional block: concatenate upsampled input with skip connection, then apply convolution
        self.conv = conv_block(out_c+out_c, out_c, time_steps = time_steps, activation = activation, embedding_dims = out_c)


    def forward(self, inputs, skip, time = None):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x, time)
        return x



# Gamma Encoding for positional encoding
class GammaEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, dim)
        self.act = nn.LeakyReLU()
    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(log(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return self.act(self.linear(encoding))


# Double Conv Block
class conv_block(nn.Module):
    def __init__(self, in_c, out_c, time_steps = 1000, activation = "relu", embedding_dims = None):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.embedding_dims = embedding_dims if embedding_dims else out_c
        
        # self.embedding = nn.Embedding(time_steps, embedding_dim = self.embedding_dims)
        self.embedding = GammaEncoding(self.embedding_dims)
        # switch to nn.Embedding if you want to pass in timestep instead; but note that it should be of dtype torch.long
        self.act = nn.ReLU() if activation == "relu" else nn.SiLU()
        
    def forward(self, inputs, time = None):
        time_embedding = self.embedding(time).view(-1, self.embedding_dims, 1, 1)
        # print(f"time embed shape => {time_embedding.shape}")
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = x + time_embedding
        return x




# Diffusion model
class DiffusionModel(nn.Module):
    def __init__(self, time_steps, 
                 beta_start = 10e-4, 
                 beta_end = 0.02,
                 image_dims = (24, 480, 480)):
        
        super().__init__()
        self.time_steps = time_steps
        self.image_dims = image_dims
        c, h, w = self.image_dims
        self.img_size, self.input_channels = h, c
        self.betas = torch.linspace(beta_start, beta_end, self.time_steps)
        self.alphas = 1 - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim = -1)
        self.model = UNet(input_channels = 2*c, output_channels = c, time_steps = self.time_steps)

    def add_noise(self, x, ts):
        # 'x' and 'ts' are expected to be batched
        noise = torch.randn_like(x)
        # print(x.shape, noise.shape)
        noised_examples = []
        for i, t in enumerate(ts):
            alpha_hat_t = self.alpha_hats[t]
            noised_examples.append(torch.sqrt(alpha_hat_t)*x[i] + torch.sqrt(1 - alpha_hat_t)*noise[i])
        return torch.stack(noised_examples), noise

    def forward(self, x, t):
        return self.model(x, t)