# Based on Saharia et al. (2021). https://arxiv.org/abs/2104.07636
# Adapted from https://github.com/aditya-nutakki/SR3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from math import log

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
        self.da3 = nn.MultiheadAttention(embed_dim=256, num_heads=4) # Attention blocks to enhance focus on features
        self.e4 = encoder_block(256, 512, time_steps=self.time_steps)
        # self.da4 = AttnBlock(512)
        self.da4 = nn.MultiheadAttention(embed_dim=512, num_heads=4)


        # Bottleneck connecting encoder and decoder
        self.b = conv_block(512, 1024, time_steps=self.time_steps) # bottleneck
        # self.ba1 = AttnBlock(1024) # Further refine features using attention
        self.ba1 = nn.MultiheadAttention(embed_dim=1024, num_heads=4) # Further refine features using attention

        # Decoder blocks for upsampling
        self.d1 = decoder_block(1024, 512, time_steps=self.time_steps)
        # self.ua1 = AttnBlock(512)
        self.ua1 = nn.MultiheadAttention(embed_dim=512, num_heads=4)
        self.d2 = decoder_block(512, 256, time_steps=self.time_steps)
        # self.ua2 = AttnBlock(256)
        self.ua2 = nn.MultiheadAttention(embed_dim=256, num_heads=4)
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
        p3 = self.da3(p3) # Attention block
        s4, p4 = self.e4(p3, t)
        p4 = self.da4(p4)

        # Bottleneck
        b = self.b(p4, t)
        b = self.ba1(b)

        # Upsampling block
        d1 = self.d1(b, s4, t)
        d1 = self.ua1(d1)
        d2 = self.d2(d1, s3, t)
        d2 = self.ua2(d2)
        d3 = self.d3(d2, s2, t)
        d4 = self.d4(d3, s1, t)

        # Output layer
        outputs = self.outputs(d4)
        return outputs




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
                 image_dims = (24, 210, 399)):
        
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