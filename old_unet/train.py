import torchshow
from old_unet.modules import *
# from helpers import *
from time import time
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

# def train_ddpm(time_steps = 2000, epochs = 20, batch_size = 4, device = "cuda", image_dims = (480, 912, 24), low_res_dims = (210, 399, 24)):
def train_ddpm(time_steps = 5, epochs = 2, batch_size = 4, device = "cuda"):
    ddpm = DiffusionModel(time_steps = time_steps)
    
    # ds = SRDataset("/mnt/d/work/datasets/nature/x128/all", hr_sz = hr_sz, lr_sz = lr_sz)
    paired_dataset = PairedDataset("data/Processed/train", "data/Processed/val")
    loader = DataLoader(paired_dataset, batch_size = batch_size, shuffle = True, drop_last = True, num_workers = 2)

    opt = torch.optim.Adam(ddpm.model.parameters(), lr = 1e-3)
    criterion = nn.MSELoss(reduction="mean")

    ddpm.model.to(device)

    for ep in range(epochs):
        ddpm.model.train()
        print(f"Epoch {ep}:")
        losses = []
        stime = time()
        
        for i, (x, y) in enumerate(loader):
            # 'y' represents the high-resolution target image, while 'x' represents the low-resolution image to be conditioned upon.
            
            bs = y.shape[0]
            x, y = x.to(device).float(), y.to(device).float()

            print(i)

            ts = torch.randint(low = 1, high = ddpm.time_steps, size = (bs, ))
            gamma = ddpm.alpha_hats[ts].to(device)
            ts = ts.to(device = device)


            # Select the 10th channel
            y_channel_10 = y[:, 10, :, :].cpu().numpy()
            torchshow.save(y_channel_10[0], f"./y_channel_10_pre.jpeg")
            x_channel_10 = x[:, 10, :, :].cpu().numpy()
            torchshow.save(x_channel_10[0], f"./x_channel_10_pre.jpeg")

            y, target_noise = ddpm.add_noise(y, ts)
            y = torch.cat([x, y], dim = 1)

            # Select the 10th channel
            y_channel_10 = y[:, 10, :, :].cpu().numpy()
            torchshow.save(y_channel_10[0], f"./y_channel_10.jpeg")
            x_channel_10 = x[:, 10, :, :].cpu().numpy()
            torchshow.save(x_channel_10[0], f"./x_channel_10.jpeg")


            predicted_noise = ddpm.model(y, gamma)
            loss = criterion(target_noise, predicted_noise)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            losses.append(loss.item())

            # if i % 250 == 0:
            print(f"Loss: {loss.item()}; step {i}; epoch {ep}")

        ftime = time()
        print(f"Epoch trained in {ftime - stime}s; Avg loss => {sum(losses)/len(losses)}")

        torch.save(ddpm.state_dict(), f"./sr_ep_{ep}.pt")
        print()
        # The above training loop saves a model at the end of every epoch and prints the loss after every 250 steps.

def sample(model, lr_img, device = "cuda"):
    # lr_img is expected to be batched
    # set to eval mode
    model.to(device)
    model.eval()
    
    stime = time()
    with torch.no_grad():
    
        y = torch.randn_like(lr_img, device = device)
        lr_img = lr_img.to(device)
        for i, t in enumerate(range(model.time_steps - 1, 0 , -1)):
            alpha_t, alpha_t_hat, beta_t = model.alphas[t], model.alpha_hats[t], model.betas[t]
    
            t = torch.tensor(t, device = device).long()
            pred_noise = model(torch.cat([lr_img, y], dim = 1), alpha_t_hat.view(-1).to(device))
            y = (torch.sqrt(1/alpha_t))*(y - (1-alpha_t)/torch.sqrt(1 - alpha_t_hat) * pred_noise)
            if t > 1:
                noise = torch.randn_like(y)
                y = y + torch.sqrt(beta_t) * noise
            
    ftime = time()
    torchshow.save(y[:, 10, :, :], f"./sr_sample.jpeg")
    print(f"Done denoising in {ftime - stime}s ")


TRAIN = False

if __name__ == "__main__":
    if TRAIN == True:
        train_ddpm(time_steps=5, epochs=2, batch_size=4)
    else:
        model = DiffusionModel(time_steps=70)
        model.load_state_dict(torch.load("./sr_ep_1.pt", weights_only=True))
        model.eval()

        # Paths to training and validation images
        train_folder = "data/Processed/train"
        val_folder = "data/Processed/val"

        # Example usage
        paired_dataset = PairedDataset(train_folder, val_folder)
        loader = DataLoader(paired_dataset, batch_size=1, shuffle = True, drop_last = True, num_workers = 2)

        # Load image as tensor
        lr_img = torch.tensor([loader.dataset[38][0]]).float()
        print(lr_img.shape)
        # Save image
        torchshow.save(lr_img[0, 10, :, :], f"./lr_sample.jpeg")

        # Random tensor
        lr_img2 = torch.randn(1, 24, 480, 480)

        sample(model, lr_img)
