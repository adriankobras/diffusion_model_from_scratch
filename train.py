from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from IPython.display import HTML
from utilities import *
from model import *


# hyperparameters

# diffusion hyperparameters
timesteps = 500
beta1 = 1e-4
beta2 = 0.02

# network hyperparameters
device = torch.device("cuda:0" if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
# device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu'))
# device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
n_feat = 256 # 64 hidden dimension feature
n_cfeat = 5 # context vector is of size 5
height = 28 # 16x16 image
save_dir = './weights/'
data_dir = './data/'

# training hyperparameters
batch_size = 100
n_epoch = 500
lrate=1e-3

if __name__ == '__main__':
    # construct DDPM noise schedule
    b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()    
    ab_t[0] = 1

    # construct model
    nn_model = ContextUnet(in_channels=3, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)

    # load dataset and construct optimizer
    dataset = CustomDataset(data_dir + "images.npy", data_dir + "labels.npy", transform, null_context=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)

    # helper function: perturbs an image to a specified noise level
    def perturb_input(x, t, noise):
        return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise


    # training without context info

    # set into train mode
    nn_model.train()

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        
        # linearly decay learning rate
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)
        
        pbar = tqdm(dataloader, mininterval=2 )
        for x, _ in pbar:   # x: images
            optim.zero_grad()
            x = x.to(device)
            
            # perturb data
            noise = torch.randn_like(x)
            t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device) 
            x_pert = perturb_input(x, t, noise)
            
            # use network to recover noise
            pred_noise = nn_model(x_pert, t / timesteps)
            
            # loss is mean squared error between the predicted and true noise
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()
            
            optim.step()

        # save model periodically
        if ep%4==0 or ep == int(n_epoch-1):
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save(nn_model.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")