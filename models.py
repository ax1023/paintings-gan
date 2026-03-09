import torch
from torch import nn

class G_block(nn.Module):
    def __init__(self, out_c, in_c, k=4, s=2, p=1):
        super(G_block, self).__init__()
        self.conv2dtrans = nn.ConvTranspose2d(in_c, out_c, k, s, p, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.batch_norm(self.conv2dtrans(x)))

def get_generator(n_G, latent_dim=100):
    return nn.Sequential(
        G_block(in_c=latent_dim, out_c=n_G*8, s=1, p=0),
        G_block(in_c=n_G*8, out_c=n_G*4),
        G_block(in_c=n_G*4, out_c=n_G*2),
        G_block(in_c=n_G*2, out_c=n_G),
        nn.ConvTranspose2d(n_G, 3, 4, 2, 1, bias=False),
        nn.Tanh()
    )

class D_block(nn.Module):
    def __init__(self, in_c, out_c, k=4, s=2, p=1, alpha=0.2):
        super(D_block, self).__init__()
        self.conv2d = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_c)
        self.act = nn.LeakyReLU(alpha, inplace=True)

    def forward(self, x):
        return self.act(self.batch_norm(self.conv2d(x)))

def get_discriminator(n_D):
    return nn.Sequential(
        D_block(3, n_D),
        D_block(n_D, n_D*2),
        D_block(n_D*2, n_D*4),
        D_block(n_D*4, n_D*8),
        nn.Conv2d(in_channels=n_D*8, out_channels=1, kernel_size=4, bias=False)
    )