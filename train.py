import torch
import torchvision
from torch import nn
import matplotlib.pyplot as plt
import os
from models import get_generator, get_discriminator
from utils import load_checkpoint


device = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "./data"  
BATCH_SIZE = 256
LATENT_DIM = 100
LR = 0.0002
EPOCHS = 130
RESUME_PATH = None 

transformer = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = torchvision.datasets.ImageFolder(DATA_DIR, transform=transformer)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

net_G = get_generator(64, LATENT_DIM).to(device)
net_D = get_discriminator(64).to(device)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

net_G.apply(weights_init)
net_D.apply(weights_init)


trainer_hp = {'lr': LR, 'betas': [0.5, 0.999]}
opt_D = torch.optim.Adam(net_D.parameters(), **trainer_hp)
opt_G = torch.optim.Adam(net_G.parameters(), **trainer_hp)

start_epoch = 0
if RESUME_PATH and os.path.exists(RESUME_PATH):
    start_epoch = load_checkpoint(RESUME_PATH, net_G, net_D, opt_G, opt_D, device)


loss_fn = nn.BCEWithLogitsLoss()
fixed_noise = torch.randn(16, LATENT_DIM, 1, 1, device=device)

for epoch in range(start_epoch, EPOCHS):
    for i, (X, _) in enumerate(data_loader):
        b_size = X.shape[0]
        real = X.to(device)
        

        opt_D.zero_grad()
        label_real = torch.ones(b_size, device=device)
        label_fake = torch.zeros(b_size, device=device)
        
        output_real = net_D(real).view(-1)
        loss_real = loss_fn(output_real, label_real)
        
        noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=device)
        fake = net_G(noise)
        output_fake = net_D(fake.detach()).view(-1)
        loss_fake = loss_fn(output_fake, label_fake)
        
        loss_d = (loss_real + loss_fake) / 2
        loss_d.backward()
        opt_D.step()


        opt_G.zero_grad()
        output_g = net_D(fake).view(-1)
        loss_g = loss_fn(output_g, label_real)
        loss_g.backward()
        opt_G.step()

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss D: {loss_d:.4f}, Loss G: {loss_g:.4f}")
    

    if (epoch + 1) % 5 == 0:
        torch.save({
            'epoch': epoch + 1,
            'net_G_state_dict': net_G.state_dict(),
            'net_D_state_dict': net_D.state_dict(),
            'opt_G_state_dict': opt_G.state_dict(),
            'opt_D_state_dict': opt_D.state_dict(),
        }, f'checkpoints/gan_epoch_{epoch+1}.pth')