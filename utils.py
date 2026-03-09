import torch

def load_checkpoint(checkpoint_path, net_G, net_D, opt_G, opt_D, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net_G.load_state_dict(checkpoint['net_G_state_dict'])
    net_D.load_state_dict(checkpoint['net_D_state_dict'])
    opt_G.load_state_dict(checkpoint['opt_G_state_dict'])
    opt_D.load_state_dict(checkpoint['opt_D_state_dict'])
    return checkpoint['epoch']