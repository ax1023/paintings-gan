import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from models import get_generator 


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "checkpoints/gan_checkpoint_epoch_130.pth" 
LATENT_DIM = 100
N_G = 64
OUTPUT_FILE = "generated_portrait.png"

def generate():

    net_G = get_generator(N_G, LATENT_DIM).to(DEVICE)
    

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    net_G.load_state_dict(checkpoint['net_G_state_dict'])
    net_G.eval() 
    
    print(f"Modello caricato correttamente da {MODEL_PATH}")


    noise = torch.randn(16, LATENT_DIM, 1, 1, device=DEVICE)


    with torch.no_grad():
        fake_images = net_G(noise).cpu()
        

    fake_images = (fake_images / 2) + 0.5


    grid = vutils.make_grid(fake_images, nrow=4, padding=2, normalize=False)
    vutils.save_image(grid, OUTPUT_FILE)
    
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Immagini Generate")
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()
    
    print(f"Immagine salvata come {OUTPUT_FILE}")

if __name__ == "__main__":
    generate()