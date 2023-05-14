import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from cGAN import ConditionalGAN
from model import ConditionalGenerator, ConditionalDiscriminator
from utils import *
import os


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Current Device: ", device)

    # Set args
    batch_size = 32
    max_epoch = 50
    lr = 1e-4
    latent_dim = 100
    trial = 1
    current_epoch = 0
    num_class = 10
    label_dim = 10
    checkpoint_dir = f'./checkpoint/cp_{trial}_{current_epoch}.pt'
    result_dir = f'./result/cp_{trial}'

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # Define Generator and Discriminator
    gen = ConditionalGenerator(latent_dim, label_dim, 1).to(device)
    disc = ConditionalDiscriminator(label_dim, 1).to(device)

    if os.path.exists(checkpoint_dir):
        state_dict = torch.load(checkpoint_dir)
        gen.load_state_dict(state_dict['gen'])
        disc.load_state_dict(state_dict['disc'])
        print("Checkpoint Loaded")
    else:
        gen.apply(weights_init)
        disc.apply(weights_init)
        print("New Model")

    
    # Define Transform
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])


    # Define Dataloder
    dataset = MNIST(root='./', train=True, transform=tf, download=True)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True,
                            num_workers=4)


    gan_trainer = ConditionalGAN(gen=gen,
                      disc=disc,
                      max_epoch=max_epoch,
                      batch_size=batch_size,
                      lr=lr,
                      dataloader=dataloader,
                      trial=trial,
                      current_epoch=current_epoch,
                      latent_dim=latent_dim,
                      num_class=num_class,
                      device=device
                      )
    
    gan_trainer.train()



if __name__=='__main__':
    main()


