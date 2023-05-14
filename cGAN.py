import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from torch.autograd import Variable
from utils import get_result_images


class ConditionalGAN():
    def __init__(self, 
                 gen, 
                 disc, 
                 max_epoch,
                 batch_size,
                 lr,
                 dataloader,
                 trial,
                 current_epoch,
                 latent_dim,
                 num_class,
                 device
                 ) -> None:
        
        self.gen = gen
        self.disc = disc
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.latent_dim = latent_dim
        self.dataloader = dataloader
        self.trial = trial
        self.current_epoch= current_epoch
        self.num_class = num_class
        self.device = device
        self.checkpoint_save_dir = './checkpoint'

        self.g_optim = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999))
        self.d_optim = torch.optim.Adam(self.disc.parameters(), lr=lr, betas=(0.5, 0.999))

        self.loss_func = nn.BCELoss()

        self.real_label = torch.ones((self.batch_size, 1), device=self.device)
        self.fake_label = torch.zeros((self.batch_size, 1), device=self.device)

        self.fixed_latent = torch.randn((10, self.latent_dim), device=self.device)
        self.fixed_label = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])



    def train(self):
        for epoch in range(1, self.max_epoch+1):
            if self.current_epoch >= epoch:
                continue
            self.gen.train()
            self.disc.train()
            for idx, (data, label) in enumerate(tqdm(self.dataloader)):
                # Latent vector(z)
                latent = torch.randn([self.batch_size, self.latent_dim], requires_grad=True, device=self.device)
                # Real Data
                real_data = Variable(data, requires_grad=True).to(self.device)
                # Label Encoding
                encoded_label = F.one_hot(label, num_classes=self.num_class).to(self.device)

                # ===== Train Discriminator =====
                self.disc.zero_grad()
                
                fake_data = self.gen(latent, encoded_label)
                
                real_pred = self.disc(real_data, encoded_label)
                fake_pred = self.disc(fake_data, encoded_label)

                real_loss = self.loss_func(real_pred, self.real_label)
                fake_loss = self.loss_func(fake_pred, self.fake_label)

                d_loss = (real_loss + fake_loss) / 2.

                d_loss.backward()
                self.d_optim.step()

                # ===== Train Generator =====
                self.gen.zero_grad()

                fake_data = self.gen(latent, encoded_label)

                fake_pred = self.disc(fake_data, encoded_label)

                g_loss = self.loss_func(fake_pred, self.real_label)

                g_loss.backward()
                self.g_optim.step()

                print(f'Epoch {epoch}/{self.max_epoch}, G_Loss: {g_loss.data}, D_Loss: {d_loss.data}')

            torch.save({
                'gen': self.gen.state_dict(),
                'disc': self.disc.state_dict()
            }, f'{self.checkpoint_save_dir}/cp_{self.trial}_{epoch}.pt')

            result = get_result_images(self.gen, self.fixed_latent, self.fixed_label, self.num_class, self.device)
            grid = make_grid(result)
            save_image(grid, f'./result/cp_{self.trial}/{str(epoch).zfill(3)}.jpg')


                
        






