import torch
import torch.nn as nn


class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim=100, label_dim=10, channel_out=3, size=(28, 28)):
        super(ConditionalGenerator, self).__init__()
        self.channel_out = channel_out
        self.size = size

        self.gen = nn.Sequential(
            nn.Linear(latent_dim+label_dim, 128),
            nn.InstanceNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.InstanceNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.InstanceNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, channel_out*size[0]*size[1]),
            nn.Tanh()
        )

    def forward(self, x, y):
        latent = torch.cat([x, y], 1)
        out = self.gen(latent)
        return out.view(x.size(0), self.channel_out, self.size[0], self.size[1])


class ConditionalDiscriminator(nn.Module):
    def __init__(self, label_dim=10, channel_in=3, size=(28, 28)):
        super(ConditionalDiscriminator, self).__init__()
        self.channel_in = channel_in
        self.size = size

        self.disc = nn.Sequential(
            nn.Linear(self.channel_in*self.size[0]*self.size[1]+label_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        x = x.view(x.size(0), -1)
        latent = torch.cat([x, y], 1)
        out = self.disc(latent)
        return out
        





