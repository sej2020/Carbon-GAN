"""
Simple GAN implementation to use as baseline in forthcoming experiments
"""
import torch
from src.utils.data import CarbonDataset
from torch.utils.data import DataLoader

class SimpleGAN(torch.nn.Module):
    """
    Simple GAN comprising and MLP generator, an RNN generator, and an MLP discriminator. This model is intended to be used as a 
        baseline with which future models can be compared.
    
    Attributes:
        metadata_generator: An MLP rmetadata generator
        data_generator: An RNN data generator
        discriminator: An MLP discriminator
    ------------
        (Established in methods):
        cfg: configuration for training
    """
    def __init__(self):
        """
        Initializes Simple GAN model.
        """
        ### placeholders
        super(SimpleGAN, self).__init__()
        self.metadata_generator = torch.nn.Sequential(
            torch.nn.Linear(100, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 784),
            torch.nn.Sigmoid()
        )
        self.data_generator = torch.nn.Sequential(
            torch.nn.Linear(100, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 784),
            torch.nn.Sigmoid()
        )
        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(784, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid()
        )
    

    def train(self, cfg):
        """
        Trains the GAN model.

        Args:
            cfg: configuration for training
        """
        # Notes on variable naming scheme:
        # z: noise, m: relevant to metadata, d: relevant to data, x: metadata+data
        # g: passed through generator, d: passed through discriminator
        # G: relevant to training generator, D: relevant to training discriminator
        self.cfg = cfg
        
        data = CarbonDataset(self.cfg.region, self.cfg.elec_source)
        dataloader = DataLoader(data, batch_size=self.cfg.minibatch_size*self.cfg.k, drop_last=True, shuffle=False)
        for epoch in range(self.cfg.n_epochs):
            dataloader.dataset = dataloader.dataset.roll(epoch % window_size)
            for k_batch in dataloader: # one step of training
                for k in range(self.cfg.k):
                    x_minibatch = k_batch[ k*self.cfg.minibatch_size : (k+1)*self.cfg.minibatch_size ]
                    z_mD = torch.randn(x_minibatch.size(0), weather_dim + time_dim)
                    z_dD = torch.randn(x_minibatch.size(0), 24)
                    with torch.no_grad():
                        g_mD = self.metadata_generator.forward(z_mD)
                        g_dD = self.data_generator.forward(z_dD)
                        g_xD = torch.cat((g_mD, g_dD), dim=1)
                    d_xD = self.discriminator.forward(x_minibatch)
                    d_gxD = self.discriminator.forward(g_xD)
                    criterion_D = lambda d_xD, d_gxD: -torch.mean(torch.log(d_xD) + torch.log(1 - d_gxD))
                    loss_D = criterion_D(d_xD, d_gxD)
                    loss_D.backward()
                    optimizer_D.step()
                z_mG = torch.randn(x_minibatch.size(0), weather_dim + time_dim)
                z_dG = torch.randn(x_minibatch.size(0), 24)
                g_mG = self.metadata_generator.forward(z_mG)
                g_dG = self.data_generator.forward(z_dG)
                g_xG = torch.cat((g_mG, g_dG), dim=1)
                with torch.no_grad():
                    d_gxG = self.discriminator.forward(g_xG)
                criterion_G = lambda d_gxG: -torch.mean(torch.log(d_gxG))
                loss_G = criterion_G(d_gxG)
                loss_G.backward()
                optimizer_G.step()


