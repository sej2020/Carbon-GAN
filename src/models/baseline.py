"""
Simple GAN implementation to use as baseline in forthcoming experiments
"""
import torch
import pathlib
import tqdm
from src.utils.data import CarbonDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from joblib import dump
import time

class SimpleGAN(torch.nn.Module):
    """
    Simple GAN comprising and MLP generator, an RNN generator, and an MLP discriminator. This model is intended to be used as a 
        baseline with which future models can be compared.
    
    Attributes:
        metadata_generator: An MLP rmetadata generator
        data_generator: An RNN data generator
        discriminator: An MLP discriminator
        metadata_dim: The dimension of the metadata
        window_size: The size of the historical data used for the RNN generator
    ------------
        (Established in methods):
        cfg: configuration for training
    """
    def __init__(self, metadata_dim: int, window_size: int):
        """
        Initializes Simple GAN model.

        Args:
            metadata_dim: The dimension of the metadata
            window_size: The size of the historical data used for the RNN generator
        """
        super().__init__()
        self.metadata_generator = torch.nn.Sequential(
            torch.nn.Linear(metadata_dim, metadata_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(metadata_dim, metadata_dim)
        )
        self.data_generator = torch.nn.Sequential(
            torch.nn.Linear(window_size, window_size),
            torch.nn.ReLU(),
            torch.nn.Linear(window_size, window_size)
        )
        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(metadata_dim+window_size, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid()
        )
        self.metadata_dim = metadata_dim
        self.window_size = window_size
    

    def train(self, cfg):
        """
        Trains the GAN model. To view the training progress, run the following command in the terminal:
        ```bash
        tensorboard --logdir=logs
        ```
        Clean up the logs directory after training is complete.

        Args:
            cfg: configuration for training
        """
        # Notes on variable naming scheme:
        # z: noise, m: relevant to metadata, d: relevant to data, x: metadata+data
        # g: passed through generator, d: passed through discriminator
        # G: relevant to training generator, D: relevant to training discriminator
        self.cfg = cfg

        # writing out a text file to the logging directory with the string of the trainer config
        hp_path = pathlib.Path(f"{self.cfg.logging_dir}/{self.cfg.run_name}")
        hp_path.mkdir(parents=True, exist_ok=True)
        with open(f"{hp_path}/trainer_config.txt", "w") as file:
            file.write(str(self.cfg))
        
        data = CarbonDataset(self.cfg.region, self.cfg.elec_source)
        # saving the fitted scalers
        scaler_path = pathlib.Path(f"{self.cfg.logging_dir}/{self.cfg.run_name}/scalers")
        scaler_path.mkdir(parents=True, exist_ok=True)
        dump(data.metadata_scaler, scaler_path / "metadata_scaler.joblib")
        dump(data.data_scaler, scaler_path / "data_scaler.joblib")

        optimizer_Gm = torch.optim.Adam(self.metadata_generator.parameters(), lr=self.cfg.lr)
        optimizer_Gd = torch.optim.Adam(self.data_generator.parameters(), lr=self.cfg.lr)
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.cfg.lr)
        criterion_D = lambda d_xD, d_gxD: -torch.mean(torch.log(d_xD) + torch.log(1 - d_gxD))
        criterion_G = lambda d_gxG: -torch.mean(torch.log(d_gxG))

        pbar = tqdm.tqdm(range(self.cfg.n_epochs), disable=self.cfg.disable_tqdm)
        writer_path = pathlib.Path(f"{self.cfg.logging_dir}/{self.cfg.run_name}/tensorboard")
        writer_path.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=writer_path)
        logging_steps = int(1 / self.cfg.logging_frequency)

        if self.cfg.lr_scheduler:
            scheduler_Gm = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_Gm, self.cfg.n_epochs)
            scheduler_Gd = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_Gd, self.cfg.n_epochs)
            scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, self.cfg.n_epochs)

        if self.cfg.resume_from_cpt:
            self.checkpoint_dict = torch.load(self.cfg.cpt_path)
            metadata_generator_dict = self.checkpoint_dict['Gm_state_dict']
            data_generator_dict = self.checkpoint_dict['Gd_state_dict']
            discriminator_dict = self.checkpoint_dict['D_state_dict']

            self.metadata_generator.load_state_dict(metadata_generator_dict)
            self.data_generator.load_state_dict(data_generator_dict)
            self.discriminator.load_state_dict(discriminator_dict)

            optimizer_Gm.load_state_dict(self.checkpoint_dict["Gm_optim_state_dict"])
            optimizer_Gd.load_state_dict(self.checkpoint_dict["Gd_optim_state_dict"])
            optimizer_D.load_state_dict(self.checkpoint_dict["D_optim_state_dict"])

            pbar = tqdm.tqdm(
                iterable=range(self.cfg.n_epochs),
                total=self.cfg.n_epochs,
                initial=self.checkpoint_dict["epoch"] + 1,
            )

        for epoch_n in pbar:
            epoch_loss_D = []
            epoch_loss_G = []
            if epoch_n % self.window_size == 0:
                data._unroll()
            else:
                data._roll(epoch_n % self.window_size)
            dataloader = DataLoader(
                data, 
                batch_size = (self.window_size + self.cfg.batch_size) * self.cfg.k, 
                drop_last = True, shuffle = False
            )
            for b_idx, (batch_m, batch_d) in enumerate(dataloader): # ( [(window+batch)*k, dims] , [(window+batch)*k, 1] )
                self.discriminator.train(True)
                for k in range(self.cfg.k):
                    k_batch_m_lst = []
                    k_batch_d_lst = []
                    for b in range(self.cfg.batch_size):
                        k_batch_m_lst.append(batch_m[k*(self.window_size + self.cfg.batch_size) + b, :]) # [1, dims]
                        k_batch_d_lst.append(batch_d[k*(self.window_size + self.cfg.batch_size) + b : k*(self.window_size+self.cfg.batch_size) + b + self.window_size, :]) # [window, 1]
                    k_batch_m = torch.stack(k_batch_m_lst, dim=0).squeeze() # [batch, dims]
                    k_batch_d = torch.stack(k_batch_d_lst, dim=1).squeeze().T # [batch, window]

                    k_batch_x = torch.cat((k_batch_m, k_batch_d), dim=1) # [batch, dims+window]

                    z_mD = torch.randn(self.cfg.batch_size, self.metadata_dim) # [batch, dims]
                    z_dD = torch.randn(self.cfg.batch_size, self.window_size)  # [batch, window]
                    with torch.no_grad():
                        # [batch, dims] -> [batch, dims]
                        g_mD = self.metadata_generator.forward(z_mD)
                        # [batch, window] -> [batch, window]
                        g_dD = self.data_generator.forward(z_dD)
                        # [batch, dims] + [batch, window] = [batch, dims+window]
                        g_xD = torch.cat((g_mD, g_dD), dim=1)
                    # [batch, dims+window] -> [batch]
                    d_gxD = self.discriminator.forward(g_xD)
                    # [batch, dims+window] -> [batch]
                    d_xD = self.discriminator.forward(k_batch_x)
                    optimizer_D.zero_grad()
                    loss_D = criterion_D(d_xD, d_gxD)
                    loss_D.backward()
                    optimizer_D.step()

                z_mG = torch.randn(self.cfg.batch_size, self.metadata_dim) # [batch, dims]
                z_dG = torch.randn(self.cfg.batch_size, self.window_size)  # [batch, window]
                # [batch, dims] -> [batch, dims]
                g_mG = self.metadata_generator.forward(z_mG)
                # [batch, window] -> [batch, window]
                g_dG = self.data_generator.forward(z_dG)
                # [batch, dims] + [batch, window] = [batch, dims+window]
                g_xG = torch.cat((g_mG, g_dG), dim=1)

                self.discriminator.train(False)
                # [batch, dims+window] -> [batch]
                d_gxG = self.discriminator.forward(g_xG)
                optimizer_Gm.zero_grad()
                optimizer_Gd.zero_grad()
                loss_G = criterion_G(d_gxG)
                loss_G.backward()

                # Monitoring gradients
                # gradient_magnitude_meta = 0
                # for layer in self.metadata_generator:
                #     if hasattr(layer, "weight"):
                #        gradient_magnitude_meta += torch.norm(layer.weight.grad).item()
                # gradient_magnitude_data = 0
                # for layer in self.data_generator:
                #     if hasattr(layer, "weight"):
                #         gradient_magnitude_data += torch.norm(layer.weight.grad).item()
                # if b_idx % 10 == 0:
                #     print(f"G_m gradient magnitude: {gradient_magnitude_meta}, G_d gradient magnitude: {gradient_magnitude_data}")
                #     time.sleep(0.2)

                optimizer_Gm.step()
                optimizer_Gd.step()

                epoch_loss_D.append(loss_D.item())
                epoch_loss_G.append(loss_G.item())

            ## still need to setup train/test/val split before this can be used
            # if (epoch_n+1) % logging_steps == 0:
            #     val_loss = self.evaluate(x_test, y_test, criterion, self.cfg.batch_size) # still need to setup train/test/val split
            #     writer.add_scalars(
            #         "Loss", 
            #         {"Training" : sum(epoch_loss)/len(epoch_loss), "Validation": sum(val_loss)/len(val_loss)}, 
            #         epoch_n
            #         )
            #     self.save_checkpoint({
            #         "epoch": epoch_n,
            #         "Gm_state_dict": self.metadata_generator.state_dict(),
            #         "Gd_state_dict": self.data_generator.state_dict(),
            #         "D_state_dict": self.discriminator.state_dict(),
            #         "Gm_optim_state_dict": optimizer_Gm.state_dict(),
            #         "Gd_optim_state_dict": optimizer_Gd.state_dict(),
            #         "D_optim_state_dict": optimizer_D.state_dict()
            #     })

            if self.cfg.lr_scheduler:
                scheduler_Gm.step()
                scheduler_Gd.step()
                scheduler_D.step()
            
            pbar.set_description(f"Disc. Loss: {sum(epoch_loss_D)/len(epoch_loss_D):.3}, Gen. Loss: {sum(epoch_loss_G)/len(epoch_loss_G):.3}")

        writer.flush()
        writer.close()


    def save_checkpoint(self, checkpoint_dict: dict):
        """
        Saves a model checkpoint to a file.

        Args:
            checkpoint_dict: dictionary of training information from checkpoint. Must contain an 'epoch' key.
        """
        checkpoint_path = pathlib.Path(f"{self.cfg.logging_dir}/{self.cfg.run_name}/checkpoints")
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        torch.save(
            checkpoint_dict,
            checkpoint_path / f"checkpt_e{checkpoint_dict['epoch']}.pt",
        )