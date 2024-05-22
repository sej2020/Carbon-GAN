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
        self.data_generator = torch.nn.RNN(metadata_dim, 1, nonlinearity='relu') 
        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(window_size, 12),
            torch.nn.ReLU(),
            torch.nn.Linear(12, 1),
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

        dataloader = DataLoader(
            data, 
            batch_size = (self.window_size + self.cfg.batch_size) * self.cfg.k, 
            drop_last = True, shuffle = False
            )

        for epoch_n in pbar:
            epoch_loss = []
            dataloader.dataset = dataloader.dataset.roll(epoch_n % self.window_size)
            for (batch_m, batch_d) in dataloader: # ( [dims, (window+batch)*k] , [1, (window+batch)*k] )
                for k in range(self.cfg.k):
                    k_batch_m_lst = []
                    k_batch_d_lst = []
                    for b in range(self.cfg.batch_size):
                        k_batch_m_lst.append(batch_m[:, k*(self.window_size + self.cfg.batch_size) + b]) # [dims, 1]
                        k_batch_d_lst.append(batch_d[:, k*(self.window_size+self.cfg.batch_size) + b : k*(self.window_size+self.cfg.batch_size) + b + self.window_size]) # [window, 1]
                    k_batch_m = torch.stack(k_batch_m_lst, dim=1).squeeze() # [dims, batch]
                    k_batch_d = torch.stack(k_batch_d_lst, dim=1).squeeze() # [window, batch]

                    k_batch_x = torch.cat((k_batch_m, k_batch_d), dim=0) # [dims+window, batch]
                    
                    z_mD = torch.randn(1, self.cfg.batch_size, self.metadata_dim) # [1, batch, dims]
                    z_dD = torch.randn(self.window_size, self.cfg.batch_size, 1)  # [window, batch, 1]
                    with torch.no_grad():
                        # [dims, batch] -> [dims, batch]
                        g_mD = self.metadata_generator.forward(z_mD.squeeze().T)
                        # [window, batch, dims] + [window, batch, 1] = [window, batch, dims+1] -> [window, batch]
                        g_dD = self.data_generator.forward(torch.cat((z_mD.repeat(self.window_size, 1, 1),z_dD), dim=2)).squeeze()
                        # [dims, batch] + [window, batch] = [dims+window, batch]
                        g_xD = torch.cat((g_mD, g_dD), dim=0)
                    # [dims+window, batch] -> [batch]
                    d_gxD = self.discriminator.forward(g_xD)
                    # [dims+window, batch] -> [batch]
                    d_xD = self.discriminator.forward(k_batch_x)
                    optimizer_D.zero_grad()
                    loss_D = criterion_D(d_xD, d_gxD)
                    loss_D.backward()
                    optimizer_D.step()

                z_mG = torch.randn(1, self.cfg.batch_size, self.metadata_dim) # [1, batch, dims]
                z_dG = torch.randn(self.window_size, self.cfg.batch_size, 1)  # [window, batch, 1]
                # [dims, batch] -> [dims, batch]
                g_mG = self.metadata_generator.forward(z_mG.squeeze().T)
                # [window, batch, dims] + [window, batch, 1] = [window, batch, dims+1] -> [window, batch]
                g_dG = self.data_generator.forward(torch.cat((z_mG.repeat(self.window_size, 1, 1),z_dG), dim=2)).squeeze()
                # [dims, batch] + [window, batch] = [dims+window, batch]
                g_xG = torch.cat((g_mG, g_dG), dim=0)

                with torch.no_grad():
                    # [dims+window, batch] -> [batch]
                    d_gxG = self.discriminator.forward(g_xG)
                optimizer_Gm.zero_grad()
                optimizer_Gd.zero_grad()
                loss_Gm = criterion_G(d_gxG)
                loss_Gd = criterion_G(d_gxG)
                loss_Gm.backward()
                loss_Gd.backward()
                optimizer_Gm.step()
                optimizer_Gd.step()

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
            
            pbar.set_description(f"Loss: {sum(epoch_loss)/len(epoch_loss)}")

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