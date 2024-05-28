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
# import time

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
    def __init__(self, metadata_dim: int, window_size: int, cpt_path: str = None):
        """
        Initializes Simple GAN model.

        Args:
            metadata_dim: The dimension of the metadata
            window_size: The size of the historical data used for the RNN generator
            cpt_path: path to a checkpoint file to use for weights initialization. If None, weights are initialized randomly.
        """
        super().__init__()
        self.metadata_generator = torch.nn.Sequential(
            torch.nn.Linear(metadata_dim, metadata_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(metadata_dim, metadata_dim)
        )
        self.data_generator = torch.nn.Sequential(
            torch.nn.Linear(metadata_dim + window_size, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, window_size)
        )
        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(metadata_dim + window_size, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid()
        )
        self.metadata_dim = metadata_dim
        self.window_size = window_size

        if cpt_path:
            self.checkpoint_dict = torch.load(cpt_path)
            metadata_generator_dict = self.checkpoint_dict['Gm_state_dict']
            data_generator_dict = self.checkpoint_dict['Gd_state_dict']
            discriminator_dict = self.checkpoint_dict['D_state_dict']

            self.metadata_generator.load_state_dict(metadata_generator_dict)
            self.data_generator.load_state_dict(data_generator_dict)
            self.discriminator.load_state_dict(discriminator_dict)

    

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
        
        training_data = CarbonDataset(self.cfg.region, self.cfg.elec_source)
        # saving the fitted scalers
        scaler_path = pathlib.Path(f"{self.cfg.logging_dir}/{self.cfg.run_name}/scalers")
        scaler_path.mkdir(parents=True, exist_ok=True)
        dump(training_data.metadata_scaler, scaler_path / "metadata_scaler.joblib")
        dump(training_data.data_scaler, scaler_path / "data_scaler.joblib")

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
                training_data._unroll()
            else:
                training_data._roll(epoch_n % self.window_size)
            dataloader = DataLoader(
                training_data, 
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
                        # [batch, dims] + [batch, window] = [batch, dims+window]
                        gz_xD = torch.cat((g_mD, z_dD), dim=1)
                        # [batch, dims+window] -> [batch, window]
                        g_dD = self.data_generator.forward(gz_xD)
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
                # [batch, dims] + [batch, window] = [batch, dims+window]
                gz_xG = torch.cat((g_mG, z_dG), dim=1)
                # [batch, dims+window] -> [batch, window]
                g_dG = self.data_generator.forward(gz_xG)
                # [batch, dims] + [batch, window] = [batch, dims+window]
                g_xG = torch.cat((g_mG, g_dG), dim=1)

                self.discriminator.train(False)
                # [batch, dims+window] -> [batch]
                d_gxG = self.discriminator.forward(g_xG)
                optimizer_Gm.zero_grad()
                optimizer_Gd.zero_grad()
                loss_G = criterion_G(d_gxG)
                loss_G.backward()

                if self.cfg.debug:
                    if b_idx % 10 == 0:
                        # print(f"Epoch: {epoch_n}, Batch: {b_idx}, Loss_D: {loss_D.item()}, Loss_G: {loss_G.item()}")
                        # Monitoring gradients
                        gradient_magnitude_meta = 0
                        for layer in self.metadata_generator:
                            if hasattr(layer, "weight"):
                                gradient_magnitude_meta += torch.norm(layer.weight.grad).item()
                        gradient_magnitude_data = 0
                        for layer in self.data_generator:
                            if hasattr(layer, "weight"):
                                gradient_magnitude_data += torch.norm(layer.weight.grad).item()
                        print(f"G_m gradient magnitude: {gradient_magnitude_meta}, G_d gradient magnitude: {gradient_magnitude_data}")

                optimizer_Gm.step()
                optimizer_Gd.step()

                epoch_loss_D.append(loss_D.item())
                epoch_loss_G.append(loss_G.item())


            if (epoch_n+1) % logging_steps == 0:
                # eval_metrics = self.evaluate(...)
                writer.add_scalars(
                    "Training Loss", 
                    {"Generator" : sum(epoch_loss_G)/len(epoch_loss_G), "Discriminator": sum(epoch_loss_D)/len(epoch_loss_D)}, 
                    epoch_n
                    )
                # writer.add_scalars(
                #     "Evaluation Metrics", 
                #     {"Metric 1" : eval_metrics[0], "Metric 2": eval_metrics[1]}, 
                #     epoch_n
                #     )
                self.save_checkpoint({
                    "epoch": epoch_n,
                    "Gm_state_dict": self.metadata_generator.state_dict(),
                    "Gd_state_dict": self.data_generator.state_dict(),
                    "D_state_dict": self.discriminator.state_dict(),
                    "Gm_optim_state_dict": optimizer_Gm.state_dict(),
                    "Gd_optim_state_dict": optimizer_Gd.state_dict(),
                    "D_optim_state_dict": optimizer_D.state_dict()
                })

            if self.cfg.lr_scheduler:
                scheduler_Gm.step()
                scheduler_Gd.step()
                scheduler_D.step()
            
            pbar.set_description(f"Disc. Loss: {sum(epoch_loss_D)/len(epoch_loss_D):.3}, Gen. Loss: {sum(epoch_loss_G)/len(epoch_loss_G):.3}")

        writer.flush()
        writer.close()


    def generate(self, n_samples: int = 1, conditional_metadata: torch.Tensor = None, conditional_data: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates data samples from the generator.

        Args:
            n_samples: number of data sequences to generate
            conditional_metadata: metadata tensor to condition the generator on of shape [1, metadata_dim] or [n_samples, metadata_dim]
            conditional_data: data tensor to condition the generator on of shape [1, 1...window_size] or [n_samples, 1...window_size]

        Returns:
            a tuple containing the generated metadata and data tensors
        """
        self.metadata_generator.train(False)
        self.data_generator.train(False)
        self.discriminator.train(False)
        
        z_mG = torch.randn(n_samples, self.metadata_dim) # [batch, dims]
        z_dG = torch.randn(n_samples, self.window_size)  # [batch, window]
        
        if conditional_metadata is not None:
            assert conditional_metadata.shape[0] in [n_samples, 1], "Conditional metadata tensor must have dimension 0 equal to n_samples or 1"
            assert conditional_metadata.shape[1] == self.metadata_dim, "Conditional metadata tensor must have the dimension 1 equal to the model's metadata dimension" 
            if conditional_metadata.shape[0] == 1:
                g_mG = conditional_metadata.repeat(n_samples, 1)
            else:
                g_mG = conditional_metadata
        else:
            # [batch, dims] -> [batch, dims]
            g_mG = self.metadata_generator.forward(z_mG)
        
        if conditional_data is not None:
            assert conditional_data.shape[0] in [n_samples, 1], "Conditional data tensor must have dimension 0 equal to n_samples or 1"
            assert conditional_data.shape[1] <= self.window_size, "Conditional data tensor must have the dimension 1 less than or equal to the model's window size"
            if conditional_data.shape[0] == 1:
                conditional_data = conditional_data.repeat(n_samples, 1)
            z_dG[:, -conditional_data.shape[1]:] = conditional_data

        # [batch, dims] + [batch, window] = [batch, dims+window]
        gz_xG = torch.cat((g_mG, z_dG), dim=1)
        # [batch, dims+window] -> [batch, window]
        g_dG = self.data_generator.forward(gz_xG)

        return g_mG, g_dG


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
