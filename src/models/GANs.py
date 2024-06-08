"""
GAN Classes
"""
import torch
import numpy as np
import pathlib
import tqdm
from src.utils.data import CarbonDataset
from torch.utils.data import DataLoader
from src.evaluation.quant_evaluation import QuantEvaluation
from torch.utils.tensorboard import SummaryWriter
from joblib import dump, load
import warnings

torch.set_printoptions(sci_mode=False)

class GANBase(torch.nn.Module):
    """
    Base GAN class for all GAN models

    Attributes:
        seq_generator: sequential data generator model
        discriminator: discriminator model
        window_size: size of the historical data used for the generator
        generates_metadata: whether the model generates metadata
        seq_scaler: standard scaler object for the sequential data
        cfg: configuration for training
    """
    def __init__(self):
        """
        Initializes the GAN model.
        """
        super().__init__()
        self.seq_generator = None
        self.discriminator = None
        self.window_size = None
        self.generates_metadata = False
        self.seq_scaler = None
        self.cfg = None
    

    def train(self, cfg):
        """
        Trains the GAN model. To view the training progress, run the following command in the terminal:
        ```bash
        tensorboard --logdir=logs
        ```
        """
        raise NotImplementedError("train method must be implemented in child class")
    

    def generate(self, n_samples):
        """
        Generates data samples from the generator.

        Args:
            n_samples: number of data sequences to generate

        Returns:
            a tuple containing the generated metadata and data tensors
        """
        raise NotImplementedError("generate method must be implemented in child class")
    

    def evaluate(self, n_samples: int = 300) -> dict[str, np.float64]:
        """
        Evaluates the model using the QuantEvaluation metric "bin_difference".

        Args:
            n_samples: number of samples to generate for evaluation
        """
        dataset = CarbonDataset(self.cfg.region, self.cfg.elec_source, mode="test")
        quant_eval = QuantEvaluation(self, dataset, n_samples)
        return {
            "Bin Difference": quant_eval.bin_difference(), 
            "Coverage": quant_eval.coverage(), 
            "JCFE": quant_eval.jcfe()
            }


    def _save_checkpoint(self, checkpoint_dict: dict):
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
        


class SimpleGAN(GANBase):
    """
    SimpleGAN is a GAN comprising an LSTM generator and an MLP discriminator
    
    Attributes:
        seq_generator: An LSTM sequence generator
        discriminator: An MLP discriminator
        window_size: The size of the historical data used for the LSTM generator
        seq_scaler: The standard scaler object for the sequential data
        cfg: configuration for training
    """
    def __init__(self, window_size: int, n_seq_gen_layers: int, cpt_path: str = None):
        """
        Initializes Simple GAN model.

        Args:
            window_size: The size of the historical data used for the data generator
            n_seq_gen_layers: The number of layers in the LSTM sequence data generator
            cpt_path: path to a checkpoint file to use for weights initialization. If None, weights are initialized randomly.
                scalers must be in a folder called 'scalers' in the same directory as the folder 'checkpoints' containing the 
                checkpoint file
        """
        super().__init__()
        self.seq_generator = torch.nn.LSTM(1, hidden_size=1, num_layers=n_seq_gen_layers, batch_first=True, dtype=torch.float32)
        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(window_size, int(window_size/2), dtype=torch.float32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(int(window_size/2), 1, dtype=torch.float32),
            torch.nn.Sigmoid()
        )
        self.window_size = window_size

        if cpt_path:
            self.checkpoint_dict = torch.load(cpt_path)
            seq_generator_dict = self.checkpoint_dict['Gs_state_dict']
            discriminator_dict = self.checkpoint_dict['D_state_dict']

            self.seq_generator.load_state_dict(seq_generator_dict)
            self.discriminator.load_state_dict(discriminator_dict)

            self.seq_scaler = load(f"{pathlib.Path(cpt_path).parent.parent}/scalers/seq_scaler.joblib")
    

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
        # z: noise, g: passed through generator, d: passed through discriminator
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
        dump(training_data.seq_scaler, scaler_path / "seq_scaler.joblib")
        self.seq_scaler = training_data.seq_scaler

        optimizer_Gs = torch.optim.Adam(self.seq_generator.parameters(), lr=self.cfg.lr_Gs)
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.cfg.lr_D)
        criterion_D = lambda d_D, d_gD: -torch.mean(torch.log(d_D) + torch.log(1 - d_gD))
        criterion_G = lambda d_gG: -torch.mean(torch.log(d_gG))

        pbar = tqdm.tqdm(range(self.cfg.n_epochs), disable=self.cfg.disable_tqdm)
        writer_path = pathlib.Path(f"{self.cfg.logging_dir}/{self.cfg.run_name}/tensorboard")
        writer_path.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=writer_path)
        logging_steps = int(1 / self.cfg.logging_frequency)

        if self.cfg.lr_scheduler is not None:
            if self.cfg.lr_scheduler == "cosine":
                scheduler_Gs = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_Gs, self.cfg.n_epochs)
                scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, self.cfg.n_epochs)
            elif self.cfg.lr_scheduler == "exponential":
                gamma = (1/10)**(1/self.cfg.n_epochs)
                scheduler_Gs = torch.optim.lr_scheduler.ExponentialLR(optimizer_Gs, gamma=gamma)
                scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=gamma)
            elif self.cfg.lr_scheduler == "triangle2":
                scheduler_Gs = torch.optim.lr_scheduler.CyclicLR(
                    optimizer_Gs, 
                    base_lr=self.cfg.lr_Gs/100, 
                    max_lr=self.cfg.lr_Gs, 
                    step_size_up=self.cfg.n_epochs/20,
                    mode="triangular2"
                    )
                scheduler_D = torch.optim.lr_scheduler.CyclicLR(
                    optimizer_D,
                    base_lr=self.cfg.lr_D/100,
                    max_lr=self.cfg.lr_D,
                    step_size_up=self.cfg.n_epochs/20,
                    mode="triangular2"
                    )
            

        if self.cfg.resume_from_cpt:
            self.checkpoint_dict = torch.load(self.cfg.cpt_path)
            seq_generator_dict = self.checkpoint_dict['Gs_state_dict']
            discriminator_dict = self.checkpoint_dict['D_state_dict']

            self.seq_generator.load_state_dict(seq_generator_dict)
            self.discriminator.load_state_dict(discriminator_dict)

            optimizer_Gs.load_state_dict(self.checkpoint_dict["Gs_optim_state_dict"])
            optimizer_D.load_state_dict(self.checkpoint_dict["D_optim_state_dict"])

            pbar = tqdm.tqdm(
                iterable=range(self.cfg.n_epochs),
                total=self.cfg.n_epochs,
                initial=self.checkpoint_dict["epoch"] + 1,
            )

        # noise_scale = 0.5
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
            for b_idx, (_, batch) in enumerate(dataloader): # ( _ , [(window+batch)*k, 1] )
                self.discriminator.train(True)
                for k in range(self.cfg.k):
                    k_batch_lst = []
                    for b in range(self.cfg.batch_size):
                        k_batch_lst.append(batch[k*(self.window_size + self.cfg.batch_size) + b : k*(self.window_size+self.cfg.batch_size) + b + self.window_size, :]) # [window, 1]
                    k_batch = torch.stack(k_batch_lst, dim=1).squeeze().T # [batch, window]

                    z_D = torch.randn(self.cfg.batch_size, self.window_size, 1, dtype=torch.float32)  # [batch, window, 1]
                    with torch.no_grad():
                        # [batch, window, 1] -> [batch, window]
                        g_D = self.seq_generator.forward(z_D)[0].squeeze(2)
                    # [batch, window] -> [batch]
                    d_gD = self.discriminator.forward(g_D)
                    # [batch, window] -> [batch]
                    # k_batch = k_batch + noise_scale * torch.randn_like(k_batch)
                    d_D = self.discriminator.forward(k_batch)
                    optimizer_D.zero_grad()
                    loss_D = criterion_D(d_D, d_gD)
                    loss_D.backward()
                    optimizer_D.step()

                z_G = torch.randn(self.cfg.batch_size, self.window_size, 1, dtype=torch.float32)  # [batch, window, 1]
                # [batch, window, 1] -> [batch, window]
                g_G = self.seq_generator.forward(z_G)[0].squeeze(2)

                self.discriminator.train(False)
                # [batch, window] -> [batch]
                d_gG = self.discriminator.forward(g_G)
                optimizer_Gs.zero_grad()
                loss_G = criterion_G(d_gG)
                loss_G.backward()

                if self.cfg.debug:
                    if b_idx % 40 == 0:
                        gradient_magnitude_disc = 0
                        gradient_magnitude_seq = 0
                        weight_magnitude_seq = 0
                        # Monitoring gradients
                        for weight in self.seq_generator.all_weights[0]:
                            gradient_magnitude_seq += torch.norm(weight.grad).item()
                            weight_magnitude_seq += torch.norm(weight).item()
                        for layer in self.discriminator:
                            if hasattr(layer, "weight"):
                                gradient_magnitude_disc += torch.norm(layer.weight.grad).item()
                        print(f"Ep {epoch_n}.{b_idx}: Loss_D = {loss_D.item():.4}, Loss_G = {loss_G.item():.4}, G_s weight mag = {weight_magnitude_seq:.4}, G_s grad mag = {gradient_magnitude_seq:.4}, D grad mag = {gradient_magnitude_disc:.4}")

                optimizer_Gs.step()

                epoch_loss_D.append(loss_D.item())
                epoch_loss_G.append(loss_G.item())


            if (epoch_n+1) % logging_steps == 0:
                eval_dict = self.evaluate()
                writer.add_scalars(
                    "Training Loss", 
                    {"Generator" : sum(epoch_loss_G)/len(epoch_loss_G), "Discriminator": sum(epoch_loss_D)/len(epoch_loss_D)}, 
                    epoch_n
                    )
                writer.add_scalars(
                    "Evaluation Metrics", 
                    {"Bin Overlap": 1 - eval_dict["Bin Difference"], "Coverage": eval_dict["Coverage"], "JCFE": eval_dict["JCFE"]},
                    epoch_n
                    )
                self._save_checkpoint({
                    "epoch": epoch_n,
                    "Gs_state_dict": self.seq_generator.state_dict(),
                    "D_state_dict": self.discriminator.state_dict(),
                    "Gs_optim_state_dict": optimizer_Gs.state_dict(),
                    "D_optim_state_dict": optimizer_D.state_dict()
                })

            if self.cfg.lr_scheduler:
                scheduler_Gs.step()
                scheduler_D.step()
            
            # noise_scale = noise_scale * (1/100)**(1/self.cfg.n_epochs)
            
            pbar.set_description(f"Disc. Loss: {sum(epoch_loss_D)/len(epoch_loss_D):.4}, Gen. Loss: {sum(epoch_loss_G)/len(epoch_loss_G):.4}")

        writer.flush()
        writer.close()


    def generate(self, n_samples: int = 1, generation_len: int = None, og_scale: bool = True, condit_seq_data: torch.Tensor = None) -> torch.Tensor:
        """
        Generates data samples from the generator.

        Args:
            n_samples: number of data sequences to generate
            generation_len: length of the generated data sequence. If None, the length is equal to the window size
            og_scale: whether to return the data on its original scale. If False, the data is returned in its scaled form
            condit_seq_data: sequential data tensor to condition the generator on of shape [1, 1...window_size] or [n_samples, 1...window_size].
                if dim(0) = 1, the tensor is repeated n_samples times. This data must be scaled.

        Returns:
            a [n_samples x window_size] tensor of generated data
        """
        assert self.seq_scaler is not None, "Model must be trained before generating data. Please train or initialize weights with a cpt file."
        if generation_len is None:
            generation_len = self.window_size
        if generation_len > self.window_size:
            warnings.warn("Generation length is greater than the window size. Performance on generations beyond window size may be poor.")

        self.seq_generator.train(False)

        z_w = torch.randn(n_samples, generation_len, 1, dtype=torch.float32)  # [n_samples, generation_len, 1]
        
        if condit_seq_data is not None:
            assert condit_seq_data.shape[0] in [n_samples, 1], "Conditional seq data tensor must have dim(0) equal to n_samples or 1"
            assert condit_seq_data.shape[1] <= self.window_size, "Conditional seq data tensor must have dim(1) less than or equal to the model's window size"
            if condit_seq_data.shape[0] == 1:
                condit_seq_data = condit_seq_data.repeat(n_samples, 1) # [n_samples, condit_seq_size]
            
            condit_seq_size = condit_seq_data.shape[1]
            n_layers = self.seq_generator.num_layers

            z_c = torch.randn(n_samples, condit_seq_size-1, 1, dtype=torch.float32)  # [n_samples, condit_seq_size, 1]
            forced_hidden = condit_seq_data.unsqueeze(0) # [1, n_samples, condit_seq_size]

            hidden = torch.zeros(n_layers, n_samples, 1) # [n_layers, n_samples, 1]
            cell = torch.zeros(n_layers, n_samples, 1) # [n_layers, n_samples, 1]

            for t in range(condit_seq_size-1):
                hidden[-1, :, :] = forced_hidden[:, :, t].unsqueeze(2) # last layer of hidden becomes forced
                out, (hidden, cell) = self.seq_generator.forward(z_c[:, t, :].unsqueeze(1), (hidden, cell))
            
            hidden[-1, :, :] = forced_hidden[:, :, -1].unsqueeze(2) # last layer of hidden becomes forced
            # [n_samples, generation_len, 1], ([n_layers, n_samples, 1], [n_layers, n_samples, 1]) -> [n_samples, generation_len]
            g = self.seq_generator.forward(z_w, (hidden, cell))[0].squeeze(2)
        
        else:
            # [n_samples, window_size, 1] -> [n_samples, window_size]
            g = self.seq_generator.forward(z_w)[0].squeeze(2)

        if og_scale:
            og_scale_g = torch.tensor(self.seq_scaler.inverse_transform(g.detach().numpy()))
            return og_scale_g
        else:
            return g


class MCGAN(GANBase):
    """
    MCGAN is a metadata-conditional GAN comprising and MLP generator, an LSTM generator, and an MLP discriminator.
    Current functionality is questionable.
    
    Attributes:
        metadata_generator: An MLP metadata generator
        generates_metadata: whether the model generates metadata
        seq_generator: An LSTM sequence generator
        discriminator: An MLP discriminator
        metadata_dim: The dimension of the metadata
        window_size: The size of the historical data used for the LSTM generator

        metadata_scaler: The standard scaler object for the metadata
        seq_scaler: The standard scaler object for the sequential data
        cfg: configuration for training
    """
    def __init__(self, metadata_dim: int, window_size: int, n_seq_gen_layers: int = 1, cpt_path: str = None):
        """
        Initializes Simple GAN model.

        Args:
            metadata_dim: The dimension of the metadata. Using the CarbonDataset class, this is 8
            window_size: The size of the historical data used for the data generator
            n_seq_gen_layers: The number of layers in the LSTM sequence data generator
            cpt_path: path to a checkpoint file to use for weights initialization. If None, weights are initialized randomly.
                scalers must be in a folder called 'scalers' in the same directory as the folder 'checkpoints' containing the 
                checkpoint file
        """
        super().__init__()
        self.metadata_generator = torch.nn.Sequential(
            torch.nn.Linear(metadata_dim, metadata_dim, dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Linear(metadata_dim, metadata_dim, dtype=torch.float32)
        )
        self.seq_generator = torch.nn.LSTM(metadata_dim + 1, hidden_size=1, num_layers=n_seq_gen_layers, batch_first=True, dtype=torch.float32)
        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(metadata_dim + window_size, 8, dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1, dtype=torch.float32),
            torch.nn.Sigmoid()
        )
        self.metadata_dim = metadata_dim
        self.window_size = window_size
        self.metadata_scaler = None
        self.seq_scaler = None
        self.generates_metadata = True

        if cpt_path:
            self.checkpoint_dict = torch.load(cpt_path)
            metadata_generator_dict = self.checkpoint_dict['Gm_state_dict']
            seq_generator_dict = self.checkpoint_dict['Gs_state_dict']
            discriminator_dict = self.checkpoint_dict['D_state_dict']

            self.metadata_generator.load_state_dict(metadata_generator_dict)
            self.seq_generator.load_state_dict(seq_generator_dict)
            self.discriminator.load_state_dict(discriminator_dict)

            self.metadata_scaler = load(f"{pathlib.Path(cpt_path).parent.parent}/scalers/metadata_scaler.joblib")
            self.seq_scaler = load(f"{pathlib.Path(cpt_path).parent.parent}/scalers/seq_scaler.joblib")
    

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
        # z: noise, m: relevant to metadata, s: relevant to sequential data, x: metadata + seq data
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
        dump(training_data.seq_scaler, scaler_path / "seq_scaler.joblib")
        self.metadata_scaler = training_data.metadata_scaler
        self.seq_scaler = training_data.seq_scaler

        optimizer_Gm = torch.optim.Adam(self.metadata_generator.parameters(), lr=self.cfg.lr_Gm)
        optimizer_Gs = torch.optim.Adam(self.seq_generator.parameters(), lr=self.cfg.lr_Gs)
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.cfg.lr_D)
        criterion_D = lambda d_xD, d_gxD: -torch.mean(torch.log(d_xD) + torch.log(1 - d_gxD))
        criterion_G = lambda d_gxG: -torch.mean(torch.log(d_gxG))

        pbar = tqdm.tqdm(range(self.cfg.n_epochs), disable=self.cfg.disable_tqdm)
        writer_path = pathlib.Path(f"{self.cfg.logging_dir}/{self.cfg.run_name}/tensorboard")
        writer_path.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=writer_path)
        logging_steps = int(1 / self.cfg.logging_frequency)

        if self.cfg.lr_scheduler:
            scheduler_Gm = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_Gm, self.cfg.n_epochs)
            scheduler_Gs = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_Gs, self.cfg.n_epochs)
            scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, self.cfg.n_epochs)

        if self.cfg.resume_from_cpt:
            self.checkpoint_dict = torch.load(self.cfg.cpt_path)
            metadata_generator_dict = self.checkpoint_dict['Gm_state_dict']
            seq_generator_dict = self.checkpoint_dict['Gs_state_dict']
            discriminator_dict = self.checkpoint_dict['D_state_dict']

            self.metadata_generator.load_state_dict(metadata_generator_dict)
            self.seq_generator.load_state_dict(seq_generator_dict)
            self.discriminator.load_state_dict(discriminator_dict)

            optimizer_Gm.load_state_dict(self.checkpoint_dict["Gm_optim_state_dict"])
            optimizer_Gs.load_state_dict(self.checkpoint_dict["Gs_optim_state_dict"])
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
            for b_idx, (batch_m, batch_s) in enumerate(dataloader): # ( [(window+batch)*k, dims] , [(window+batch)*k, 1] )
                self.discriminator.train(True)
                for k in range(self.cfg.k):
                    k_batch_m_lst = []
                    k_batch_s_lst = []
                    for b in range(self.cfg.batch_size):
                        k_batch_m_lst.append(batch_m[k*(self.window_size + self.cfg.batch_size) + b, :]) # [1, dims]
                        k_batch_s_lst.append(batch_s[k*(self.window_size + self.cfg.batch_size) + b : k*(self.window_size+self.cfg.batch_size) + b + self.window_size, :]) # [window, 1]
                    k_batch_m = torch.stack(k_batch_m_lst, dim=0).squeeze() # [batch, dims]
                    k_batch_s = torch.stack(k_batch_s_lst, dim=1).squeeze().T # [batch, window]

                    k_batch_x = torch.cat((k_batch_m, k_batch_s), dim=1) # [batch, dims+window]

                    z_mD = torch.randn(self.cfg.batch_size, self.metadata_dim, dtype=torch.float32) # [batch, dims]
                    z_sD = torch.randn(self.cfg.batch_size, self.window_size, dtype=torch.float32)  # [batch, window]
                    with torch.no_grad():
                        # [batch, dims] -> [batch, dims]
                        g_mD = self.metadata_generator.forward(z_mD)
                        # [batch, window, dims] + [batch, window, 1] = [batch, window, dims+1]
                        gz_xD = torch.cat((g_mD.unsqueeze(1).repeat(1, self.window_size, 1), z_sD.unsqueeze(2)), dim=2)
                        # [batch, window, dims+1] -> [batch, window]
                        g_sD = self.seq_generator.forward(gz_xD)[0].squeeze()
                        # [batch, dims] + [batch, window] = [batch, dims+window]
                        g_xD = torch.cat((g_mD, g_sD), dim=1)
                    # [batch, dims+window] -> [batch]
                    d_gxD = self.discriminator.forward(g_xD)
                    # [batch, dims+window] -> [batch]
                    d_xD = self.discriminator.forward(k_batch_x)
                    optimizer_D.zero_grad()
                    loss_D = criterion_D(d_xD, d_gxD)
                    loss_D.backward()
                    optimizer_D.step()

                z_mG = torch.randn(self.cfg.batch_size, self.metadata_dim, dtype=torch.float32) # [batch, dims]
                z_sG = torch.randn(self.cfg.batch_size, self.window_size, dtype=torch.float32)  # [batch, window]
                # [batch, dims] -> [batch, dims]
                g_mG = self.metadata_generator.forward(z_mG)
                # [batch, window, dims] + [batch, window, 1] = [batch, window, dims+1]
                gz_xG = torch.cat((g_mG.unsqueeze(1).repeat(1, self.window_size, 1), z_sG.unsqueeze(2)), dim=2)
                # [batch, window, dims+1] -> [batch, window]
                g_sG = self.seq_generator.forward(gz_xG)[0].squeeze()
                # [batch, dims] + [batch, window] = [batch, dims+window]
                g_xG = torch.cat((g_mG, g_sG), dim=1)

                self.discriminator.train(False)
                # [batch, dims+window] -> [batch]
                d_gxG = self.discriminator.forward(g_xG)
                optimizer_Gm.zero_grad()
                optimizer_Gs.zero_grad()
                loss_G = criterion_G(d_gxG)
                loss_G.backward()

                if self.cfg.debug:
                    if b_idx % 40 == 0:
                        # Monitoring gradients
                        gradient_magnitude_meta = 0
                        for layer in self.metadata_generator:
                            if hasattr(layer, "weight"):
                                gradient_magnitude_meta += torch.norm(layer.weight.grad).item()
                        gradient_magnitude_seq = 0
                        for weight in self.seq_generator.all_weights[0]:
                            gradient_magnitude_seq += torch.norm(weight.grad).item()
                        if torch.isnan(torch.tensor(gradient_magnitude_meta)) or torch.isnan(torch.tensor(gradient_magnitude_seq)):
                            breakpoint()
                        else:
                            print(f"Ep {epoch_n}.{b_idx}: Loss_D = {loss_D.item():.3}, Loss_G = {loss_G.item():.3}, G_m grad mag = {gradient_magnitude_meta:.3}, G_s grad mag = {gradient_magnitude_seq:.3}")

                optimizer_Gm.step()
                optimizer_Gs.step()

                epoch_loss_D.append(loss_D.item())
                epoch_loss_G.append(loss_G.item())


            if (epoch_n+1) % logging_steps == 0:
                eval_bin_difference = self.evaluate()
                writer.add_scalars(
                    "Training Loss", 
                    {"Generator" : sum(epoch_loss_G)/len(epoch_loss_G), "Discriminator": sum(epoch_loss_D)/len(epoch_loss_D)}, 
                    epoch_n
                    )
                writer.add_scalars(
                    "Evaluation Metrics", 
                    {"Seq Bin Diff": eval_bin_difference[1], "Meta Bin Diff (avg)": np.mean(eval_bin_difference[0])},
                    epoch_n
                    )
                self._save_checkpoint({
                    "epoch": epoch_n,
                    "Gm_state_dict": self.metadata_generator.state_dict(),
                    "Gs_state_dict": self.seq_generator.state_dict(),
                    "D_state_dict": self.discriminator.state_dict(),
                    "Gm_optim_state_dict": optimizer_Gm.state_dict(),
                    "Gs_optim_state_dict": optimizer_Gs.state_dict(),
                    "D_optim_state_dict": optimizer_D.state_dict()
                })

            if self.cfg.lr_scheduler:
                scheduler_Gm.step()
                scheduler_Gs.step()
                scheduler_D.step()
            
            pbar.set_description(f"Disc. Loss: {sum(epoch_loss_D)/len(epoch_loss_D):.3}, Gen. Loss: {sum(epoch_loss_G)/len(epoch_loss_G):.3}")

        writer.flush()
        writer.close()


    def generate(self, n_samples: int = 1, og_scale: bool = True, conditional_metadata: torch.Tensor = None, condit_seq_data: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        CURRENTLY NOT WORKING.

        Generates data samples from the generator.

        Args:
            n_samples: number of data sequences to generate
            og_scale: whether to return the data on its original scale. If False, the data is returned in its scaled form
            conditional_metadata: metadata tensor to condition the generator on of shape [1, metadata_dim] or [n_samples, metadata_dim]
            condit_seq_data: sequential data tensor to condition the generator on of shape [1, 1...window_size] or [n_samples, 1...window_size]

        Returns:
            a tuple containing the generated metadata and seq data tensors
        """
        assert self.metadata_scaler is not None, "Model must be trained before generating data. Please train or initialize weights with a cpt file."

        self.metadata_generator.train(False)
        self.seq_generator.train(False)
        self.discriminator.train(False)
        
        z_m = torch.randn(n_samples, self.metadata_dim, dtype=torch.float32) # [batch, dims]
        z_s = torch.randn(n_samples, self.window_size, dtype=torch.float32)  # [batch, window]
        
        if conditional_metadata is not None:
            assert conditional_metadata.shape[0] in [n_samples, 1], "Conditional metadata tensor must have dimension 0 equal to n_samples or 1"
            assert conditional_metadata.shape[1] == self.metadata_dim, "Conditional metadata tensor must have the dimension 1 equal to the model's metadata dimension" 
            if conditional_metadata.shape[0] == 1:
                g_m = conditional_metadata.repeat(n_samples, 1)
            else:
                g_m = conditional_metadata
        else:
            # [batch, dims] -> [batch, dims]
            g_m = self.metadata_generator.forward(z_m)
        
        if condit_seq_data is not None:
            assert condit_seq_data.shape[0] in [n_samples, 1], "Conditional seq data tensor must have dimension 0 equal to n_samples or 1"
            assert condit_seq_data.shape[1] <= self.window_size, "Conditional seq data tensor must have the dimension 1 less than or equal to the model's window size"
            if condit_seq_data.shape[0] == 1:
                condit_seq_data = condit_seq_data.repeat(n_samples, 1)
            z_s[:, -condit_seq_data.shape[1]:] = condit_seq_data

        # [batch, window, dims] + [batch, window, 1] = [batch, window, dims+1]
        gz_x = torch.cat((g_m.unsqueeze(1).repeat(1, self.window_size, 1), z_s.unsqueeze(2)), dim=2)
        # [batch, window, dims+1] -> [batch, window]
        g_s = self.seq_generator.forward(gz_x)[0].squeeze(2)

        if og_scale:
            og_scale_gm = torch.tensor(self.metadata_scaler.inverse_transform(g_m.detach().numpy()))
            og_scale_gs = torch.tensor(self.seq_scaler.inverse_transform(g_s.detach().numpy()))
            return og_scale_gm, og_scale_gs
        else:
            return g_m, g_s
    

# if __name__ == "__main__":
#     cpt = "logs\debug\CISO-hydro-2024-06-03_13-19-52\checkpoints\checkpt_e29.pt"
#     gan = SimpleGAN(window_size=24, n_seq_gen_layers=2, cpt_path=cpt)
#     seq = gan.generate(10, condit_seq_data=torch.randn(1, 8))
#     print(seq)