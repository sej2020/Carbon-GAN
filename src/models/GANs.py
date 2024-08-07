"""
GAN Classes
"""
import torch
import numpy as np
import pathlib
import tqdm
from src.utils.data import CarbonDataset
from src.config.trainer_configs import TrainerConfig
from src.evaluation.quant_evaluation import QuantEvaluation
from src.utils.stat_algos import ewmv
from torch.utils.tensorboard import SummaryWriter
from joblib import dump, load
import warnings
import wandb

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
        device: device on which to run the model
    """
    def __init__(self, device="cpu"):
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
        self.device = device
    

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
    

    def prepare_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Creates training dataset by scaling, windowing, then shuffling the sequential data
        """
        training_data = CarbonDataset(self.cfg.region, self.cfg.elec_source)
        # saving the fitted scalers
        scaler_path = pathlib.Path(f"{self.cfg.logging_dir}/{self.cfg.run_name}/scalers")
        scaler_path.mkdir(parents=True, exist_ok=True)
        dump(training_data.seq_scaler, scaler_path / "seq_scaler.joblib")
        self.seq_scaler = training_data.seq_scaler

        dataX = []
        dataY = []
        for i in range(0, len(training_data) - self.window_size):
            dataX.append(training_data[i:i + self.window_size][1].squeeze())
            dataY.append(training_data[i + self.window_size][1])

        rand_idx = np.random.permutation(len(dataX))

        X_train = []
        y_train = []
        for i in range(len(dataX)):
            X_train.append(dataX[rand_idx[i]])
            y_train.append(dataY[rand_idx[i]])

        X_train = torch.vstack(X_train)
        y_train = torch.vstack(y_train)

        return X_train, y_train
    

    def evaluate(self, n_samples: int = 300) -> dict[str, np.float64]:
        """
        Evaluates the model using the QuantEvaluation metric "bin_overlap".

        Args:
            n_samples: number of samples to generate for evaluation
        """
        dataset = CarbonDataset(self.cfg.region, self.cfg.elec_source, mode="val")
        quant_eval = QuantEvaluation(self, dataset, n_samples)
        return {
            "Bin Overlap": quant_eval.bin_overlap(), 
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
    SimpleGAN is a GAN comprising an LSTM generator and an MLP (or LSTM) discriminator
    
    Attributes:
        disc_type: The type of discriminator to use. Options: "mlp", "lstm"
        seq_generator: An LSTM sequence generator
        discriminator: An MLP/LSTM discriminator
        window_size: The size of the historical data used for the LSTM generator
        seq_scaler: The standard scaler object for the sequential data
        cfg: configuration for training,
        device: device on which to run the model
    """
    def __init__(
            self,
            window_size: int,
            n_seq_gen_layers: int,
            dropout_D_in: float = 0,
            dropout_D_hid: float = 0,
            dropout_Gs: float = 0,
            cpt_path: str = None,
            disc_type: str = "mlp",
            disc_hidden_dim: int = 12,
            device="cpu"
        ):
        """
        Initializes Simple GAN model.

        Args:
            window_size: The size of the historical data used for the data generator
            n_seq_gen_layers: The number of layers in the LSTM sequence data generator
            dropout_D_in: Dropout rate for the input layer of the MLP discriminator
            dropout_D_hid: Dropout rate for the hidden layer of the MLP discriminator
            dropout_Gs: Dropout rate for the LSTM sequence data generator
            cpt_path: path to a checkpoint file to use for weights initialization. If None, weights are initialized randomly.
                scalers must be in a folder called 'scalers' in the same directory as the folder 'checkpoints' containing the 
                checkpoint file
            disc_type: The type of discriminator to use. Options: "mlp", "lstm"
            disc_hidden_dim: The hidden dimension of the MLP discriminator
            device: device on which to run the model
        """
        super().__init__(device=device)
        self.disc_type = disc_type
        if disc_type == "lstm":
            if dropout_D_hid > 0 or dropout_D_in > 0:
                warnings.warn("Cannot use discriminator dropout with LSTM discriminator. Dropout will be ignored.")
            if disc_hidden_dim != 12:
                warnings.warn("Cannot use hidden dim with LSTM discriminator. Hidden dim parameter will be ignored.")

        self.seq_generator = torch.nn.LSTM(1, hidden_size=1, num_layers=n_seq_gen_layers, dropout=dropout_Gs, batch_first=True, dtype=torch.float64, device=device)
        
        if disc_type == "lstm":
            discriminator = torch.nn.LSTM(1, hidden_size=1, batch_first=True, dtype=torch.float64, device=device)
            self.discriminator = discriminator.to(device)
            self.discriminator.prep_and_forward = lambda x: torch.sigmoid(
                self.discriminator.forward(x.unsqueeze(2))[1][0].squeeze()
                )
        else:
            discriminator = torch.nn.Sequential(
                torch.nn.Dropout(dropout_D_in),
                torch.nn.Linear(window_size, disc_hidden_dim, dtype=torch.float64),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(dropout_D_hid),
                torch.nn.Linear(disc_hidden_dim, 1, dtype=torch.float64),
                torch.nn.Sigmoid()
            )
            self.discriminator = discriminator.to(device)
        self.window_size = window_size
        if cpt_path:
            self.checkpoint_dict = torch.load(cpt_path)
            seq_generator_dict = self.checkpoint_dict['Gs_state_dict']
            discriminator_dict = self.checkpoint_dict['D_state_dict']

            self.seq_generator.load_state_dict(seq_generator_dict)
            self.discriminator.load_state_dict(discriminator_dict)

            self.seq_scaler = load(f"{pathlib.Path(cpt_path).parent.parent}/scalers/seq_scaler.joblib")


    def train(self, cfg=None, hp_search=False) -> tuple[str, np.float64, int]:
        """
        Trains the GAN model. To view the training progress, run the following command in the terminal:
        ```bash
        tensorboard --logdir=logs
        ```
        Clean up the logs directory after training is complete.

        Args:
            cfg: TrainerConfig object OR dictionary of training hyperparameters if hp_search is True.
            hp_search: whether this fine-tuning run is a part of a wandb hyperparameter search. Default is False.
        
        Returns:
            a tuple containing the path to the logging directory, the top combined evaluation score, and the epoch at which the top combined score was achieved
        """
        # Notes on variable naming scheme:
        # z: noise, g: passed through generator, d: passed through discriminator
        # G: relevant to training generator, D: relevant to training discriminator
        
        self.cfg = cfg
        
        hp_path = pathlib.Path(f"{self.cfg.logging_dir}/{self.cfg.run_name}")
        # writing out a text file to the logging directory with the string of the trainer config
        hp_path.mkdir(parents=True, exist_ok=True)
        with open(f"{hp_path}/trainer_config.txt", "w") as file:
            file.write(str(self.cfg))

        X_train, y_train = self.prepare_data()
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)

        optimizer_Gs = torch.optim.Adam(self.seq_generator.parameters(), lr=self.cfg.lr_Gs)
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.cfg.lr_D)
        criterion_G = lambda d_gG: -torch.mean(torch.log(d_gG))
        if self.cfg.sup_loss:  
            criterion_G = lambda d_gG, y_hat, y : -(
                torch.mean(torch.log(d_gG)) - 
                self.cfg.eta * torch.mean(
                    torch.linalg.vector_norm(y_hat - y, dim=1)
                    )
            )

        # 2 sided label smoothing
        if self.cfg.label_smoothing == 2:
           criterion_D = lambda d_D, d_gD: -torch.mean(
               torch.log(1 - torch.sqrt((d_D - (1 - torch.rand_like(d_D) * 0.7))**2)) + 
               torch.log(1 - torch.sqrt((d_gD - torch.rand_like(d_gD) * 0.7)**2))
           )
        # 1 sided label smoothing
        elif self.cfg.label_smoothing == 1:
           criterion_D = lambda d_D, d_gD: -torch.mean(
               torch.log(1 - torch.sqrt((d_D - (1 - torch.rand_like(d_D) * 0.3))**2)) + 
               torch.log(1 - d_gD)
           )
        # no label smoothing
        else:
            criterion_D = lambda d_D, d_gD: -torch.mean(torch.log(d_D) + torch.log(1 - d_gD))

        pbar = tqdm.tqdm(range(self.cfg.n_epochs), disable=self.cfg.disable_tqdm)
        if not hp_search:
            writer_path = pathlib.Path(f"{self.cfg.logging_dir}/{self.cfg.run_name}/tensorboard")
            writer_path.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=writer_path)
        logging_steps = int(1 / self.cfg.logging_frequency)
        saving_steps = int(1 / self.cfg.saving_frequency)

        if self.cfg.lr_scheduler is not None:
            if self.cfg.lr_scheduler == "cosine":
                scheduler_Gs = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_Gs, self.cfg.n_epochs)
                scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, self.cfg.n_epochs)
            elif self.cfg.lr_scheduler == "exponential":
                gamma = (1/5)**(1/self.cfg.n_epochs)
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
            elif self.cfg.lr_scheduler == "adaptive":
                pass
            

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

        top_combined_score = 0
        epoch_at_top = 0

        noise_scale = 0.5
        
        for epoch_n in pbar:
            epoch_loss_D = []
            epoch_loss_G = []

            for b_idx in range(0, len(X_train), self.cfg.batch_size):
                batch_x = X_train[b_idx:b_idx + self.cfg.batch_size, :] # [batch, window]
                batch_y = y_train[b_idx:b_idx + self.cfg.batch_size, :] # [batch, 1]
                if len(batch_x) < self.cfg.batch_size:
                    break

                ### Training the discriminator ###
                optimizer_D.zero_grad()

                z_D = torch.randn(self.cfg.batch_size, self.window_size, 1, dtype=torch.float64, device=self.device)  # [batch, window, 1]

                # [batch, window, 1] -> [batch, window]
                g_D = self.seq_generator.forward(z_D)[0].squeeze(2)
                if self.disc_type == "lstm":
                    # [batch, window] -> [batch]
                    d_gD = self.discriminator.prep_and_forward(g_D)
                else:
                    # [batch, window] -> [batch]
                    d_gD = self.discriminator.forward(g_D)
                
                # [batch, window] -> [batch]
                if self.cfg.noisy_input:
                    noisy_batch_x = batch_x + noise_scale * torch.randn_like(batch_x, device=self.device)
                    if self.disc_type == "lstm":
                        d_D = self.discriminator.prep_and_forward(noisy_batch_x)
                    else:
                        d_D = self.discriminator.forward(noisy_batch_x)
                else:
                    if self.disc_type == "lstm":
                        d_D = self.discriminator.prep_and_forward(batch_x)
                    else:
                        d_D = self.discriminator.forward(batch_x)

                loss_D = criterion_D(d_D, d_gD)
                loss_D.backward()
                optimizer_D.step()
                optimizer_D.zero_grad()
                

                ### Training the generator ###
                self.seq_generator.train(True)
                optimizer_Gs.zero_grad()

                z_G = torch.randn(self.cfg.batch_size, self.window_size, 1, dtype=torch.float64, device=self.device)  # [batch, window, 1]
                # [batch, window, 1] -> [batch, window]
                g_G = self.seq_generator.forward(z_G)[0].squeeze(2)

                # [batch, window] -> [batch]
                if self.disc_type == "lstm":
                    d_gG = self.discriminator.prep_and_forward(g_G)
                else:
                    d_gG = self.discriminator.forward(g_G)

                if self.cfg.sup_loss:
                    batch_y_hat = self.generate(n_samples=self.cfg.batch_size, generation_len=1, og_scale=False, condit_seq_data=batch_x, training=True) # [batch, 1]    

                if self.cfg.sup_loss:
                    loss_G = criterion_G(d_gG, batch_y_hat, batch_y)
                else:
                    loss_G = criterion_G(d_gG)
                loss_G.backward()

                if self.cfg.debug:
                    norm_dict = self.monitor_weights()
                    gms, wms, gmd = norm_dict["grad_mag_seq"], norm_dict["wgt_mag_seq"], norm_dict["grad_mag_disc"]
                    print(f"Ep {epoch_n}.{b_idx}: Loss_D = {loss_D.item():.4}, Loss_G = {loss_G.item():.4}, G_s weight mag = {wms:.4}, G_s grad mag = {gms:.4}, D grad mag = {gmd:.4}")

                optimizer_Gs.step()
                optimizer_Gs.zero_grad()


                epoch_loss_D.append(loss_D.item())
                epoch_loss_G.append(loss_G.item())

            if hp_search:
                wandb.log({"gen_train_loss": sum(epoch_loss_G)/len(epoch_loss_G), "epoch": epoch_n})
                wandb.log({"disc_train_loss": sum(epoch_loss_D)/len(epoch_loss_D), "epoch": epoch_n})
                # training stability metric (Exponentially weighted moving variance)
                if epoch_n == 0:
                    ewmv_loss_G = 0
                    ewmv_loss_D = 0
                    ewma_loss_G = sum(epoch_loss_G)/len(epoch_loss_G)
                    ewma_loss_D = sum(epoch_loss_D)/len(epoch_loss_D)
                else:
                    ewma_loss_G, ewmv_loss_G = ewmv(sum(epoch_loss_G)/len(epoch_loss_G), ewma_loss_G, ewmv_loss_G)
                    ewma_loss_D, ewmv_loss_D = ewmv(sum(epoch_loss_D)/len(epoch_loss_D), ewma_loss_D, ewmv_loss_D)
                if epoch_n >= self.cfg.n_epochs / 10:
                    wandb.log({"ewmv_loss_G": ewmv_loss_G, "epoch": epoch_n})
                    wandb.log({"ewmv_loss_D": ewmv_loss_D, "epoch": epoch_n})
                    wandb.log({"ewmv_combined": (ewmv_loss_G + ewmv_loss_D)/2, "epoch": epoch_n})

            if (epoch_n+1) % logging_steps == 0:
                for norm_name, norm in self.monitor_weights().items():
                    if torch.isnan(torch.tensor(norm)):
                        with open(f"{hp_path}/error.txt", "a") as file:
                            file.write(f"Epoch {epoch_n}: {norm_name} is NaN\n")

                try:
                    self.seq_generator.train(False)
                    eval_dict = self.evaluate()
                    self.seq_generator.train(True)
                except Exception as e:
                    print(f"Error in evaluation: {e}")
                    return f"{self.cfg.logging_dir}/{self.cfg.run_name}/", top_combined_score, epoch_at_top
                if hp_search:
                    wandb.log({"bin_overlap": eval_dict["Bin Overlap"], "epoch": epoch_n})
                    wandb.log({"coverage": eval_dict["Coverage"], "epoch": epoch_n})
                    wandb.log({"jcfe": eval_dict["JCFE"], "epoch": epoch_n})
                else:
                    writer.add_scalars(
                        "Training Loss", 
                        {"Generator" : sum(epoch_loss_G)/len(epoch_loss_G), "Discriminator": sum(epoch_loss_D)/len(epoch_loss_D)}, 
                        epoch_n
                        )
                    writer.add_scalars(
                        "Evaluation Metrics", 
                        {"Bin Overlap": eval_dict["Bin Overlap"], "Coverage": eval_dict["Coverage"], "JCFE": eval_dict["JCFE"]},
                        epoch_n
                        )
    
                top_combined_score = max(top_combined_score, (eval_dict["Bin Overlap"] + eval_dict["Coverage"]*0.5 + eval_dict["JCFE"]*0.5))
                if top_combined_score == (eval_dict["Bin Overlap"] + eval_dict["Coverage"]*0.5 + eval_dict["JCFE"]*0.5):
                    epoch_at_top = epoch_n
            
                if self.cfg.lr_scheduler == "adaptive":
                    if eval_dict["Bin Overlap"] > 0.8:
                        optimizer_Gs.param_groups[0]['lr'] = 0.1 * self.cfg.lr_Gs
                        optimizer_D.param_groups[0]['lr'] = 0.1 * self.cfg.lr_D
                    elif eval_dict["Bin Overlap"] > 0.7:
                        optimizer_Gs.param_groups[0]['lr'] = 0.25 * self.cfg.lr_Gs
                        optimizer_D.param_groups[0]['lr'] = 0.25 * self.cfg.lr_D
                    elif eval_dict["Bin Overlap"] > 0.6:
                        optimizer_Gs.param_groups[0]['lr'] = 0.5 * self.cfg.lr_Gs
                        optimizer_D.param_groups[0]['lr'] = 0.5 * self.cfg.lr_D
                    else:
                        optimizer_Gs.param_groups[0]['lr'] = self.cfg.lr_Gs
                        optimizer_D.param_groups[0]['lr'] = self.cfg.lr_D
            
            if not hp_search:
                if (epoch_n+1) % saving_steps == 0:
                    self._save_checkpoint({
                        "epoch": epoch_n,
                        "Gs_state_dict": self.seq_generator.state_dict(),
                        "D_state_dict": self.discriminator.state_dict(),
                        "Gs_optim_state_dict": optimizer_Gs.state_dict(),
                        "D_optim_state_dict": optimizer_D.state_dict()
                    })
                
            if self.cfg.lr_scheduler and self.cfg.lr_scheduler != "adaptive":
                scheduler_Gs.step()
                scheduler_D.step()
            
            if self.cfg.noisy_input:
                noise_scale = noise_scale * (1/100)**(1/self.cfg.n_epochs)
            
            if self.cfg.disable_tqdm:
                if epoch_n % 100 == 0:
                    print(f"ep: {epoch_n}")
            else:
                pbar.set_description(f"Disc. Loss: {sum(epoch_loss_D)/len(epoch_loss_D):.4}, Gen. Loss: {sum(epoch_loss_G)/len(epoch_loss_G):.4}")

        if hp_search:
            wandb.finish()
        else:
            writer.flush()
            writer.close()

        return f"{self.cfg.logging_dir}/{self.cfg.run_name}/", top_combined_score, epoch_at_top


    def generate(
        self, 
        n_samples: int = 1, 
        generation_len: int = None, 
        og_scale: bool = True, 
        condit_seq_data: torch.Tensor = None, 
        training: bool = False
        ) -> torch.Tensor:
        """
        Generates data samples from the generator.

        Args:
            n_samples: number of data sequences to generate
            generation_len: length of the generated data sequence. If None, the length is equal to the window size
            og_scale: whether to return the data on its original scale. If False, the data is returned in its scaled form
            condit_seq_data: sequential data tensor to condition the generator on of shape [1, 1...window_size] or [n_samples, 1...window_size].
                if dim(0) = 1, the tensor is repeated n_samples times. This data must be scaled.
            training: whether the generator is being used for training. If True, the generator is set to training mode

        Returns:
            a [n_samples x window_size] tensor of generated data
        """
        assert self.seq_scaler is not None, "Model must be trained before generating data. Please train or initialize weights with a cpt file."
        if generation_len is None:
            generation_len = self.window_size
        if generation_len > self.window_size:
            warnings.warn("Generation length is greater than the window size. Performance on generations beyond window size may be poor.")

        self.seq_generator.train(training)

        z_w = torch.randn(n_samples, generation_len, 1, dtype=torch.float64, device=self.device)  # [n_samples, generation_len, 1]
        
        if condit_seq_data is not None:
            assert condit_seq_data.shape[0] in [n_samples, 1], "Conditional seq data tensor must have dim(0) equal to n_samples or 1"
            assert condit_seq_data.shape[1] <= self.window_size, "Conditional seq data tensor must have dim(1) less than or equal to the model's window size"
            if condit_seq_data.shape[0] == 1:
                condit_seq_data = condit_seq_data.repeat(n_samples, 1) # [n_samples, condit_seq_size]
            
            condit_seq_size = condit_seq_data.shape[1]
            n_layers = self.seq_generator.num_layers

            z_c = torch.randn(n_samples, condit_seq_size-1, 1, dtype=torch.float64, device=self.device)  # [n_samples, condit_seq_size, 1]
            forced_hidden = condit_seq_data.unsqueeze(0) # [1, n_samples, condit_seq_size]

            hidden = torch.zeros(n_layers, n_samples, 1, dtype=torch.float64, device=self.device) # [n_layers, n_samples, 1]
            cell = torch.zeros(n_layers, n_samples, 1, dtype=torch.float64, device=self.device) # [n_layers, n_samples, 1]

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
            og_scale_g = torch.tensor(self.seq_scaler.inverse_transform(g.to("cpu").detach().numpy()), dtype=torch.float64, device=self.device)
            return og_scale_g
        else:
            return g


    def monitor_weights(self):
        """
        Reports the magnitude of the gradients and weights of the generator and discriminator for debugging purposes.

        Returns:
            a dictionary containing the magnitude of the gradients and weights of the generator and discriminator
        """
        grad_mag_disc = 0
        grad_mag_seq = 0
        wgt_mag_seq = 0
        wgt_mag_disc = 0
        # Monitoring gradients
        for weight in self.seq_generator.all_weights[0]:
            if type(weight.grad) == torch.Tensor:
                grad_mag_seq += torch.norm(weight.grad).item()
            wgt_mag_seq += torch.norm(weight).item()
        if self.disc_type == "lstm":
            for weight in self.discriminator.all_weights[0]:
                if type(weight.grad) == torch.Tensor:
                    grad_mag_disc += torch.norm(weight.grad).item()
                wgt_mag_disc += torch.norm(weight).item()
        else:
            for layer in self.discriminator:
                if hasattr(layer, "weight"):
                    if type(layer.weight.grad) == torch.Tensor:
                        grad_mag_disc += torch.norm(layer.weight.grad).item()
                    wgt_mag_disc += torch.norm(layer.weight).item()
        
        return {
            "grad_mag_seq": grad_mag_seq,
            "wgt_mag_seq": wgt_mag_seq,
            "grad_mag_disc": grad_mag_disc,
            "wgt_mag_disc": wgt_mag_disc
        }


# if __name__ == "__main__":
#     cpt = "logs\debug\CISO-hydro-2024-06-03_13-19-52\checkpoints\checkpt_e29.pt"
#     gan = SimpleGAN(window_size=24, n_seq_gen_layers=2, cpt_path=cpt)
#     seq = gan.generate(10, condit_seq_data=torch.randn(1, 8))
#     print(seq)
