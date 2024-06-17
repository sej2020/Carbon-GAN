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
    SimpleGAN is a GAN comprising an LSTM generator and an MLP discriminator
    
    Attributes:
        seq_generator: An LSTM sequence generator
        discriminator: An MLP discriminator
        window_size: The size of the historical data used for the LSTM generator
        seq_scaler: The standard scaler object for the sequential data
        cfg: configuration for training
    """
    def __init__(
            self,
            window_size: int,
            n_seq_gen_layers: int,
            dropout_D_in: float = 0,
            dropout_D_hid: float = 0,
            cpt_path: str = None,
        ):
        """
        Initializes Simple GAN model.

        Args:
            window_size: The size of the historical data used for the data generator
            n_seq_gen_layers: The number of layers in the LSTM sequence data generator
            dropout_D_in: Dropout rate for the input layer of the discriminator
            dropout_D_hid: Dropout rate for the hidden layer of the discriminator
            cpt_path: path to a checkpoint file to use for weights initialization. If None, weights are initialized randomly.
                scalers must be in a folder called 'scalers' in the same directory as the folder 'checkpoints' containing the 
                checkpoint file
        """
        super().__init__()
        self.seq_generator = torch.nn.LSTM(1, hidden_size=1, num_layers=n_seq_gen_layers, batch_first=True, dtype=torch.float64)
        self.discriminator = torch.nn.Sequential(
            torch.nn.Dropout(dropout_D_in),
            torch.nn.Linear(window_size, 12, dtype=torch.float64),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout_D_hid),
            torch.nn.Linear(12, 1, dtype=torch.float64),
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


    def train(self, cfg: TrainerConfig) -> tuple[str, np.float64, int]:
        """
        Trains the GAN model. To view the training progress, run the following command in the terminal:
        ```bash
        tensorboard --logdir=logs
        ```
        Clean up the logs directory after training is complete.

        Args:
            cfg: configuration for training
        
        Returns:
            a tuple containing the path to the logging directory, the top combined evaluation score, and the epoch at which the top combined score was achieved
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

        X_train, y_train = self.prepare_data()

        optimizer_Gs = torch.optim.Adam(self.seq_generator.parameters(), lr=self.cfg.lr_Gs)
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.cfg.lr_D)
        criterion_D = lambda d_D, d_gD: -torch.mean(torch.log(d_D) + torch.log(1 - d_gD))
        criterion_G = lambda d_gG: -torch.mean(torch.log(d_gG))
        if self.cfg.sup_loss:  
            criterion_G = lambda d_gG, y_hat, y : -(
                torch.mean(torch.log(d_gG)) - 
                self.cfg.eta * torch.mean(
                    torch.linalg.vector_norm(y_hat - y, dim=1)
                    )
            )

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
        
        for epoch_n in pbar:
            epoch_loss_D = []
            epoch_loss_G = []

            for b_idx in range(0, len(X_train), self.cfg.batch_size):
                batch_x = X_train[b_idx:b_idx + self.cfg.batch_size, :] # [batch, window]
                batch_y = y_train[b_idx:b_idx + self.cfg.batch_size, :] # [batch, 1]
                if len(batch_x) < self.cfg.batch_size:
                    break

                ### Training the discriminator ###
                self.discriminator.train(True)

                z_D = torch.randn(self.cfg.batch_size, self.window_size, 1, dtype=torch.float64)  # [batch, window, 1]
                with torch.no_grad():
                    # [batch, window, 1] -> [batch, window]
                    g_D = self.seq_generator.forward(z_D)[0].squeeze(2)
                # [batch, window] -> [batch]
                d_gD = self.discriminator.forward(g_D)
                
                # [batch, window] -> [batch]
                d_D = self.discriminator.forward(batch_x)

                optimizer_D.zero_grad()
                loss_D = criterion_D(d_D, d_gD)
                loss_D.backward()
                optimizer_D.step()

                ### Training the generator ###
                z_G = torch.randn(self.cfg.batch_size, self.window_size, 1, dtype=torch.float64)  # [batch, window, 1]
                # [batch, window, 1] -> [batch, window]
                g_G = self.seq_generator.forward(z_G)[0].squeeze(2)

                self.discriminator.train(False)
                # [batch, window] -> [batch]
                d_gG = self.discriminator.forward(g_G)

                if self.cfg.sup_loss:
                    batch_y_hat = self.generate(n_samples=self.cfg.batch_size, generation_len=1, og_scale=False, condit_seq_data=batch_x, training=True) # [batch, 1]    

                optimizer_Gs.zero_grad()

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

                epoch_loss_D.append(loss_D.item())
                epoch_loss_G.append(loss_G.item())


            if (epoch_n+1) % logging_steps == 0:
                for norm_name, norm in self.monitor_weights().items():
                    if torch.isnan(torch.tensor(norm)):
                        with open(f"{hp_path}/error.txt", "a") as file:
                            file.write(f"Epoch {epoch_n}: {norm_name} is NaN\n")

                eval_dict = self.evaluate()
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
                self._save_checkpoint({
                    "epoch": epoch_n,
                    "Gs_state_dict": self.seq_generator.state_dict(),
                    "D_state_dict": self.discriminator.state_dict(),
                    "Gs_optim_state_dict": optimizer_Gs.state_dict(),
                    "D_optim_state_dict": optimizer_D.state_dict()
                })

                top_combined_score = max(top_combined_score, (eval_dict["Bin Overlap"] + eval_dict["Coverage"]*0.5 + eval_dict["JCFE"]*0.5))
                if top_combined_score == (eval_dict["Bin Overlap"] + eval_dict["Coverage"]*0.5 + eval_dict["JCFE"]*0.5):
                    epoch_at_top = epoch_n
            
                if self.cfg.lr_scheduler == "adaptive":
                    if eval_dict["Bin Overlap"] > 0.9:
                        optimizer_Gs.param_groups[0]['lr'] = 0.01 * self.cfg.lr_Gs
                        optimizer_D.param_groups[0]['lr'] = 0.01 * self.cfg.lr_D
                    elif eval_dict["Bin Overlap"] > 0.825:
                        optimizer_Gs.param_groups[0]['lr'] = 0.05 * self.cfg.lr_Gs
                        optimizer_D.param_groups[0]['lr'] = 0.05 * self.cfg.lr_D
                    elif eval_dict["Bin Overlap"] > 0.75:
                        optimizer_Gs.param_groups[0]['lr'] = 0.1 * self.cfg.lr_Gs
                        optimizer_D.param_groups[0]['lr'] = 0.1 * self.cfg.lr_D
                    else:
                        optimizer_Gs.param_groups[0]['lr'] = self.cfg.lr_Gs
                        optimizer_D.param_groups[0]['lr'] = self.cfg.lr_D

            if self.cfg.lr_scheduler and self.cfg.lr_scheduler != "adaptive":
                scheduler_Gs.step()
                scheduler_D.step()
            
            if self.cfg.disable_tqdm:
                if epoch_n % 100 == 0:
                    print(f"ep: {epoch_n}")
            else:
                pbar.set_description(f"Disc. Loss: {sum(epoch_loss_D)/len(epoch_loss_D):.4}, Gen. Loss: {sum(epoch_loss_G)/len(epoch_loss_G):.4}")

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

        z_w = torch.randn(n_samples, generation_len, 1, dtype=torch.float64)  # [n_samples, generation_len, 1]
        
        if condit_seq_data is not None:
            assert condit_seq_data.shape[0] in [n_samples, 1], "Conditional seq data tensor must have dim(0) equal to n_samples or 1"
            assert condit_seq_data.shape[1] <= self.window_size, "Conditional seq data tensor must have dim(1) less than or equal to the model's window size"
            if condit_seq_data.shape[0] == 1:
                condit_seq_data = condit_seq_data.repeat(n_samples, 1) # [n_samples, condit_seq_size]
            
            condit_seq_size = condit_seq_data.shape[1]
            n_layers = self.seq_generator.num_layers

            z_c = torch.randn(n_samples, condit_seq_size-1, 1, dtype=torch.float64)  # [n_samples, condit_seq_size, 1]
            forced_hidden = condit_seq_data.unsqueeze(0) # [1, n_samples, condit_seq_size]

            hidden = torch.zeros(n_layers, n_samples, 1, dtype=torch.float64) # [n_layers, n_samples, 1]
            cell = torch.zeros(n_layers, n_samples, 1, dtype=torch.float64) # [n_layers, n_samples, 1]

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
            grad_mag_seq += torch.norm(weight.grad).item()
            wgt_mag_seq += torch.norm(weight).item()
        for layer in self.discriminator:
            if hasattr(layer, "weight"):
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