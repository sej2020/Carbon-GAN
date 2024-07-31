"""
Implements the configuration for the training a GAN model.
"""
import datetime

class TrainerConfig:
    def __init__(
        self,
        region: str,
        elec_source: str,
        n_epochs: int,
        batch_size: int,
        lr_Gs: float,
        lr_D: float,
        run_name: str,
        lr_scheduler: str = None,
        sup_loss = False,
        eta = 1,
        disable_tqdm = False,
        logging_dir = "logs",
        logging_frequency = 0.1,
        saving_frequency = 0.1,
        resume_from_cpt = False,
        cpt_path = None,
        debug = False,
        # could be temporary
        label_smoothing = False,
        noisy_input = False,
    ):
        """
        Configuration for training a simple GAN.

        Args:
            region: The region from which the data is reported
            elec_source: The type of fuel/method for electricity generation
            n_epochs: The number of epochs to train the model
            batch_size: The batch size for training
            lr_Gs: The learning rate for training the data generator
            lr_D: The learning rate for training the discriminator
            run_name: The name of the run
            lr_scheduler: The lr scheduler to use. Options: "cosine", "exponential", "triangle2", "adaptive"
            sup_loss: Whether to introduce a supervised loss for the generator (in addition to the GAN loss)
            eta: The weight of the supervised loss
            disable_tqdm: Whether to disable progress bar
            logging_dir: The directory to save the logs
            logging_frequency: how often to evaluate the model on the test set, log the training metrics, and save the model
            saving_frequency: how often to save the model 
            resume_from_cpt: Whether to resume training from a checkpoint
            cpt_path: The path to the checkpoint to resume training from
            debug: Whether to run in debug mode, activating print statements monitoring the model loss and gradients
        """
        self.region = region
        self.elec_source = elec_source
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr_Gs = lr_Gs
        self.lr_D = lr_D
        if run_name:
            self.run_name = run_name
        else:
            self.run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.lr_scheduler = lr_scheduler
        self.sup_loss = sup_loss
        self.eta = eta
        self.disable_tqdm = disable_tqdm
        self.logging_dir = logging_dir
        self.logging_frequency = logging_frequency
        self.saving_frequency = saving_frequency
        self.resume_from_cpt = resume_from_cpt
        self.cpt_path = cpt_path
        self.debug = debug
        # could be temporary
        self.label_smoothing = label_smoothing
        self.noisy_input = noisy_input

    def __str__(self):
        return f"""TrainerConfig(
        Run Name: {self.run_name}
        --------------------------
        Region: {self.region}, Electricity Source: {self.elec_source}, Number of Epochs: {self.n_epochs}, 
        Batch Size: {self.batch_size}, Learning Rates (Gs, D): ({self.lr_Gs}, {self.lr_D}), Learning Rate Scheduler: {self.lr_scheduler},
        Sup Loss: {self.sup_loss}, Eta: {self.eta}, Logging Directory: {self.logging_dir}, 
        (Logging, Saving) Frequency: {self.logging_frequency},{self.saving_frequency}, Resume from Checkpoint: {self.resume_from_cpt}, 
        Checkpoint Path: {self.cpt_path}
        )
        """