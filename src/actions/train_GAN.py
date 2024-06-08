"""
Script for training a simple GAN model.

Typical usage example:
```bash
>>> python -m src.actions.train_GAN simple --debug --region CISO --elec_source hydro --n_epochs 500 --batch_size 128 --lr_scheduler exponential --disable_tqdm
```
And to view the training progress, run the following command in the terminal:
```bash
>>> tensorboard --logdir logs
```
Clean up the logs directory after training is complete.
"""

import argparse
import datetime
from src.config.trainer_configs import TrainerConfig, MCGANTrainerConfig
from src.models.GANs import MCGAN
from src.models.GANs import SimpleGAN

parser = argparse.ArgumentParser(description='GAN training')

parser.add_argument("model_type", type=str, choices=["simple", "mcgan"], help="Model to train")

# Debugging
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)


# Model configuration
parser.add_argument("--window_size", type=int, default=24, help="Size of historical data window for training and generation")
parser.add_argument("--n_seq_gen_layers", type=int, default=1)

# Training configuration
parser.add_argument("--region", type=str, 
    choices=[
        "AUS_QLD", "BPAT", "CISO", "DE", "ERCO", "ES", 
        "FPL", "ISNE", "NL", "NYSIO", "PJM", "PL", "SE"
        ], required=True, help="Region of data to train the model on")
parser.add_argument("--elec_source", type=str,
    choices=[
        "biomass", "coal", "nat_gas", "hydro", "geothermal",
        "nuclear", "oil", "solar", "wind", "other", "unknown"
    ])
parser.add_argument("--n_epochs", type=int, required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--lr_Gs", type=float, default=0.0005)
parser.add_argument("--lr_D", type=float, default=0.0005)
parser.add_argument("--k", type=int, default=1, help="Number of times to train the discriminator for each generator training step")
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--lr_scheduler", type=str, default=None, choices=["cosine", "exponential", "triangle2"], help="Learning rate scheduler to use")
parser.add_argument("--disable_tqdm", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--logging_frequency", type=float, default=0.1)
parser.add_argument("--resume_from_cpt", type=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--cpt_path", type=str, default=None)

# MCGAN specific
parser.add_argument("--lr_Gm", type=float, default=0.001, help="Learning rate for the metadata generator")
parser.add_argument("--metadata_dim", type=int, default=8, help="Dimension of the metadata: if using provided dataset, this is 8.")

args = parser.parse_args()

if args.run_name:
    NAME = args.run_name
else:
    # may need to replace with different scheme if I end up doing hyperparameter tuning using this script
    NAME = f"{args.region}-{args.elec_source}-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

if args.debug:
    LOGGING_DIR = f"logs/debug"
else:
    LOGGING_DIR = f"logs"

if args.model_type == "simple":
    GAN = SimpleGAN(window_size=args.window_size, n_seq_gen_layers=args.n_seq_gen_layers)
    config = TrainerConfig(
        region=args.region,
        elec_source=args.elec_source,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr_Gs=args.lr_Gs,
        lr_D=args.lr_D,
        k=args.k,
        run_name=NAME,
        lr_scheduler=args.lr_scheduler,
        disable_tqdm=args.disable_tqdm,
        logging_dir=LOGGING_DIR,
        logging_frequency=args.logging_frequency,
        resume_from_cpt=args.resume_from_cpt,
        cpt_path=args.cpt_path,
        debug=args.debug
    )
elif args.model_type == "mcgan":
    GAN = MCGAN(metadata_dim=args.metadata_dim, window_size=args.window_size, n_seq_gen_layers=args.n_seq_gen_layers)
    config = MCGANTrainerConfig(
        region=args.region,
        elec_source=args.elec_source,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr_Gm=args.lr_Gm,
        lr_Gs=args.lr_Gs,
        lr_D=args.lr_D,
        k=args.k,
        run_name=NAME,
        lr_scheduler=args.lr_scheduler,
        disable_tqdm=args.disable_tqdm,
        logging_dir=LOGGING_DIR,
        logging_frequency=args.logging_frequency,
        resume_from_cpt=args.resume_from_cpt,
        cpt_path=args.cpt_path,
        debug=args.debug
    )

GAN.train(config)
print(f"Training complete.", flush=True)
print(f"Logs saved at {LOGGING_DIR}: don't forget to clean up the logging directory when you're done", flush=True)
