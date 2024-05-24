"""
Script for training a simple GAN model.

Typical usage example:
```bash
>>> python -m src.actions.train_simpleGAN --debug --region AUS_QLD --elec_source solar --n_epochs 1000 --batch_size 64 --k 1
```
And to view the training progress, run the following command in the terminal:
```bash
>>> tensorboard --logdir logs
```
Clean up the logs directory after training is complete.
"""

import argparse
import datetime
from src.config.trainer_configs import GANTrainerConfig
from src.models.baseline import SimpleGAN

parser = argparse.ArgumentParser(description='Train a simple GAN')

# Debugging
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)

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
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--k", type=int, default=4, help="Number of times to train the discriminator for each generator training step")
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--lr_scheduler", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--disable_tqdm", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--logging_frequency", type=float, default=0.1)
parser.add_argument("--resume_from_cpt", type=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--cpt_path", type=str, default=None)

# Initializing models
parser.add_argument("--metadata_dim", type=int, default=9, help="Dimension of the metadata: if using provided dataset, this is 9.")
parser.add_argument("--window_size", type=int, default=24, help="Size of historical data window for training and generation")

args = parser.parse_args()

if args.run_name:
    NAME = args.run_name
else:
    # may need to replace with different scheme if I end up doing hyperparameter tuning using this script
    NAME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if args.debug:
    NAME = 'debug'
    LOGGING_DIR = f"logs/debug"
else:
    LOGGING_DIR = f"logs/{args.run_name}"

GAN = SimpleGAN(metadata_dim=args.metadata_dim, window_size=args.window_size)

config = GANTrainerConfig(
    region=args.region,
    elec_source=args.elec_source,
    n_epochs=args.n_epochs,
    batch_size=args.batch_size,
    lr=args.lr,
    k=args.k,
    run_name=args.run_name,
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
