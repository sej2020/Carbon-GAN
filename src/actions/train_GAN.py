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
import multiprocessing as mp
import random
import argparse
import datetime
from src.config.trainer_configs import TrainerConfig
from src.models.GANs import SimpleGAN

parser = argparse.ArgumentParser(description='GAN training')

parser.add_argument("model_type", type=str, choices=["simple"], help="Model to train")

# Debugging
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)


# Model configuration
parser.add_argument("--window_size", type=int, default=24, help="Size of historical data window for training and generation")
parser.add_argument("--n_seq_gen_layers", type=int, default=1)
parser.add_argument("--dropout_D_hid", type=float, default=0.0, help="Dropout rate for the hidden layer of the discriminator")
parser.add_argument("--dropout_D_in", type=float, default=0.0, help="Dropout rate for the input layer of the discriminator")

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
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--lr_scheduler", type=str, default=None, choices=["cosine", "exponential", "triangle2", "adaptive"], help="Learning rate scheduler to use")
parser.add_argument("--sup_loss", action=argparse.BooleanOptionalAction, default=False, help="Whether to use supervised training for the generator")
parser.add_argument("--eta", type=float, default=1, help="tuning supervised loss influence on generator")
parser.add_argument("--disable_tqdm", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--logging_frequency", type=float, default=0.1)
parser.add_argument("--resume_from_cpt", type=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--cpt_path", type=str, default=None)

parser.add_argument("--n_jobs", type=int, default=1, help="How many runs to execute in parallel")

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
    GAN = SimpleGAN(window_size=args.window_size, n_seq_gen_layers=args.n_seq_gen_layers, dropout_D_hid=args.dropout_D_hid, dropout_D_in=args.dropout_D_in)
    config = TrainerConfig(
        region=args.region,
        elec_source=args.elec_source,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr_Gs=args.lr_Gs,
        lr_D=args.lr_D,
        run_name=NAME,
        sup_loss=args.sup_loss,
        eta=args.eta,
        lr_scheduler=args.lr_scheduler,
        disable_tqdm=args.disable_tqdm,
        logging_dir=LOGGING_DIR,
        logging_frequency=args.logging_frequency,
        resume_from_cpt=args.resume_from_cpt,
        cpt_path=args.cpt_path,
        debug=args.debug
    )

def name_and_train(config):
    # adding random number to run name to avoid conflicts
    config.run_name = config.run_name + f"--{random.randint(0, 100000)}"
    GAN.train(config)

if __name__ == "__main__":
    if args.n_jobs > 1:
        with mp.Pool(args.n_jobs) as pool:
            pool.map(name_and_train, [config for _ in range(args.n_jobs)])
    else:
        GAN.train(config)
    print(f"Training complete.", flush=True)
    print(f"Logs saved at {LOGGING_DIR}: don't forget to clean up the logging directory when you're done", flush=True)
