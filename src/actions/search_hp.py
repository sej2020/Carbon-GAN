"""
Searching over hyperparameters for finetuning the TEMPO model on energy insights data.

Typical usage example:
```bash
>>> python -m src.actions.search_hp simple --device cuda:0 --n_runs 8
```
"""

import wandb
wandb.login()
import argparse
from src.models.GANs import SimpleGAN

parser = argparse.ArgumentParser("hyperparameter search SimpleGAN")

# Training Meta-parameters
parser.add_argument("model_type", type=str, choices=["simple"], help="Model to train")
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--n_runs", type=int, default=1)
parser.add_argument("--device", type=str, default="cpu")

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
parser.add_argument("--disable_tqdm", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--logging_frequency", type=float, default=0.05)
parser.add_argument("--saving_frequency", type=float, default=0.01)
parser.add_argument("--resume_from_cpt", type=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--cpt_path", type=str, default=None)
parser.add_argument("--run_name", type=str, default=None)


args = parser.parse_args()

if args.run_name is None:
    NAME = 'hp_run_'
else:
    NAME = args.run_name

if args.debug:
    NAME = 'debug'
    LOGGING_DIR = f"logs/hp_search"
else:
    LOGGING_DIR = f"logs/hp_search"

hp_search_trainer_config = {
    'window_size': {'value': 24},
    'n_seq_gen_layers': {
        'values': [1, 2]
        },
    'dropout_D_hid': {
        'values': [0, 0.5, 0.65, 0.8, 0.9]
        },
    'dropout_D_in': {
        'distribution': 'uniform',
        'max': 0.5,
        'min': 0
        },
    'dropout_Gs': {
        'distribution': 'uniform',
        'max': 0.3,
        'min': 0
        },
    'disc_type': {
        'values': ['mlp', 'lstm']
        },
    'disc_hidden_dim': {
        'values': [4, 8, 12, 24]
        },
    ############################################
    'region': {'value': args.region},
    'elec_source': {'value': args.elec_source},
    'n_epochs': {
        'distribution': 'int_uniform',
        'max': 60,
        'min': 50
        },
    'batch_size': {
        'values': [512, 1024, 2048, 4096]
        },
    'lr_Gs': {
        'distribution': 'uniform',
        'max': 5e-2,
        'min': 1e-5
        },
    'lr_D': {
        'distribution': 'uniform',
        'max': 5e-2,
        'min': 1e-5
        },
    'logging_dir': {'value': LOGGING_DIR},
    'logging_frequency': {'value': args.logging_frequency},
    'saving_frequency': {'value': args.saving_frequency},
    'disable_tqdm': {'value': args.disable_tqdm},
    'resume_from_cpt': {'value': args.resume_from_cpt},
    'cpt_path': {'value': args.cpt_path},
    'run_name': {'value': NAME},
    ############################################
    'label_smoothing': {
        'values': [0, 1, 2]
        },
    'noisy_input': {
        'values': [True, False]
        },
    'lr_scheduler': {
        'values': ['adaptive', 'cosine', 'exponential', 'triangle2', None],
        'probabilities': [0.4, 0.15, 0.15, 0.15, 0.15]
        },
    'sup_loss': {
        'values': [True, False]
        },
    'eta': {
        'values': [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
        },
    'debug': {'value': args.debug},
    'device': {'value': args.device}
}

def setup_and_train(cfg=None):
    wandb.init(project="search-hp-SimpleGAN")
    config = wandb.config
    GAN = SimpleGAN(
        window_size=config.window_size,
        n_seq_gen_layers=config.n_seq_gen_layers,
        dropout_D_hid=config.dropout_D_hid,
        dropout_D_in=config.dropout_D_in,
        dropout_Gs=config.dropout_Gs,
        disc_type=config.disc_type,
        disc_hidden_dim=config.disc_hidden_dim,
        device=config.device,
        )
    GAN.train(cfg=config, hp_search=True)

sweep_config = {
    'method': 'random'
}

sweep_config['parameters'] = hp_search_trainer_config
sweep_id = wandb.sweep(sweep_config, project="search-hp-SimpleGAN")
wandb.agent(sweep_id, function=lambda: setup_and_train(), project="search-hp-SimpleGAN", count=args.n_runs)

print(f"Hyperparameter Search complete for {NAME}", flush=True)
