"""
Script for training a simple GAN model.

Typical usage example:
```bash
>>> python -m src.actions.train_simpleGAN --region "AUS_QLD" --elec_source "solar"
```
"""

import argparse

parser = argparse.ArgumentParser(description='Train a simple GAN')

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