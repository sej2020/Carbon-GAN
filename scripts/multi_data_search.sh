#!/bin/bash

python -m src.actions.search_hp simple --device cuda:0 --n_runs 100 --region AUS_QLD --elec_source coal
python -m src.actions.search_hp simple --device cuda:0 --n_runs 100 --region BPAT --elec_source nuclear
python -m src.actions.search_hp simple --device cuda:0 --n_runs 100 --region CISO --elec_source nat_gas
python -m src.actions.search_hp simple --device cuda:0 --n_runs 100 --region DE --elec_source geothermal
python -m src.actions.search_hp simple --device cuda:0 --n_runs 100 --region ERCO --elec_source solar
python -m src.actions.search_hp simple --device cuda:0 --n_runs 100 --region ES --elec_source biomass
python -m src.actions.search_hp simple --device cuda:0 --n_runs 100 --region FPL --elec_source other
python -m src.actions.search_hp simple --device cuda:0 --n_runs 100 --region ISNE --elec_source hydro
python -m src.actions.search_hp simple --device cuda:0 --n_runs 100 --region NL --elec_source wind
python -m src.actions.search_hp simple --device cuda:0 --n_runs 100 --region NYSIO --elec_source oil


