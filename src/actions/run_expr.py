"""
Script for running multi-dataset experiment with GAN models.

Typical usage example:
```bash
>>> python -m src.actions.run_expr simple --n_jobs 7
```
"""
import multiprocessing as mp
import random
import argparse
import yaml
from src.config.trainer_configs import TrainerConfig
from src.models.GANs import SimpleGAN
from src.evaluation.qual_evaluation import QualEvaluation
from src.evaluation.quant_evaluation import QuantEvaluation
from src.utils.data import CarbonDataset

parser = argparse.ArgumentParser(description='Multi-dataset GAN experiment')

parser.add_argument("model_type", type=str, choices=["simple"], help="Model to train")
parser.add_argument("--disable_tqdm", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--logging_frequency", type=float, default=0.05)
parser.add_argument("--n_jobs", type=int, default=1)

args = parser.parse_args()

def name_and_train(config):
    gan = SimpleGAN(
        window_size=24, 
        n_seq_gen_layers=1, 
        dropout_D_hid=0.6,
        dropout_D_in=0.2
        )
    # adding random number to run name to avoid conflicts
    config.run_name = config.run_name + f"--{random.randint(0, 1000000)}"
    return gan.train(config)

if __name__ == "__main__":
    results_dict = {}
    data_list = [
        ("AUS_QLD","coal"), ("BPAT","nuclear"), ("CISO","nat_gas"), ("DE","geothermal"), ("ERCO","solar"),
        ("ES","biomass"), ("FPL","other"), ("ISNE","hydro"), ("NL","wind"), ("NYISO","oil")
    ]
    for reg, source in data_list:
        if args.model_type == "simple":
            config = TrainerConfig(
                region=reg,
                elec_source=source,
                n_epochs=1600,
                batch_size=1024,
                lr_Gs=1e-3,
                lr_D=1e-3,
                run_name = f"{reg}-{source}",
                sup_loss=True,
                eta=0.2,
                lr_scheduler="adaptive",
                disable_tqdm=args.disable_tqdm,
                logging_dir="logs/simple",
                logging_frequency=args.logging_frequency,
            )
            if args.n_jobs > 1:
                with mp.Pool(args.n_jobs) as pool:
                    best_performers = pool.imap(name_and_train, [config for _ in range(7)])
                    best_performers = list(best_performers)
            else:
                best_performers = []
                for _ in range(7):
                    best_performers.append(name_and_train(config))
            best_performers.sort(key=lambda x: x[1], reverse=True)
            best_result = best_performers[0]
            test_data = CarbonDataset(region=reg, elec_source=source, mode="test")
            test_model = SimpleGAN(
                window_size=24, 
                n_seq_gen_layers=1, 
                dropout_D_hid=0.6, 
                dropout_D_in=0.2,
                cpt_path=f"{best_result[0]}checkpoints/checkpt_e{best_result[2]}.pt"
                )
            quant = QuantEvaluation(test_model, test_data, 1000)
            results_dict[f"{reg}-{source}"] = {}
            results_dict[f"{reg}-{source}"]["training_combined_score"] = best_result[1].item()
            results_dict[f"{reg}-{source}"]["coverage"] = float(quant.coverage())
            results_dict[f"{reg}-{source}"]["bin_overlap"] = float(quant.bin_overlap())
            results_dict[f"{reg}-{source}"]["disc_acc"] = float(quant.discriminator_accuracy())
            results_dict[f"{reg}-{source}"]["jcfe"] = float(quant.jcfe())

            qual = QualEvaluation(test_model, test_data, 1000)
            qual.plot_histograms(save=True, save_dir=f"expr_results/simple")
            qual.plot_autocorr(save=True, save_dir=f"expr_results/simple")
            
    yaml.dump(results_dict, open("expr_results/simple/results.yaml", "w"))
    expr_setup_dict = {"model_type": args.model_type, "n_datasets": len(data_list), "n_models": 7, "testing_sample_size": 1000}
    yaml.dump(expr_setup_dict, open("expr_results/simple/setup.yaml", "w"))