"""
Quantitative evaluation of the model includes the following metrics:
- Coverage
- JCFE
- Post-Hoc Discriminator Performance
- Anomaly Detection Performance (train on generated, predict on real)
- Forecasting Performance (train on generated, predict on real)
"""

import torch
import abc
import numpy as np
from sklearn.neighbors._kde import KernelDensity
from src.utils.data import CarbonDataset

torch.set_printoptions(sci_mode=False)

class EvalBase:
    def __init__(self, model, dataset):
        """
        Initializes the EvalBase class.

        Args:
            model: The model to be evaluated with a data generation method
            dataset: The dataset to be used for evaluation
        """
        self.model = model
        self.dataset = dataset
    
    @abc.abstractmethod
    def calculate(self, n_samples):
        """
        Calculate the evaluation method by using the model to generate synthetic data and comparing it to the dataset.
        
        Args:
            n_samples: The number of samples to generate from the model for comparison
        """



class Coverage(EvalBase):
    def __init__(self, model, dataset):
        super().__init__(model, dataset)

    def calculate(self, n_samples):
        """
        TBD
        """
        # Generated data
        gen_meta, gen_data = self.model.generate(n_samples, og_scale=False)
        gen_meta = gen_meta.detach().numpy()
        gen_data = gen_data.flatten().unsqueeze(1).detach().numpy()

        kde_gen_meta = KernelDensity(kernel='gaussian', bandwidth='silverman')
        kde_gen_data = KernelDensity(kernel='gaussian', bandwidth='silverman')
        
        kde_gen_meta.fit(gen_meta) # [n_samples, n_features]
        kde_gen_data.fit(gen_data) # [n_samples, 1]

        # Real data
        real_meta, real_data = self.dataset.metadata, self.dataset.data
        real_meta = real_meta.detach().numpy()
        real_data = real_data.detach().numpy()


        # Computing Coverage: Ref: https://github.com/tolstikhin/adagan/blob/master/metrics.py
        gen_log_density_meta = kde_gen_meta.score_samples(gen_meta) # [batch, 1]
        gen_log_density_data = kde_gen_data.score_samples(gen_data)


        # np.percentaile(a, 10) returns t s.t. np.mean( a <= t ) = 0.1
        threshold_meta = np.percentile(gen_log_density_meta, 5)
        threshold_data = np.percentile(gen_log_density_data, 5)


        real_log_density_meta = kde_gen_meta.score_samples(real_meta)
        real_log_density_data = kde_gen_data.score_samples(real_data)


        ratio_not_covered_meta = np.mean(real_log_density_meta <= threshold_meta)
        ratio_not_covered_data = np.mean(real_log_density_data <= threshold_data)
        
        C_meta = 1. - ratio_not_covered_meta
        C_data = 1. - ratio_not_covered_data
        
        return (C_meta, C_data), (np.mean(real_log_density_meta), np.mean(real_log_density_data))


class JCFE:
    def __init__(self, model, dataset):
        super().__init__(model, dataset)

    def calculate(self, n_samples):
        rand_idx = np.random.randint(0, len(self.dataset))
        (_, x_t), window_meta_and_data = self.dataset[rand_idx], [self.dataset[rand_idx-w] for w in self.model.window_size]
        # and then generate conditional on this
        breakpoint()
        self.model.generate(n_samples, og_scale=False, conditional_metadata=window_meta_and_data)






if __name__ == '__main__':
    from src.models.baseline import SimpleGAN
    torch.set_printoptions(sci_mode=False)
    gan = SimpleGAN(8, 24, cpt_path="logs/2024-05-30_19-34-39/2024-05-30_19-34-39/checkpoints/checkpt_e9.pt")
    region = 'AUS_QLD'
    elec_source = 'solar'
    data = CarbonDataset(region, elec_source, mode='test')
    cov = Coverage(model=gan, dataset=data)
    print(cov.calculate(100))

    