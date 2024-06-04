"""
Classes for quantitative evaluation of a GAN model. Most metrics are methods under the QuantEvaluation class, which is instantiated 
with a model and dataset. The JCFE class is a separate class that is called on a model and dataset to evaluate the model's ability 
to model temporal relationships in the data and requires data generation conditional on a window of metadata and sequential data, unlike 
the other metrics.

The QuantEvaluation class has the following methods:
- Coverage - Ref: https://github.com/tolstikhin/adagan/blob/master/metrics.py
- Bin Difference
- Post-Hoc Discriminator Performance
- Anomaly Detection Performance (train on generated, predict on real)
- Forecasting Performance (train on generated, predict on real)

Typical usage example:
```python
>>> from src.models.GANs import MCGAN
>>> from src.utils.data import CarbonDataset
>>> model = MCGAN(metadata_dim=8, window_size=24, cpt_path="path/to/checkpoint.pt")
>>> dataset = CarbonDataset("CISO", "hydro", mode="test")
>>> quant = QuantEvaluation(model, dataset, 1000)
>>> print(quant.coverage())
```
"""

import torch
import numpy as np
from sklearn.neighbors._kde import KernelDensity

torch.set_printoptions(sci_mode=False)


class QuantEvaluation:
    """
    Stores methods for quantitative evaluation of a GAN model.

    Attributes:
        model: The model to be evaluated
        dataset: The dataset to be used for evaluation
        n_samples: The number of samples to generate from the model for comparison
        gen_meta: The metadata generated by the model
        gen_seq: The sequential data generated by the model
        real_meta: The metadata from the dataset
        real_seq: The sequential data from the dataset

    Methods:
        coverage: 'Computes the probability mass of the true data "covered" by the 95th quantile of the model density'
        bin_difference: Calculates the average absolute difference between the bin values of the real and generated data
    """
    def __init__(self, model, dataset, n_samples=1000):
        """
        Initializes the QuantEvaluation class.

        Args:
            model: The model to be evaluated with a data generation method
            dataset: The dataset to be used for evaluation
            n_samples: The number of samples to generate from the model for comparison
        """
        self.model = model
        self.dataset = dataset
        if model.generates_metadata:
            self.gen_meta, self.gen_seq = self.model.generate(n_samples, og_scale=False)
            self.real_meta, self.real_seq = self.dataset.metadata, self.dataset.seq_data
        else:
            self.gen_seq = self.model.generate(n_samples, og_scale=False)
            self.real_seq = self.dataset.seq_data


    def coverage(self) -> tuple[float, float] | float:
        """
        'Computes the probability mass of the true data "covered" by the 95th quantile of the model density.'
        The model density is estimated using a Gaussian kernel density estimator.
        
        Returns:
            C_meta: The coverage statistic for metadata
            C_seq: The coverage statistic for the sequential data
        """
        # Generated data
        gen_seq = self.gen_seq.flatten().unsqueeze(1).detach().numpy()
        kde_gen_seq = KernelDensity(kernel='gaussian', bandwidth='silverman')
        kde_gen_seq.fit(gen_seq) # [n_samples, 1]
        # Real data
        real_seq = self.real_seq.detach().numpy()
        # Computing Coverage:
        gen_log_density_seq = kde_gen_seq.score_samples(gen_seq)

        threshold_seq = np.percentile(gen_log_density_seq, 5)
        real_log_density_seq = kde_gen_seq.score_samples(real_seq)
        ratio_not_covered_seq = np.mean(real_log_density_seq <= threshold_seq)
        C_seq = 1. - ratio_not_covered_seq

        if not self.model.generates_metadata:
            return C_seq
        else:
            gen_meta = self.gen_meta.detach().numpy()
            kde_gen_meta = KernelDensity(kernel='gaussian', bandwidth='silverman')
            kde_gen_meta.fit(gen_meta) # [n_samples, n_features]
            # Real data
            real_meta = self.real_meta.detach().numpy()
            # Computing Coverage:
            gen_log_density_meta = kde_gen_meta.score_samples(gen_meta) # [batch, 1]

            # np.percentaile(a, 10) returns t s.t. np.mean( a <= t ) = 0.1
            threshold_meta = np.percentile(gen_log_density_meta, 5)
            real_log_density_meta = kde_gen_meta.score_samples(real_meta)
            ratio_not_covered_meta = np.mean(real_log_density_meta <= threshold_meta)
            C_meta = 1. - ratio_not_covered_meta
            
            return C_meta, C_seq


    def bin_difference(self, n_bins: int = 50) -> tuple[np.ndarray, np.float64] | np.float64:
        """
        Calculates the average absolute difference between the bin values of the real and generated sequential data (and metadata). 
        The bin values are calculated by histogramming the seq data (and metadata) into n_bins bins, and applying a density normalization
        to remove the effect of bin size and number of samples.

        Args:
            n_bins: The number of bins to use for the histogram

        Returns:
            (diff_meta): The average absolute difference between the bin values of the real and generated metadata if model generates metadata
            diff_seq: The average absolute difference between the bin values of the real and generated sequential data
        """
        gen_seq = self.gen_seq.flatten().unsqueeze(1).detach().numpy()
        real_seq = self.real_seq.detach().numpy()

        # sequence of the left edge of n_bins bins equally spaced in the range of real_meta + one right edge
        bins = np.linspace(real_seq.min(), real_seq.max(), n_bins)
        real_vals = np.histogram(real_seq, bins=bins, density=True)[0]
        gen_vals = np.histogram(gen_seq, bins=bins, density=True)[0]
        diff_seq = np.mean(np.abs(real_vals - gen_vals))

        if self.model.generates_metadata:
            gen_meta = self.gen_meta.detach().numpy()
            real_meta = self.real_meta.detach().numpy()
            diff_meta = np.zeros(real_meta.shape[1])
            for i in range(real_meta.shape[1]):
                bins = np.linspace(real_meta[:, i].min(), real_meta[:, i].max(), n_bins)
                real_vals = np.histogram(real_meta[:, i], bins=bins, density=True)[0]
                gen_vals = np.histogram(gen_meta[:, i], bins=bins, density=True)[0]
                diff_meta[i] = np.mean(np.abs(real_vals - gen_vals))

            return diff_meta, diff_seq
        else:
            return diff_seq


class JCFE:
    def __init__(self, model, dataset, n_samples=1000):
        """
        A bit different than the other guys. TBD
        """
        self.model = model
        self.dataset = dataset
        self.n_samples = n_samples

    def __call__(self):
        """
        """
        rand_idx = np.random.randint(0, len(self.dataset))
        (m, x_t), window_meta_and_seq = self.dataset[rand_idx], self.dataset[rand_idx-24:rand_idx]
        # m: [1, metadata_dim], x_t: [1, 1], window_meta_and_data: ([window_size, metadata_dim], [window_size, 1])
        self.model.generate(self.n_samples, og_scale=False, conditional_metadata=window_meta_and_seq)



# if __name__ == '__main__':
#     from src.models.GANs import SimpleGAN
#     from src.utils.data import CarbonDataset
#     model1 = SimpleGAN(window_size=24, n_seq_gen_layers=1, cpt_path="logs\debug\CISO-hydro-2024-06-03_14-43-34\checkpoints\checkpt_e9.pt")
#     model2 = SimpleGAN(window_size=24, n_seq_gen_layers=1, cpt_path="logs\debug\CISO-hydro-2024-06-03_14-43-34\checkpoints\checkpt_e119.pt")
#     model3 = SimpleGAN(window_size=24, n_seq_gen_layers=1, cpt_path="logs\debug\CISO-hydro-2024-06-03_14-43-34\checkpoints\checkpt_e299.pt")
#     dataset = CarbonDataset("CISO", "hydro", mode="test")
#     quant = QuantEvaluation(model3, dataset, 100)
#     print(quant.coverage())
#     print(quant.bin_difference())
