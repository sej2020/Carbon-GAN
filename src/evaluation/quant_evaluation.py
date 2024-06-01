"""
Classes for quantitative evaluation of a GAN model. Most metrics are methods under the QuantEvaluation class, which is instantiated 
with a model and dataset. The JCFE class is a separate class that is called on a model and dataset to evaluate the model's ability 
to model temporal relationships in the data and requires data generation conditional on a window of metadata and data, unlike the 
other metrics.

The QuantEvaluation class has the following methods:
- Coverage - Ref: https://github.com/tolstikhin/adagan/blob/master/metrics.py
- Bin Difference
- Post-Hoc Discriminator Performance
- Anomaly Detection Performance (train on generated, predict on real)
- Forecasting Performance (train on generated, predict on real)

Typical usage example:
```python
>>> from src.models.baseline import SimpleGAN
>>> from src.utils.data import CarbonDataset
>>> model = SimpleGAN(metadata_dim=8, window_size=24, cpt_path="path/to/checkpoint.pt")
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
        gen_data: The temporal data generated by the model
        real_meta: The metadata from the dataset
        real_data: The temporal data from the dataset

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
        self.gen_meta, self.gen_data = self.model.generate(n_samples, og_scale=False)
        self.real_meta, self.real_data = self.dataset.metadata, self.dataset.data


    def coverage(self) -> tuple[float, float]:
        """
        'Computes the probability mass of the true data "covered" by the 95th quantile of the model density.'
        The model density is estimated using a Gaussian kernel density estimator.
        
        Returns:
            C_meta: The coverage statistic for metadata, averaged across features
            C_data: The coverage statistic for the temporal data
        """
        # Generated data
        gen_meta = self.gen_meta.detach().numpy()
        gen_data = self.gen_data.flatten().unsqueeze(1).detach().numpy()

        kde_gen_meta = KernelDensity(kernel='gaussian', bandwidth='silverman')
        kde_gen_data = KernelDensity(kernel='gaussian', bandwidth='silverman')
        
        kde_gen_meta.fit(gen_meta) # [n_samples, n_features]
        kde_gen_data.fit(gen_data) # [n_samples, 1]

        # Real data
        real_meta = self.real_meta.detach().numpy()
        real_data = self.real_data.detach().numpy()

        # Computing Coverage:
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
        
        return C_meta, C_data


    def bin_difference(self, n_bins: int = 50) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the average absolute difference between the bin values of the real and generated temporal data and metadata. 
        The bin values are calculated by histogramming the data and metadata into n_bins bins, and applying a density normalization
        to remove the effect of bin size and number of samples.

        Args:
            n_bins: The number of bins to use for the histogram

        Returns:
            diff_meta: The average absolute difference between the bin values of the real and generated metadata
            diff_data: The average absolute difference between the bin values of the real and generated temporal data
        """
        gen_meta = self.gen_meta.detach().numpy()
        gen_data = self.gen_data.flatten().unsqueeze(1).detach().numpy()
        
        real_meta = self.real_meta.detach().numpy()
        real_data = self.real_data.detach().numpy()

        # sequence of the left edge of n_bins bins equally spaced in the range of real_meta + one right edge
        bins = np.linspace(real_data.min(), real_data.max(), n_bins)
        real_vals = np.histogram(real_data, bins=bins, density=True)[0]
        gen_vals = np.histogram(gen_data, bins=bins, density=True)[0]
        diff_data = np.mean(np.abs(real_vals - gen_vals))

        diff_meta = np.zeros(real_meta.shape[1])
        for i in range(real_meta.shape[1]):
            bins = np.linspace(real_meta[:, i].min(), real_meta[:, i].max(), n_bins)
            real_vals = np.histogram(real_meta[:, i], bins=bins, density=True)[0]
            gen_vals = np.histogram(gen_meta[:, i], bins=bins, density=True)[0]
            diff_meta[i] = np.mean(np.abs(real_vals - gen_vals))

        return diff_meta, diff_data


class JCFE:
    def __init__(self, model, dataset, n_samples=1000):
        """
        A bit different than the other guys. TBD
        """

    def __call__(self):
        """
        """
        rand_idx = np.random.randint(0, len(self.dataset))
        (_, x_t), window_meta_and_data = self.dataset[rand_idx], [self.dataset[rand_idx-w] for w in self.model.window_size]
        # and then generate conditional on this
        breakpoint()
        self.model.generate(self.n_samples, og_scale=False, conditional_metadata=window_meta_and_data)



if __name__ == '__main__':
    from src.models.baseline import SimpleGAN
    from src.utils.data import CarbonDataset
    torch.set_printoptions(sci_mode=False)
    model1 = SimpleGAN(metadata_dim=8, window_size=24, cpt_path="logs/debug/CISO-hydro-2024-05-31_13-48-06/checkpoints/checkpt_e29.pt")
    model2 = SimpleGAN(metadata_dim=8, window_size=24, cpt_path="logs/debug/CISO-hydro-2024-05-31_13-48-06/checkpoints/checkpt_e219.pt")
    model3 = SimpleGAN(metadata_dim=8, window_size=24, cpt_path="logs/debug/CISO-hydro-2024-05-31_13-48-06/checkpoints/checkpt_e699.pt")
    dataset = CarbonDataset("CISO", "hydro", mode="test")
    
    quant = QuantEvaluation(model3, dataset, 1000)

    print(quant.coverage())
    print(quant.bin_difference())
    