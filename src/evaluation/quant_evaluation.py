"""
Classes for quantitative evaluation of a GAN model. Most metrics are methods under the QuantEvaluation class, which is instantiated 
with a model and dataset

The QuantEvaluation class has the following methods:
- Coverage - Ref: https://github.com/tolstikhin/adagan/blob/master/metrics.py
- Bin Overlap
- Johnson Conditional Fidelity Estimate (JCFE)
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
        bin_overlap: Calculates the average overlap between the bin values of the real and generated data
        jcfe: Calculates the Johnson Conditional Fidelity Estimate of the model
        discriminator_accuracy: Trains a discriminator on sequences of real and generated data, and returns accuracy on a test set
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
        self.n_samples = n_samples
        if model.generates_metadata:
            self.gen_meta, self.gen_seq = self.model.generate(n_samples, og_scale=False)
            self.real_meta, self.real_seq = self.dataset.metadata, self.dataset.seq_data
        else:
            self.gen_seq = self.model.generate(n_samples, og_scale=False)
            self.real_seq = self.dataset.seq_data


    def coverage(self) -> tuple[float, float] | np.float64:
        """
        'Computes the probability mass of the true data "covered" by the 95th quantile of the model density.'
        The model density is estimated using a Gaussian kernel density estimator.
        
        Returns:
            C_meta: The coverage statistic for metadata
            C_seq: The coverage statistic for the sequential data

        Notes:
            Higher coverage values indicate better model performance
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


    def bin_overlap(self, n_bins: int = 50) -> tuple[np.ndarray, np.float64] | np.float64:
        """
        Calculates the average overlap of the bin values of the real and generated sequential data (and metadata). 
        The bin values are calculated by histogramming the seq data (and metadata) into n_bins bins, and applying a density normalization
        to remove the effect of bin size and number of samples.

        Args:
            n_bins: The number of bins to use for the histogram

        Returns:
            (1-diff_meta): The average overlap of the bin values of the real and generated metadata if model generates metadata
            1-diff_seq: The average overlap of the bin values of the real and generated sequential data
        
        Notes:
            Higher bin overlap values indicate better model performance
        """
        gen_seq = self.gen_seq.flatten().unsqueeze(1).detach().numpy()
        real_seq = self.real_seq.detach().numpy()

        # sequence of the left edge of n_bins bins equally spaced in the range of real_meta + one right edge
        bins = np.linspace(real_seq.min(), real_seq.max(), n_bins)
        real_vals = np.histogram(real_seq, bins=bins, density=True)[0]
        gen_vals = np.histogram(gen_seq, bins=bins, density=True)[0]
        diff_seq = np.sum(np.abs(real_vals - gen_vals)) / 2
        if self.model.generates_metadata:
            pass
            # TODO: Implement metadata bin overlap
        else:
            return 1-(diff_seq/sum(real_vals))


    def jcfe(self, gen_per_sample: int = 100) -> np.float64:
        """
        Calculates the Johnson Conditional Fidelity Estimate of the model. A point x_t is sampled from real data, along with its preceding
        points x_t_minus. The model is then conditioned on x_t_minus to generate gen_per_sample x_t_hats. A model probability density function
        conditioned on x_t_minus is created using a Gaussian kernel density estimator, and the probability of x_t in the x_t_hat distribution
        is calculated. This process is repeated n_samples times, and the average probability of x_t in the x_t_hat distribution is returned.

        Args: 
            gen_per_sample: The number of samples to generate per sample from the real data

        Returns:
            The Johnson Conditional Fidelity Estimate of the model

        Notes:
            Higher JCFE values indicate better model performance
        """
        if not self.model.generates_metadata:
            total_x_t_prob = 0
            for _ in range(self.n_samples):
                # sampling x_t from real data
                rand_idx = np.random.randint(self.model.window_size, len(self.dataset))
                sample = self.real_seq[rand_idx - self.model.window_size : rand_idx] # [window_size, 1]
                x_t_minus, x_t = sample.view(1,-1)[:,:-1], sample.view(1,-1)[:,-1] # [1, window_size-1], [1, 1]
                
                # generating x_t_hat conditioned on x_t_minus
                x_t_hat = self.model.generate(n_samples=gen_per_sample, generation_len=1, og_scale=False, condit_seq_data=x_t_minus) # [gen_per_sample, 1]    
                # creating x_t_hat distribution using KDE
                kde = KernelDensity(kernel='gaussian', bandwidth='silverman')
                kde.fit(x_t_hat.detach().numpy())
                # calculating the probability of x_t in x_t_hat distribution
                x_t_prob = np.exp(kde.score_samples(x_t.reshape(-1,1)))
                total_x_t_prob += x_t_prob.item()

            # calculating the average probability of x_t in x_t_hat distribution 
            jcfe = total_x_t_prob / self.n_samples
            return np.float64(jcfe)


    def discriminator_accuracy(self) -> np.float64:
        """
        Trains a discriminator on sequences of real and generated data, and returns accuracy on a test set.

        Returns:
            The accuracy of the discriminator on the generated data

        Notes:
            Lower discriminator error values indicate better model performance
        """
        gen_train_set = self.gen_seq
        real_train_set = torch.zeros((self.n_samples, self.model.window_size))
        for i in range(self.n_samples):
            rand_idx = np.random.randint(self.model.window_size, int(self.real_seq.shape[0]*0.66))
            sample = self.real_seq[rand_idx - self.model.window_size : rand_idx]
            real_train_set[i] = sample.squeeze()

        train_set = torch.cat((gen_train_set, real_train_set), dim=0)
        labels = torch.cat((torch.zeros(self.n_samples, dtype=torch.float64), torch.ones(self.n_samples, dtype=torch.float64)), dim=0)
        train_perm = torch.randperm(train_set.size()[0])
        train_set = train_set[train_perm]
        labels = labels[train_perm]
        
        half_test_size = int(self.n_samples/2)
        gen_test_set = self.model.generate(half_test_size, og_scale=False)
        real_test_set = torch.zeros((half_test_size, self.model.window_size))
        for i in range(half_test_size):
            rand_idx = np.random.randint(int(self.real_seq.shape[0]*0.66) + self.model.window_size, self.real_seq.shape[0])
            sample = self.real_seq[rand_idx - self.model.window_size : rand_idx]
            real_test_set[i] = sample.squeeze()
        
        test_set = torch.cat((gen_test_set, real_test_set), dim=0)
        test_labels = torch.cat((torch.zeros(half_test_size), torch.ones(half_test_size)), dim=0)
        test_perm = torch.randperm(test_set.size()[0])
        test_set = test_set[test_perm]
        test_labels = test_labels[test_perm]

        post_hoc_disc = torch.nn.LSTM(1, hidden_size=1, batch_first=True, dtype=torch.float64)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(post_hoc_disc.parameters(), lr=0.01)
        for epoch in range(500):
            batch_x = train_set.detach().unsqueeze(2)
            _, (h, _) = post_hoc_disc(batch_x)
            h_bin = torch.sigmoid(h.squeeze())
            loss = criterion(h_bin, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            _, (h, _) = post_hoc_disc(test_set.unsqueeze(2))
            h_bin = torch.sigmoid(h.squeeze())
            acc = torch.sum((h_bin.squeeze() > 0.5) == test_labels).item() / (2*half_test_size)
            return np.float64(acc)






if __name__ == '__main__':
    from src.models.GANs import SimpleGAN
    from src.utils.data import CarbonDataset
    model1 = SimpleGAN(window_size=24, n_seq_gen_layers=1, cpt_path="logs\CISO-hydro-2024-06-18_11-40-32--64176\checkpoints\checkpt_e199.pt")
    dataset = CarbonDataset("CISO", "nat_gas", mode="val")
    quant = QuantEvaluation(model1, dataset, 1000)
    print(quant.bin_overlap())
