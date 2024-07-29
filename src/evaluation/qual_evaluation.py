"""
This class produces plots for qualitative evaluation of a GAN model.

The plots include:
- t-SNE
- PCA
- Autocorrelation
- Moving Average
- Histograms

Typical usage example:
```python
>>> from src.models.baseline import SimpleGAN
>>> from src.utils.data import CarbonDataset
>>> model = SimpleGAN(metadata_dim=8, window_size=24, cpt_path="path/to/checkpoint.pt")
>>> dataset = CarbonDataset("CISO", "hydro", mode="test")
>>> qual = QualEvaluation(model, dataset, 1000)
>>> qual.plot_histograms()
```
"""
import pathlib
import numpy as np
import matplotlib.pyplot as plt


class QualEvaluation:
    """
    Stores methods for qualitative evaluation of a GAN model.

    Attributes:
        model: The gan model to be evaluated
        dataset: The dataset to be used for evaluation
        n_samples: The number of samples to generate from the model for comparison
        gen_meta: The metadata generated by the model
        gen_seq: The sequential data generated by the model
        real_meta: The metadata from the dataset
        real_seq: The sequential data from the dataset
    
    Methods:
        plot_tsne: TBD
        plot_pca: TBD
        plot_autocorr: TBD
        plot_moving_avg: TBD
        plot_histograms: creates two figures, one for metadata and one for data, with density histograms for generated data and real data
    """
    def __init__(self, model, dataset, n_samples=1000):
        """
        Initializes the QualEvaluation class.

        Args:
            model: The gan model to be evaluated, must have a data generation method
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
            self.gen_meta = None
            self.real_meta = None
    

    def plot_tsne(self, save: bool = False, save_dir: str = None):
        """
        To be implemented when the model is generating multivariate time series data.
        """
        raise NotImplementedError
    

    def plot_pca(self, save: bool = False, save_dir: str = None):
        """
        To be implemented when the model is generating multivariate time series data. 
        """
        raise NotImplementedError
    

    def plot_autocorr(self, save: bool = False, save_dir: str = None):
        """
        Plots the averaged autocorrelation of many samples of generated and real data. 

        Args:
            save: Whether to save the plot
            save_dir: The directory to save the plot to
        """
        gen_seq = self.gen_seq.detach().numpy() # [n_samples, window_size]
        total_auto_corr = np.zeros((2*gen_seq.shape[1]-1))
        for row in range(gen_seq.shape[0]): 
            lags, c, line, b = plt.acorr(gen_seq[row], maxlags=None)
            total_auto_corr += c
        total_auto_corr /= gen_seq.shape[0]
        plt.clf()

        real_seq_samples = np.zeros((self.n_samples, self.model.window_size))
        for i in range(self.n_samples):
            rand_idx = np.random.randint(self.model.window_size, self.real_seq.shape[0])
            sample = self.real_seq[rand_idx - self.model.window_size : rand_idx]
            real_seq_samples[i] = sample.flatten()
        
        total_auto_corr_real = np.zeros((2*real_seq_samples.shape[1]-1))
        for row in range(real_seq_samples.shape[0]): 
            lags, c, line, b = plt.acorr(real_seq_samples[row], maxlags=None)
            total_auto_corr_real += c
        total_auto_corr_real /= real_seq_samples.shape[0]
        plt.clf()
        len_win = self.model.window_size - 1
        plt.plot(lags[len_win:], total_auto_corr[len_win:], label='Generated')
        plt.plot(lags[len_win:], total_auto_corr_real[len_win:], label='Real')
        plt.title("Autocorrelation of Generated and Real Data")
        plt.xlabel("Lag")
        plt.ylabel("Autocorrelation")
        plt.legend()

        if save:
            pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
            fig_name = f"{self.dataset.region}_{self.dataset.elec_source}_autocorr.png"
            plt.savefig(pathlib.PurePath(save_dir) / fig_name, bbox_inches='tight')
            plt.clf()
        else:
            plt.show()
    

    def plot_moving_avg(self, save: bool = False, save_dir: str = None):
        """
        TBD  
        """
        raise NotImplementedError


    def plot_histograms(self, n_bins: int = 50, save: bool = False, save_dir: str = None):
        """
        Creates two figures, one for metadata and one for data, with density histograms for generated data and real data.

        Args:
            n_bins: The number of bins to use in the histograms
            save: Whether to save the plots
            save_dir: The directory to save the plots to
        """
        gen_seq = self.gen_seq.flatten().unsqueeze(1).detach().numpy() # [n_samples, 1]
        real_seq = self.real_seq.detach().numpy() # [n_samples, 1]

        real_seq_hist = plt.hist(real_seq, bins=n_bins, alpha=0.5, label='Real', density=True)
        gen_seq_hist = plt.hist(gen_seq, bins=real_seq_hist[1], alpha=0.5, label='Generated', density=True)
        plt.title("Distribution of Generated and Real Data")
        plt.ylabel("Density")
        plt.legend()

        if save:
            pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
            fig_name = f"{self.dataset.region}_{self.dataset.elec_source}_seq_hist.png"
            plt.savefig(pathlib.PurePath(save_dir) / fig_name, bbox_inches='tight')
            plt.clf()
        else:
            plt.show() 

        if self.model.generates_metadata:

            fig, axs = plt.subplots(4, 2)
            fig.suptitle("Distribution of Generated and Real Metadata")
            fig.supylabel("Density")

            gen_meta = self.gen_meta.detach().numpy() # [n_samples, dims]
            real_meta = self.real_meta.detach().numpy() # [n_samples, dims]

            for i in range(4):
                for j in range(2):
                    real_meta_hist = axs[i, j].hist(real_meta[:, i*2+j], bins=n_bins, alpha=0.5, label='Real', density=True)
                    gen_meta_hist = axs[i, j].hist(gen_meta[:, i*2+j], bins=real_meta_hist[1], alpha=0.5, label='Generated', density=True)
                    if i==0 and j ==1:
                        axs[i, j].legend()
                
            if save:
                pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
                fig_name = f"{self.dataset.region}_{self.dataset.elec_source}_meta_hist.png"
                plt.savefig(pathlib.PurePath(save_dir) / fig_name, bbox_inches='tight')
                plt.clf()
            else:
                plt.show()

        pass



if __name__ == "__main__":
    from src.models.GANs import SimpleGAN
    from src.utils.data import CarbonDataset
    model1 = SimpleGAN(window_size=24, n_seq_gen_layers=1, cpt_path="logs\CISO-hydro-2024-06-18_11-40-32--64176\checkpoints\checkpt_e199.pt")
    dataset = CarbonDataset("CISO", "nat_gas", mode="val")
    qual = QualEvaluation(model1, dataset, 1000)
    qual.plot_histograms()
