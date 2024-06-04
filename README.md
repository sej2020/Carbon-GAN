# Carbon-GAN
Creating a GAN that enables reduced storage costs of time series datasets

How to configure this repo:
1. clone the repository
2. install miniconda - https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html
3. run `conda env create -f environment.yml`
4. run `conda activate carbon-gan`
5. run `conda install -c conda-forge tensorboard` to install tensorboard

How to run tests: `python -m pytest tests/`