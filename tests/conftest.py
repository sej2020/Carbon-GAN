import pytest

from src.config.trainer_configs import TrainerConfig
from src.utils.data import CarbonDataset

@pytest.fixture
def simple_training_config():
    yield TrainerConfig(
        region="CISO",
        elec_source="hydro",
        n_epochs=5,
        batch_size=128,
        lr_Gs=0.005,
        lr_D=0.001,
        k=1,
        run_name="TEMP_SIMPLE_GAN",
        lr_scheduler=True,
        disable_tqdm=True,
        logging_frequency=0.2,
        resume_from_cpt=False,
        cpt_path=None,
        debug=False
    )

@pytest.fixture
def fpl_other_training_set():
    yield CarbonDataset("FPL", "other", mode="train")

@pytest.fixture
def fpl_other_test_set():
    yield CarbonDataset("FPL", "other", mode="test")