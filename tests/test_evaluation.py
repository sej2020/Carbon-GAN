import pathlib
import numpy as np
import shutil
from src.models.GANs import SimpleGAN
from src.evaluation.qual_evaluation import QualEvaluation
from src.evaluation.quant_evaluation import QuantEvaluation
from src.config.trainer_configs import TrainerConfig

def setup_module():
    gan = SimpleGAN(window_size=12, n_seq_gen_layers=1)
    simple_training_config = TrainerConfig(
        region="CISO",
        elec_source="hydro",
        n_epochs=5,
        batch_size=128,
        lr_Gs=5e-4,
        lr_D=5e-4,
        run_name="TEMP_SIMPLE_GAN",
        lr_scheduler="adaptive",
        sup_loss=True,
        disable_tqdm=True,
        logging_frequency=0.2,
        resume_from_cpt=False,
        cpt_path=None,
        debug=False
    )
    gan.train(cfg=simple_training_config)


def teardown_module():
    temp_path = pathlib.Path("logs/TEMP_SIMPLE_GAN")
    shutil.rmtree(temp_path)


class TestQualEvalSimpleGAN:

    def test_histograms(self, fpl_other_test_set):
        gan = SimpleGAN(window_size=12, n_seq_gen_layers=1, cpt_path="logs/TEMP_SIMPLE_GAN/checkpoints/checkpt_e4.pt")
        qual_eval = QualEvaluation(gan, fpl_other_test_set, 100)
        assert qual_eval.gen_meta is None
        assert qual_eval.gen_seq is not None
        qual_eval.plot_histograms(n_bins=20, save=True, save_dir="logs/TEMP_SIMPLE_GAN/histograms")
        assert pathlib.Path.exists(pathlib.Path("logs/TEMP_SIMPLE_GAN/histograms/FPL_other_seq_hist.png"))

    def test_plot_autocorr(self, fpl_other_test_set):
        gan = SimpleGAN(window_size=12, n_seq_gen_layers=1, cpt_path="logs/TEMP_SIMPLE_GAN/checkpoints/checkpt_e4.pt")
        qual_eval = QualEvaluation(gan, fpl_other_test_set, 100)
        qual_eval.plot_autocorr(save=True, save_dir="logs/TEMP_SIMPLE_GAN/autocorr")
        assert pathlib.Path.exists(pathlib.Path("logs/TEMP_SIMPLE_GAN/autocorr/FPL_other_autocorr.png"))
  

class TestQuantEvalSimpleGAN:

    def test_coverage(self, fpl_other_test_set):
        gan = SimpleGAN(window_size=12, n_seq_gen_layers=1, cpt_path="logs/TEMP_SIMPLE_GAN/checkpoints/checkpt_e4.pt")
        quant_eval = QuantEvaluation(gan, fpl_other_test_set, 100)
        result = quant_eval.coverage()
        assert type(result) == np.float64
        assert result >= 0.0 and result <= 1.0

    def test_bin_overlap(self, fpl_other_test_set):
        gan = SimpleGAN(window_size=12, n_seq_gen_layers=1, cpt_path="logs/TEMP_SIMPLE_GAN/checkpoints/checkpt_e4.pt")
        quant_eval = QuantEvaluation(gan, fpl_other_test_set, 100)
        result = quant_eval.bin_overlap()
        assert type(result) == np.float64
        assert result >= 0.0 and result <= 1.0
    
    def test_jcfe(self, fpl_other_test_set):
        gan = SimpleGAN(window_size=12, n_seq_gen_layers=1, cpt_path="logs/TEMP_SIMPLE_GAN/checkpoints/checkpt_e4.pt")
        quant_eval = QuantEvaluation(gan, fpl_other_test_set, 100)
        result = quant_eval.jcfe()
        assert type(result) == np.float64
        assert result >= 0.0 and result <= 1.0

    def test_discriminator_accuracy(self, fpl_other_test_set):
        gan = SimpleGAN(window_size=12, n_seq_gen_layers=1, cpt_path="logs/TEMP_SIMPLE_GAN/checkpoints/checkpt_e4.pt")
        quant_eval = QuantEvaluation(gan, fpl_other_test_set, 100)
        result = quant_eval.discriminator_accuracy()
        assert type(result) == np.float64
        assert result >= 0.0 and result <= 1.0