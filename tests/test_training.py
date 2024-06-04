import pathlib
import shutil
import torch
from src.models.GANs import SimpleGAN

class TestTrainingSimpleGAN:

    def test_one_layer_from_scratch(self, simple_training_config):
        one_layer_gan = SimpleGAN(window_size=12, n_seq_gen_layers=1)
        one_layer_gan.train(cfg=simple_training_config)
        assert pathlib.Path.exists(pathlib.Path("logs/TEMP_SIMPLE_GAN/checkpoints/checkpt_e4.pt"))
        assert pathlib.Path.exists(pathlib.Path("logs/TEMP_SIMPLE_GAN/scalers"))
        assert pathlib.Path.exists(pathlib.Path("logs/TEMP_SIMPLE_GAN/tensorboard"))

        single_gen = one_layer_gan.generate(n_samples=1)
        assert single_gen.shape == (1, 12)
        
        condit_data = torch.randn(1,8)
        condit_gen = one_layer_gan.generate(n_samples=100, condit_seq_data=condit_data)
        assert condit_gen.shape == (100, 12)
        
        temp_path = pathlib.Path("logs/TEMP_SIMPLE_GAN")
        shutil.rmtree(temp_path)


    def test_two_layers_from_scratch(self, simple_training_config):
        two_layer_gan = SimpleGAN(window_size=6, n_seq_gen_layers=2)
        two_layer_gan.train(cfg=simple_training_config)
        assert pathlib.Path.exists(pathlib.Path("logs/TEMP_SIMPLE_GAN/checkpoints/checkpt_e4.pt"))
        assert pathlib.Path.exists(pathlib.Path("logs/TEMP_SIMPLE_GAN/scalers"))
        assert pathlib.Path.exists(pathlib.Path("logs/TEMP_SIMPLE_GAN/tensorboard"))

        mult_gen = two_layer_gan.generate(n_samples=100)
        assert mult_gen.shape == (100, 6)
        
        condit_data = torch.randn(100,6)
        condit_gen = two_layer_gan.generate(n_samples=100, generation_len=12, condit_seq_data=condit_data)
        assert condit_gen.shape == (100, 12)
        
        temp_path = pathlib.Path("logs/TEMP_SIMPLE_GAN")
        shutil.rmtree(temp_path)


    def test_resume_training(self, simple_training_config):
        gan = SimpleGAN(window_size=12, n_seq_gen_layers=1)
        simple_training_config.n_epochs = 10
        gan.train(cfg=simple_training_config)
        assert pathlib.Path.exists(pathlib.Path("logs/TEMP_SIMPLE_GAN/checkpoints/checkpt_e4.pt"))
        assert pathlib.Path.exists(pathlib.Path("logs/TEMP_SIMPLE_GAN/scalers"))
        assert pathlib.Path.exists(pathlib.Path("logs/TEMP_SIMPLE_GAN/tensorboard"))

        simple_training_config.resume_from_cpt = True
        simple_training_config.cpt_path = "logs/TEMP_SIMPLE_GAN/checkpoints/checkpt_e9.pt"
        gan.train(cfg=simple_training_config)
        assert pathlib.Path.exists(pathlib.Path("logs/TEMP_SIMPLE_GAN/checkpoints/checkpt_e9.pt"))
        
        temp_path = pathlib.Path("logs/TEMP_SIMPLE_GAN")
        shutil.rmtree(temp_path)