import os
from pathlib import Path
import torch

from mrnet.training.trainer import MRTrainer
from mrnet.datasets.signals import Signal1D
from mrnet.networks.mrnet import MRFactory
from mrnet.training.listener import TrainingListener
from mrnet.datasets.procedural import perlin_noise

from mrnet.training.utils import load_hyperparameters, get_optim_handler
from IPython import embed


if __name__ == '__main__':
    nfeatures = [32, 64, 128, 256]
    omegas = [8, 32, 128, 256, 512]
    h_layers = [1, 2, 3]
    noise_params = { "scale": 10, "octaves": 16, "p": 1.4}
    
    for features in nfeatures:
        for layers in h_layers:
            for omega_0 in omegas:
                torch.manual_seed(777)
                #-- hyperparameters in configs --#
                hyper = load_hyperparameters('experiments/ch4/configs/noise.yml')
                hyper['hidden_features'] = features
                hyper['hidden_layers'] = layers
                hyper['omega_0'] = omega_0
                
                project_name = hyper['project_name']
                nsamples = hyper['nsamples']
                
                X = torch.linspace(-1, 1, nsamples)
                scale = noise_params['scale']
                octaves = noise_params['octaves']
                p = noise_params['p']
                noise = perlin_noise(nsamples, scale, octaves, p)
                
                base_signal = Signal1D(noise.view(1, -1), 
                                    domain=hyper['domain'],
                                    batch_size=hyper['batch_size'])
                train_dataset = [base_signal]
                test_dataset = [base_signal]

                # you can substitute this line by your custom handler class
                optim_handler = get_optim_handler(hyper.get('optim_handler', 'regular'))
                
                mrmodel = MRFactory.from_dict(hyper)
                print("Model: ", type(mrmodel))
                print(mrmodel)

                name = f"noise-s{scale}-o{octaves}-p{p}"
                training_listener = TrainingListener(project_name,
                                            f"{name}{hyper['model']}",
                                            hyper,
                                            Path(hyper.get("log_path", "runs")))

                mrtrainer = MRTrainer.init_from_dict(mrmodel,
                                                    train_dataset,
                                                    test_dataset,
                                                    training_listener,
                                                    hyper,
                                                    optim_handler=optim_handler)
                mrtrainer.train(hyper['device'])