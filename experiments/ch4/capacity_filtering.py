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

def make_noise(hyper):
    nsamples = hyper['nsamples']
    X = torch.linspace(-1, 1, nsamples)
    scale = hyper['scale']
    octaves = hyper['octaves']
    p = hyper['p']
    noise = perlin_noise(nsamples, scale, octaves, p)
    
    base_signal = Signal1D(noise.view(1, -1), 
                        domain=hyper['domain'],
                        batch_size=hyper['batch_size'])
    return base_signal

def training_pipeline(hyper):
    project_name = hyper['project_name']
    scale, octaves, p = hyper['scale'], hyper['octaves'], hyper['p']
    name = f"cap-s{scale}-o{octaves}-p{p}"
    
    base_signal = make_noise(hyper)
    train_dataset = [base_signal]
    test_dataset = [base_signal]

    # you can substitute this line by your custom handler class
    optim_handler = get_optim_handler(hyper.get('optim_handler', 'regular'))
    
    mrmodel = MRFactory.from_dict(hyper)
    print("Model: ", type(mrmodel))
    print(mrmodel)
    
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


if __name__ == '__main__':
    nfeatures = [256] #[32, 64, 128, 256]
    omegas = [2, 4, 8, 16, 32, 64] #[8, 32, 64, 128, 256, 512, 1024]
    h_layers = [1]#, 1, 2]#, 3]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for features in nfeatures:
        for layers in h_layers:
            for omega_0 in omegas:
                torch.manual_seed(777)
                #-- hyperparameters in configs --#
                hyper = load_hyperparameters('experiments/ch4/configs/capacity.yml')
                hyper['device'] = device
                hyper['hidden_features'] = features
                hyper['hidden_layers'] = layers
                hyper['omega_0'] = omega_0 * 2 * torch.pi
                
                training_pipeline(hyper)