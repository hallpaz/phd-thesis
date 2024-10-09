import os
from pathlib import Path
import torch
import random
import string

from mrnet.training.trainer import MRTrainer
from mrnet.datasets.signals import Signal1D
from mrnet.networks.mrnet import MRFactory
from mrnet.training.listener import TrainingListener
from mrnet.datasets.procedural import perlin_noise
from mrnet.datasets.pyramids import create_MR_structure

from mrnet.training.utils import load_hyperparameters, get_optim_handler
from IPython import embed
import wandb

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

# def training_pipeline(hyper):
#     project_name = hyper['project_name']
#     scale, octaves, p = hyper['scale'], hyper['octaves'], hyper['p']
#     # name = f"fit-s{scale}-o{octaves}-p{p}"
#     NUM_LEVELS = 2
#     OMEGAS = [128, 24]
#     H_FEATURES = [128, 64]
#     base_signal = make_noise(hyper)
#     train_dataset = create_MR_structure(base_signal, 
#                                         NUM_LEVELS,
#                                         hyper['filter'],
#                                         hyper['decimation'])
#     test_dataset = create_MR_structure(base_signal, 
#                                         NUM_LEVELS,
#                                         hyper['filter'],
#                                         False)

#     # you can substitute this line by your custom handler class
#     optim_handler = get_optim_handler(hyper.get('optim_handler', 'regular'))

    
#     for i in range(NUM_LEVELS):
#         torch.manual_seed(7777)
        
#         current_train = [train_dataset[i]]
#         current_test = [test_dataset[i]]
#         hyper['nsamples'] = current_train[0].size()[-1]
#         # omega_0 = hyper['nsamples'] // 16
#         # omega_0 = hyper['omega_0']
#         omega_0 = OMEGAS[i]
#         print("OMEGA_0 hz:", omega_0)
#         hyper['omega_0'] = omega_0 * 2 * torch.pi

#         hyper['hidden_features'] = [H_FEATURES[i]]
        
#         mrmodel = MRFactory.from_dict(hyper)
#         print("Model: ", type(mrmodel))
#         print(mrmodel)
#         name = "fit" + ''.join(
#             random.choices(string.ascii_uppercase + string.digits, k=4))
#         training_listener = TrainingListener(project_name,
#                                     f"{name}{hyper['model']}",
#                                     hyper,
#                                     Path(hyper.get("log_path", "runs")))

#         mrtrainer = MRTrainer.init_from_dict(mrmodel,
#                                             current_train,
#                                             current_test,
#                                             training_listener,
#                                             hyper,
#                                             optim_handler=optim_handler)
#         mrtrainer.train(hyper['device'])

def training_pipeline(hyper):
    project_name = hyper['project_name']
    scale, octaves, p = hyper['scale'], hyper['octaves'], hyper['p']
    # name = f"fit-s{scale}-o{octaves}-p{p}"
    NUM_LEVELS = 4
    OMEGAS = [128, 24]
    H_FEATURES = [128, 64]
    base_signal = make_noise(hyper)
    train_dataset = create_MR_structure(base_signal, 
                                        NUM_LEVELS,
                                        hyper['filter'],
                                        hyper['decimation'])
    test_dataset = create_MR_structure(base_signal, 
                                        NUM_LEVELS,
                                        hyper['filter'],
                                        False)

    # you can substitute this line by your custom handler class
    optim_handler = get_optim_handler(hyper.get('optim_handler', 'regular'))
        
    current_train = [train_dataset[NUM_LEVELS-1]]
    # current_train = [test_dataset[NUM_LEVELS-1]]
    current_test = [test_dataset[NUM_LEVELS-1]]
    hyper['nsamples'] = current_train[0].size()[-1]
    # omega_0 = hyper['nsamples'] // 16
    # omega_0 = hyper['omega_0']
    omega_0 = hyper['omega_0']
    # print("OMEGA_0 hz:", omega_0)
    hyper['omega_0'] = omega_0 * 2 * torch.pi

    # hyper['hidden_features'] = [H_FEATURES[i]]
    
    mrmodel = MRFactory.from_dict(hyper)
    print("Model: ", type(mrmodel))
    print(mrmodel)
    name = "fit" + ''.join(
        random.choices(string.ascii_uppercase + string.digits, k=4))
    training_listener = TrainingListener(project_name,
                                f"{name}{hyper['model']}",
                                hyper,
                                Path(hyper.get("log_path", "runs")))

    mrtrainer = MRTrainer.init_from_dict(mrmodel,
                                        current_train,
                                        current_test,
                                        training_listener,
                                        hyper,
                                        optim_handler=optim_handler)
    mrtrainer.train(hyper['device'])


if __name__ == '__main__':
    nfeatures = [32]
    h_layers = [1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(777)
    for features in nfeatures:
        for layers in h_layers:
            #-- hyperparameters in configs --#
            hyper = load_hyperparameters('experiments/ch3/configs/fitting.yml')
            hyper['device'] = device
            hyper['hidden_features'] = features
            hyper['hidden_layers'] = layers

            training_pipeline(hyper)