import os
from pathlib import Path
import torch

from mrnet.training.trainer import MRTrainer
from mrnet.datasets.signals import Signal1D
from mrnet.networks.mrnet import MRFactory
from mrnet.datasets.pyramids import create_MR_structure
from mrnet.training.listener import TrainingListener
from mrnet.datasets.procedural import perlin_noise

from mrnet.training.utils import load_hyperparameters, get_optim_handler
from IPython import embed


def exp_noise512():
    torch.manual_seed(777)
    nsamples = 512
    #-- hyperparameters in configs --#
    hyper = load_hyperparameters('experiments/ch3/siren1d.yml')
    hyper['nsamples'] = nsamples
    project_name = hyper.get('project_name', 'phd-thesis')
    
    noise = perlin_noise(nsamples, 8, 4, 1.4)
    base_signal = Signal1D(noise.view(1, -1), 
                           domain=hyper['domain'],
                           batch_size=hyper['batch_size'])

    train_dataset = create_MR_structure(base_signal,
                                        hyper['max_stages'],
                                        hyper['filter'],
                                        hyper['decimation'],
                                        hyper['pmode'])
    test_dataset = create_MR_structure(base_signal,
                                        hyper['max_stages'],
                                        hyper['filter'],
                                        False,
                                        hyper['pmode'])

    # you can substitute this line by your custom handler class
    optim_handler = get_optim_handler(hyper.get('optim_handler', 'regular'))
    
    mrmodel = MRFactory.from_dict(hyper)
    print("Model: ", type(mrmodel))

    name = os.path.basename(hyper['data_path'])
    training_listener = TrainingListener(project_name,
                                f"{name[0:7]}{hyper['model']}{hyper['filter'][0].upper()}",
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
    exp_noise512()