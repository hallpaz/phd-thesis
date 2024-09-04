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

def crafted_signal(nsamples):
    X = torch.linspace(-1, 1, nsamples)
    return (torch.sin(2 * 2 * torch.pi * X) 
                + torch.sin(5 * 2 * torch.pi * X)
                + torch.sin(31 * 2 * torch.pi * X)
                + torch.sin(42 * 2 * torch.pi * X)) / 3
     

def exp_crafted_frequencies():
    for hidden_layers in [0, 1, 2]:
        torch.manual_seed(777)
        #-- hyperparameters in configs --#
        hyper = load_hyperparameters('experiments/ch4/configs/shallow.yml')
        nsamples = hyper['nsamples']
        project_name = hyper['project_name']
        hyper['hidden_layers'] = hidden_layers
        hyper['omega_0'] = hyper['omega_0'] * 2 * torch.pi
        
        frequencies = crafted_signal(nsamples)
        base_signal = Signal1D(frequencies.view(1, -1), 
                            domain=hyper['domain'],
                            batch_size=hyper['batch_size'])

        train_dataset = [base_signal]
        test_dataset = [base_signal]

        # you can substitute this line by your custom handler class
        optim_handler = get_optim_handler(hyper.get('optim_handler', 'regular'))
        
        mrmodel = MRFactory.from_dict(hyper)
        print("Model: ", type(mrmodel))
        print(mrmodel)

        name = os.path.basename(hyper['data_path'])
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

def exp_smoothed():
    torch.manual_seed(777)
    #-- hyperparameters in configs --#
    hyper = load_hyperparameters('experiments/ch4/configs/shallow.yml')
    nsamples = hyper['nsamples']
    project_name = hyper['project_name']
    
    frequencies = crafted_signal(nsamples)
    base_signal = Signal1D(frequencies.view(1, -1), 
                        domain=hyper['domain'],
                        batch_size=hyper['batch_size'])

    train_dataset = [base_signal]
    test_dataset = [base_signal]

    # you can substitute this line by your custom handler class
    optim_handler = get_optim_handler(hyper.get('optim_handler', 'regular'))
    
    mrmodel = MRFactory.from_dict(hyper)
    print("Model: ", type(mrmodel))
    print(mrmodel)

    name = "smoothed"
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


def exp_high_frequencies_only():
    for hidden_layers in [0, 1, 2]:
        for low_tone in [25 * 2 * torch.pi, 35 * 2 * torch.pi]:
            torch.manual_seed(777)
            #-- hyperparameters in configs --#
            hyper = load_hyperparameters('experiments/ch4/configs/shallow.yml')
            nsamples = hyper['nsamples']
            project_name = hyper['project_name']
            hyper['hidden_layers'] = hidden_layers
            
            high_tone = 45 * 2 * torch.pi
            hyper['omega_0'] = high_tone
            
            frequencies = crafted_signal(nsamples)
            base_signal = Signal1D(frequencies.view(1, -1), 
                                domain=hyper['domain'],
                                batch_size=hyper['batch_size'])

            train_dataset = [base_signal]
            test_dataset = [base_signal]

            # you can substitute this line by your custom handler class
            optim_handler = get_optim_handler(hyper.get('optim_handler', 'regular'))
            
            mrmodel = MRFactory.from_dict(hyper)
            print("Model: ", type(mrmodel))
            with torch.no_grad():
                layer_width = len(mrmodel.stages[0].first_layer.linear.weight)
                print(layer_width)
                mrmodel.stages[0].first_layer.linear.weight[:layer_width//2].uniform_(-1, -low_tone/high_tone)
                mrmodel.stages[0].first_layer.linear.weight[layer_width//2:].uniform_(low_tone/high_tone, 1)
            print(mrmodel)

            name = os.path.basename(hyper['data_path'])
            training_listener = TrainingListener(project_name,
                                        f"{name}{hyper['model']}l{low_tone:.1f}h{high_tone:.1f}",
                                        hyper,
                                        Path(hyper.get("log_path", "runs")))

            mrtrainer = MRTrainer.init_from_dict(mrmodel,
                                                train_dataset,
                                                test_dataset,
                                                training_listener,
                                                hyper,
                                                optim_handler=optim_handler)
            mrtrainer.train(hyper['device'])

def exp_all_frequencies():
    torch.manual_seed(777)
    #-- hyperparameters in configs --#
    hyper = load_hyperparameters('experiments/ch4/configs/shallow.yml')
    nsamples = hyper['nsamples']
    project_name = hyper['project_name']
    
    hyper['omega_0'] = 45 * 2 * torch.pi
    hyper['hidden_features'] = 128
    
    frequencies = crafted_signal(nsamples)
    base_signal = Signal1D(frequencies.view(1, -1), 
                        domain=hyper['domain'],
                        batch_size=hyper['batch_size'])

    train_dataset = [base_signal]
    test_dataset = [base_signal]

    # you can substitute this line by your custom handler class
    optim_handler = get_optim_handler(hyper.get('optim_handler', 'regular'))
    
    mrmodel = MRFactory.from_dict(hyper)
    print("Model: ", type(mrmodel))
    print(mrmodel)

    name = os.path.basename(hyper['data_path'])
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
    exp_smoothed()
    # exp_crafted_frequencies()
    # exp_high_frequencies_only()
    # exp_all_frequencies()    