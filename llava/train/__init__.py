import os
import importlib
import sys

from transformers import Trainer
from llava.train.llava_trainer import LLaVATrainer
# current_directory = os.getcwd()
current_directory = os.path.dirname(__file__)
sys.path.append(current_directory)


def get_all_trainer_file_prefixes():
    return [trainer_file_name.split('.')[0] for trainer_file_name in os.listdir(current_directory)
            if not trainer_file_name.find('__') > -1 and 'Trainer' in trainer_file_name]


names = {}
all_trainer_file_prefixes = get_all_trainer_file_prefixes()
for file_prefixes in all_trainer_file_prefixes:
    mod = importlib.import_module(file_prefixes)

    for v in mod.__dict__.values():
        if isinstance(v, type):
            if issubclass(v, LLaVATrainer):
                trainer_name = v.__name__
                names[trainer_name] = v


def get_trainer(name,
                model,
                tokenizer,
                training_args,
                **data_module):
    return names[name](model=model,
                       tokenizer=tokenizer,
                       args=training_args,
                       **data_module)