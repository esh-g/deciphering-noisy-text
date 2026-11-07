import random 
from pathlib import Path 
import numpy as np 
import yaml 

config_path = "configs/generator.yaml"

def load_configs(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg 

cfg  = load_configs(config_path)

# print(cfg)

