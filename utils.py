import os
from typing import Union

import numpy as np
import torch
import yaml


def dict_to_yaml_file(data: dict, file_path: Union[str, np.byte]) -> int:
    """
    Speichert ein Dictionary als YAML-Datei auf der Festplatte.

    :param data: Das Dictionary, das gespeichert werden soll.
    :param file_path: Der Pfad zur YAML-Datei, in die das Dictionary gespeichert werden soll.
    """
    try:
        with open(file_path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)
        print(f'Dictionary erfolgreich in YAML-Datei gespeichert: {file_path}')
        return 0
    except Exception as e:
        print(f'Fehler beim Speichern des Dictionary in YAML-Datei: {e}')
        return 1


def read_yaml_file(file_path: str) -> Union[None, dict]:
    with open(file_path, 'r') as file:
        try:
            yaml_data = yaml.safe_load(file)
            return yaml_data
        except yaml.YAMLError as e:
            print(f"Error reading YAML file {file_path}: {e}")
            return None

def create_directories(paths:list) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
