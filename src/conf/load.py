import os

import bunch
import munch
import yaml


def _load_path_from_yaml(file_name, root_path=os.getcwd()):
    """Loads a configuration file in YAML format and returns a Bunch object containing the paths specified in the file.

    Args:
        config_file (str): Path to the YAML configuration file.
        root_path (str, optional): Root path to use when constructing the paths. Defaults to the current working directory.

    Returns:
        bunch.Bunch: A Bunch object containing the paths specified in the configuration file.
    """
    
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    try:
        with open(config_path) as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        return None

    def recursive_dict(dict_obj, root_path):
        """Recursive function to convert a dictionary into a Bunch object and construct the paths.

        Args:
            dict_obj (dict): Dictionary to convert.
            root_path (str): Root path to use when constructing the paths.

        Returns:
            bunch.Bunch: A Bunch object containing the constructed paths.
        """
        result = bunch.Bunch()

        for k, v in dict_obj.items():
            result.setdefault(k, bunch.Bunch())
            
            if isinstance(v, dict):
                result[k] = recursive_dict(v, root_path)
            else:
                result[k] = os.path.join(root_path, v)
        
        return result

    return recursive_dict(config, root_path)

def _yaml_to_dict(file_name):
    """Carica un file YAML e lo converte in un dizionario Python.

    Args:
        file_path (str): Percorso del file YAML da caricare.

    Returns:
        dict: Dizionario Python contenente i dati del file YAML.

    """
    
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    try:
        with open(file_path) as file:
            yaml_dict = yaml.safe_load(file)
    except FileNotFoundError:
        return None

    def recursive_dict(dict_obj):
        """Funzione ricorsiva per convertire tutti i dizionari nidificati in oggetti dict.

        Args:
            dict_obj (dict): Dizionario da convertire.

        Returns:
            dict: Dizionario convertito.
        """
        for k, v in dict_obj.items():
            if isinstance(v, dict):
                dict_obj[k] = recursive_dict(v)
        return dict_obj

    return recursive_dict(yaml_dict)

paths = _load_path_from_yaml("path.yaml")

columns = munch.munchify(_yaml_to_dict("columns.yaml"))
variables = munch.munchify(_yaml_to_dict("variables.yaml"))

