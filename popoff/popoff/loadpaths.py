import os
import json
import sys
from pathlib import Path

def loadpaths():
    # Where is this file?
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    json_path = os.path.join(Path(__location__).parent.parent, 'data_paths.json')
    json_path = str(json_path)

    with open(json_path, 'r') as config_file:
        config_info = json.load(config_file)
        user_paths_dict = config_info['paths']

    # Expand tildes in the json paths
    user_paths_dict = {k:str(v) for k, v in user_paths_dict.items()}
    return {k:os.path.expanduser(v) for k, v in user_paths_dict.items()}
