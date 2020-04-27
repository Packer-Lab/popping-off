import os
import json

def loadpaths():
    # Where is this file?
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    with open(os.path.join(__location__, 'data_paths.json'), 'r') as config_file:
        config_info = json.load(config_file)
        user_paths_dict = config_info['paths']

    # Expand tildes in the json paths
    return {k:os.path.expanduser(v) for k, v in user_paths_dict.items()}
