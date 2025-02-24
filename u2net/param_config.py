from typing import Any, Dict, Text
import yaml

def parse_from_yaml(yaml_file_path: Text) -> Dict[Any, Any]:
    with tf.io.gfile.GFile(yaml_file_path, 'r') as f:
        config_dict = yaml_load(f, Loader=yaml.FullLoader)
        
    return config_dict

def get_config() -> Dict[Any, Any]:
    config = parse_from_yaml('config.yaml')
    return config