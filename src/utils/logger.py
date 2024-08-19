import logging
import os
import sys

import yaml


def setup_logger(name, save_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Print log to console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Save log to file
    fh = logging.FileHandler(os.path.join(save_dir, f"logs_{name}.txt"), mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def dump_yaml_to_log(yaml_file_path, logger):
    try:
        with open(yaml_file_path, "r") as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
            yaml_string = yaml.dump(yaml_data, default_flow_style=False, sort_keys=False)
            logger.info("Trainings configuration:\n%s", yaml_string)
    except Exception as e:
        logger.error("An error occurred: %s", str(e))
