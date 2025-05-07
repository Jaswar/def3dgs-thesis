import os
import random
import time
import argparse


TEMPLATE_CONFIG = '''
ModelParams = dict(
    deform_depth = <deform_depth>,
    deform_width = <deform_width>
)
OptimizationParams = dict(
    iterations = <iterations>
)
'''

SAMPLING_SETTINGS = {
    'iterations': [15_000, 25_000, 40_000],
    'deform_depth': [6, 8, 10],
    'deform_width': [128, 256, 512]
}


def sample_config():
    config = {}
    for key, values in SAMPLING_SETTINGS.items():
        config[key] = random.choice(values)
    return config


def write_config(config, path):
    new_config = TEMPLATE_CONFIG
    for key, value in config.items():
        new_config = new_config.replace(f'<{key}>', str(value))
    with open(path, 'w') as f:
        f.write(new_config)


def execute_in_env(command, env):
    return os.system(f'/bin/bash -c \"source /opt/miniconda3/etc/profile.d/conda.sh && conda activate {env} && {command} \"')


def run_experiment(data_path, model_path, config_path):
    ret_val = execute_in_env(f'python train.py -s \"{data_path}\" ' \
                   f'-m \"{model_path}\" ' \
                   f'--configs \"{config_path}\" --set_seed', 'def3dgs')
    if ret_val != 0:
        raise ValueError('Execution failed')


def main(data_path, model_path, timeout=5 * 60 * 60):
    configs_path = './arguments/ego_exo/random_configs'
    os.makedirs(configs_path, exist_ok=True)
    start_time = time.time()
    config_index = 0
    while time.time() - start_time < timeout:
        config = sample_config()
        config_path = os.path.join(configs_path, f'config_{config_index}.py')
        model_path_ = os.path.join(model_path, f'model_{config_index}')
        write_config(config, config_path)
        run_experiment(data_path, model_path_, config_path)
        config_index += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--model_path', type=str)
    args = parser.parse_args()
    data_path = args.data_path
    model_path = args.model_path
    main(data_path, model_path)