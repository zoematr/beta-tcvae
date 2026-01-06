import sys, subprocess, platform
import math
import os, sys, subprocess, datetime
print('Python executable:', sys.executable)
print('Python version:', platform.python_version())
print('Verifying torch in current kernel...')
subprocess.run([sys.executable, '-m', 'pip', 'show', 'torch'])

# Common arguments
DATASET = 'shapes'
BETA = 6
TCVAE = True
LOG_FREQ = 50
BATCH_SIZES = [32, 64, 256, 1024, 2048]
SEEDS = [0, 1, 2, 3, 4]

# Budget: fix optimizer steps ('steps') or total examples processed ('examples')
BUDGET = 'steps'
DATASET_SIZE = 737280
TARGET_STEPS = 100000

# W&B
USE_WANDB = True
WANDB_PROJECT = 'beta-tcvae'
WANDB_ENTITY = None
WANDB_MODE = 'online'

def epochs_for_bs(n, bs):
    steps_per_epoch = math.ceil(n / bs)
    return math.ceil(TARGET_STEPS / steps_per_epoch)

EPOCHS_PER_BS = {bs: epochs_for_bs(DATASET_SIZE, bs) for bs in BATCH_SIZES}
print('EPOCHS_PER_BS:', EPOCHS_PER_BS)


# Sweep over batch sizes and seeds; save logs

os.makedirs('runs', exist_ok=True)


for seed in SEEDS:
    for bs in BATCH_SIZES:
        cmd = [sys.executable, 'vae_quant.py',
               '--dataset', DATASET,
               '--beta', str(BETA),
               '--batch-size', str(bs),
               '--num-epochs', str(epochs_for_bs(DATASET_SIZE, bs)),
               '--log_freq', str(LOG_FREQ),
               '--seed', str(seed)]
        if TCVAE:
            cmd.append('--tcvae')
        if USE_WANDB:
            cmd += ['--wandb', '--wandb_project', WANDB_PROJECT, '--wandb_mode', WANDB_MODE]
            if WANDB_ENTITY:
                cmd += ['--wandb_entity', WANDB_ENTITY]
            cmd += ['--wandb_run_name', f'bs{bs}_seed{seed}']
        print(f'\n=== Running batch-size {bs} | seed {seed} ===\n', ' '.join(cmd))
        log_path = f'runs/bs{bs}_seed{seed}.log'
        with open(log_path, 'w') as f:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                print(line, end='')
                f.write(line)
            proc.wait()
        print(f'[exit code {proc.returncode}] log saved to {log_path}')