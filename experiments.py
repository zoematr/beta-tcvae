import sys, subprocess, platform
import math
import os, sys, subprocess
print('Python executable:', sys.executable)
print('Python version:', platform.python_version())
print('Verifying torch in current kernel...')
subprocess.run([sys.executable, '-m', 'pip', 'show', 'torch'])

# Common arguments
DATASET = 'shapes'
BETA = 6
TCVAE = True
LOG_FREQ = 50

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


# W&B
USE_WANDB = True
WANDB_PROJECT = 'beta-tcvae'
WANDB_ENTITY = None
WANDB_MODE = 'online'



os.makedirs('runs', exist_ok=True)


for seed in SEEDS:

    cmd = [sys.executable, 'vae_quant.py',
            '--dataset', DATASET,
            '--beta', str(BETA),
            '--log_freq', str(LOG_FREQ),
            '--seed', str(seed)]
    if TCVAE:
        cmd.append('--tcvae')
    if USE_WANDB:
        cmd += ['--wandb', '--wandb_project', WANDB_PROJECT, '--wandb_mode', WANDB_MODE]
        if WANDB_ENTITY:
            cmd += ['--wandb_entity', WANDB_ENTITY]
        cmd += ['--wandb_run_name', f'tcvae_seed{seed}']
    log_path = f'runs/tcvae_seed{seed}.log'
    with open(log_path, 'w') as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end='')
            f.write(line)
        proc.wait()
    print(f'[exit code {proc.returncode}] log saved to {log_path}')