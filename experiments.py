import sys, subprocess, platform, os
print('Python executable:', sys.executable)
print('Python version:', platform.python_version())
print('Verifying torch in current kernel...')
subprocess.run([sys.executable, '-m', 'pip', 'show', 'torch'])

# Common arguments
DATASET = 'shapes'
BETA = 6
TCVAE = True
NUM_EPOCHS = 1
LOG_FREQ = 50
MWS_BATCH_SIZES = [32, 64, 256, 512, 1024]
SEEDS = [0, 1, 2]

# W&B
USE_WANDB = True
WANDB_PROJECT = 'beta-tcvae'
WANDB_ENTITY = None
WANDB_MODE = 'online'

env = os.environ.copy()
env['PYTHONUNBUFFERED'] = '1'
os.makedirs('runs', exist_ok=True)

for seed in SEEDS:
    print(f'\n########## Running experiments for seed {seed} ##########\n')
    for bs in MWS_BATCH_SIZES:
        cmd = [sys.executable, 'vae_quant.py',
               '--dataset', DATASET,
               '--beta', str(BETA),
               '--tcvae',
               '--mws-batch-size', str(bs),
               '--num-epochs', str(NUM_EPOCHS),
               '--log_freq', str(LOG_FREQ),
               '--seed', str(seed)]
        if TCVAE:
            cmd.append('--tcvae')
        if USE_WANDB:
            cmd += ['--wandb', '--wandb_project', WANDB_PROJECT, '--wandb_mode', WANDB_MODE,
                    '--wandb_run_name', f'mws-bs{bs}_seed{seed}']
        print(f'\n=== Running mws batch-size {bs} | seed {seed} ===\n', ' '.join(cmd))
        log_path = f'runs/bs{bs}_seed{seed}.log'
        with open(log_path, 'w') as f:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                print(line, end=''); f.write(line)
            proc.wait()
        print(f'[exit code {proc.returncode}] log saved to {log_path}')