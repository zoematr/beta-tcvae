import math
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

import lib.utils as utils
# from metric_helpers.loader import load_model_and_dataset
from metric_helpers.mi_metric import compute_metric_shapes, compute_metric_faces


def estimate_entropies(qz_samples, qz_params, q_dist, n_samples=10000, weights=None):
    """Computes:
        E_{p(x)} E_{q(z|x)} [-log q(z)] and E_{p(x)} E_{q(z_j|x)} [-log q(z_j)]
    """
    device = qz_params.device

    # sample subset on the correct device
    S = qz_samples.size(1)
    take = min(n_samples, S)
    if weights is None:
        idx = torch.randperm(S, device=qz_samples.device)[:take]
        qz_samples = qz_samples.index_select(1, idx)
    else:
        weights = weights.to(qz_samples.device)
        sample_inds = torch.multinomial(weights, take, replacement=True)
        qz_samples = qz_samples.index_select(1, sample_inds)

    K, S = qz_samples.size()
    N, _, nparams = qz_params.size()
    assert nparams == q_dist.nparams
    assert K == qz_params.size(1)

    if weights is None:
        weights = -math.log(N)
    else:
        weights = torch.log(weights.view(N, 1, 1) / weights.sum())

    entropies = torch.zeros(K, device=device)

    pbar = tqdm(total=S)
    k = 0
    while k < S:
        batch_size = min(10, S - k)
        logqz_i = q_dist.log_density(
            qz_samples.view(1, K, S).expand(N, K, S)[:, :, k:k + batch_size],
            qz_params.view(N, K, 1, nparams).expand(N, K, S, nparams)[:, :, k:k + batch_size]
        )
        k += batch_size

        entropies += - utils.logsumexp(logqz_i + weights, dim=0, keepdim=False).sum(1)
        pbar.update(batch_size)
    pbar.close()

    entropies /= S
    return entropies


def mutual_info_metric_shapes(vae, shapes_dataset):
    device = next(vae.parameters()).device
    dataset_loader = DataLoader(shapes_dataset, batch_size=1000, num_workers=0, shuffle=False)

    N = len(dataset_loader.dataset)
    K = vae.z_dim
    nparams = vae.q_dist.nparams
    vae.eval()

    print('Computing q(z|x) distributions.')
    qz_params = torch.empty(N, K, nparams, device=device)

    n = 0
    with torch.no_grad():
        for xs in dataset_loader:
            batch_size = xs.size(0)
            xs = xs.view(batch_size, 1, 64, 64).to(device)
            qz_params[n:n + batch_size] = vae.encoder(xs).view(batch_size, K, nparams)
            n += batch_size

    qz_params = qz_params.view(3, 6, 40, 32, 32, K, nparams)
    qz_samples = vae.q_dist.sample(params=qz_params)

    print('Estimating marginal entropies.')
    marginal_entropies = estimate_entropies(
        qz_samples.view(N, K).transpose(0, 1),
        qz_params.view(N, K, nparams),
        vae.q_dist
    ).cpu()

    cond_entropies = torch.zeros(4, K)
    print('Estimating conditional entropies for scale.')
    for i in range(6):
        cond_entropies_i = estimate_entropies(
            qz_samples[:, i, :, :, :, :].contiguous().view(N // 6, K).transpose(0, 1),
            qz_params[:, i, :, :, :, :].contiguous().view(N // 6, K, nparams),
            vae.q_dist
        )
        cond_entropies[0] += cond_entropies_i.cpu() / 6

    print('Estimating conditional entropies for orientation.')
    for i in range(40):
        cond_entropies_i = estimate_entropies(
            qz_samples[:, :, i, :, :, :].contiguous().view(N // 40, K).transpose(0, 1),
            qz_params[:, :, i, :, :, :].contiguous().view(N // 40, K, nparams),
            vae.q_dist
        )
        cond_entropies[1] += cond_entropies_i.cpu() / 40

    print('Estimating conditional entropies for pos x.')
    for i in range(32):
        cond_entropies_i = estimate_entropies(
            qz_samples[:, :, :, i, :, :].contiguous().view(N // 32, K).transpose(0, 1),
            qz_params[:, :, :, i, :, :].contiguous().view(N // 32, K, nparams),
            vae.q_dist
        )
        cond_entropies[2] += cond_entropies_i.cpu() / 32

    print('Estimating conditional entropies for pos y.')
    for i in range(32):
        cond_entropies_i = estimate_entropies(
            qz_samples[:, :, :, :, i, :].contiguous().view(N // 32, K).transpose(0, 1),
            qz_params[:, :, :, :, i, :].contiguous().view(N // 32, K, nparams),
            vae.q_dist
        )
        cond_entropies[3] += cond_entropies_i.cpu() / 32

    metric = compute_metric_shapes(marginal_entropies, cond_entropies)
    return metric, marginal_entropies, cond_entropies


def mutual_info_metric_faces(vae, shapes_dataset):
    device = next(vae.parameters()).device
    dataset_loader = DataLoader(shapes_dataset, batch_size=1000, num_workers=0, shuffle=False)

    N = len(dataset_loader.dataset)
    K = vae.z_dim
    nparams = vae.q_dist.nparams
    vae.eval()

    print('Computing q(z|x) distributions.')
    qz_params = torch.empty(N, K, nparams, device=device)

    n = 0
    with torch.no_grad():
        for xs in dataset_loader:
            batch_size = xs.size(0)
            xs = xs.view(batch_size, 1, 64, 64).to(device)
            qz_params[n:n + batch_size] = vae.encoder(xs).view(batch_size, K, nparams)
            n += batch_size

    qz_params = qz_params.view(50, 21, 11, 11, K, nparams)
    qz_samples = vae.q_dist.sample(params=qz_params)

    print('Estimating marginal entropies.')
    marginal_entropies = estimate_entropies(
        qz_samples.view(N, K).transpose(0, 1),
        qz_params.view(N, K, nparams),
        vae.q_dist
    ).cpu()
    cond_entropies = torch.zeros(3, K)

    print('Estimating conditional entropies for azimuth.')
    for i in range(21):
        cond_entropies_i = estimate_entropies(
            qz_samples[:, i, :, :, :].contiguous().view(N // 21, K).transpose(0, 1),
            qz_params[:, i, :, :, :].contiguous().view(N // 21, K, nparams),
            vae.q_dist
        )
        cond_entropies[0] += cond_entropies_i.cpu() / 21

    print('Estimating conditional entropies for elevation.')
    for i in range(11):
        cond_entropies_i = estimate_entropies(
            qz_samples[:, :, i, :, :].contiguous().view(N // 11, K).transpose(0, 1),
            qz_params[:, :, i, :, :].contiguous().view(N // 11, K, nparams),
            vae.q_dist
        )
        cond_entropies[1] += cond_entropies_i.cpu() / 11

    print('Estimating conditional entropies for lighting.')
    for i in range(11):
        cond_entropies_i = estimate_entropies(
            qz_samples[:, :, :, i, :].contiguous().view(N // 11, K).transpose(0, 1),
            qz_params[:, :, :, i, :].contiguous().view(N // 11, K, nparams),
            vae.q_dist
        )
        cond_entropies[2] += cond_entropies_i.cpu() / 11

    metric = compute_metric_faces(marginal_entropies, cond_entropies)
    return metric, marginal_entropies, cond_entropies


"""
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpt', required=True)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--save', type=str, default='.')
    args = parser.parse_args()

    # Optional and safe on non-CUDA systems
    if args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    vae, dataset, cpargs = load_model_and_dataset(args.checkpt)
    metric, marginal_entropies, cond_entropies = eval('mutual_info_metric_' + cpargs.dataset)(vae, dataset)
    torch.save({
        'metric': metric,
        'marginal_entropies': marginal_entropies,
        'cond_entropies': cond_entropies,
    }, os.path.join(args.save, 'disentanglement_metric.pth'))
    print('MIG: {:.2f}'.format(metric))
"""
