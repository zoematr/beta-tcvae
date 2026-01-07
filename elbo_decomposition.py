import os
import math
import torch
from numbers import Number
from tqdm import tqdm

import lib.dist as dist
import lib.flows as flows

EPS = 1e-12


def estimate_entropies(qz_samples, qz_params, q_dist):
    """
    qz_samples: (K, S) tensor on same device as qz_params
    qz_params:  (N, K, nparams) tensor
    """
    device = qz_params.device
    # Only take a subset of samples (up to 10k)
    S = qz_samples.size(1)
    take = min(10000, S)
    idx = torch.randperm(S, device=qz_samples.device)[:take]
    qz_samples = qz_samples.index_select(1, idx)

    K, S = qz_samples.size()
    N, _, nparams = qz_params.size()
    assert nparams == q_dist.nparams
    assert K == qz_params.size(1)

    marginal_entropies = torch.zeros(K, device=device)
    joint_entropy = torch.zeros(1, device=device)

    pbar = tqdm(total=S)
    k = 0
    while k < S:
        batch_size = min(10, S - k)
        logqz_i = q_dist.log_density(
            qz_samples.view(1, K, S).expand(N, K, S)[:, :, k:k + batch_size],
            qz_params.view(N, K, 1, nparams).expand(N, K, S, nparams)[:, :, k:k + batch_size])
        k += batch_size

        # computes - log q(z_i) summed over minibatch
        marginal_entropies += (math.log(N) - logsumexp(logqz_i, dim=0, keepdim=False)).sum(1).detach()
        # computes - log q(z) summed over minibatch
        logqz = logqz_i.sum(1)  # (N, S)
        joint_entropy += (math.log(N) - logsumexp(logqz, dim=0, keepdim=False)).sum(0).detach()
        pbar.update(batch_size)
    pbar.close()

    marginal_entropies /= S
    joint_entropy /= S

    return marginal_entropies, joint_entropy


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


def analytical_NLL(qz_params, q_dist, prior_dist, qz_samples=None):
    """
    Returns:
        nlogqz_condx (K,) tensor
        nlogpz (K,) tensor
    """
    pz_params = torch.zeros_like(qz_params)
    nlogqz_condx = q_dist.NLL(qz_params).mean(0)
    nlogpz = prior_dist.NLL(pz_params, qz_params).mean(0)
    return nlogqz_condx, nlogpz


def elbo_decomposition(vae, dataset_loader, mws_batch_size=None, device=None):
    """
    Computes logpx, TC (dependence), MI, dimwise KL, etc.
    Uses mws_batch_size columns for the MWS estimator if provided,
    otherwise uses the current minibatch size.
    """
    vae.eval()
    if device is None:
        device = next(vae.parameters()).device

    logpx_all = []
    tc_all = []
    mi_all = []
    dkl_all = []

    dataset_size = len(dataset_loader.dataset)

    with torch.no_grad():
        for xs in dataset_loader:
            x = xs.to(device)
            B = x.size(0)                           # rows from this minibatch
            Bcols = min(mws_batch_size or B, B)     # cols for MWS (<= B)
            # encode/forward
            x_recon, x_params, z, q_params = vae.reconstruct_img(x)
            # per-sample pieces
            logpx = vae.x_dist.log_density(x, params=x_params).view(B, -1).sum(1)
            logqz_condx = vae.q_dist.log_density(z, params=q_params).view(B, -1).sum(1)
            prior_params = vae._get_prior_params(B)
            logpz = vae.prior_dist.log_density(z, params=prior_params).view(B, -1).sum(1)

            # MWS: rows B, cols Bcols
            z_rows = z[:B]
            q_cols = q_params[:Bcols]

            # shape (B, Bcols, z_dim)
            logqz_ij = vae.q_dist.log_density(
                z_rows.view(B, 1, vae.z_dim),
                q_cols.view(1, Bcols, vae.z_dim, vae.q_dist.nparams),
            )

            # joint and product-of-marginals normalizers
            norm = math.log(max(Bcols * dataset_size, 1))  # safe
            # product of marginals
            logqz_prodm = (torch.logsumexp(logqz_ij, dim=1) - norm).sum(1)  # (B,)
            # joint
            logqz_joint = torch.logsumexp(logqz_ij.sum(2), dim=1) - norm    # (B,)

            # MI/TC/DWKL terms
            tc = (logqz_joint - logqz_prodm)          # (B,)
            mi = (logqz_condx - logqz_joint)          # (B,)
            dkl = (logqz_prodm - logpz)               # (B,)

            # collect
            logpx_all.append(logpx)
            tc_all.append(tc)
            mi_all.append(mi)
            dkl_all.append(dkl)

    # stack and return scalars/tensors
    logpx_all = torch.cat(logpx_all)
    tc_all = torch.cat(tc_all)
    mi_all = torch.cat(mi_all)
    dkl_all = torch.cat(dkl_all)

    return (logpx_all.mean().item(),
            tc_all.mean().item(),
            mi_all.mean().item(),
            dkl_all.mean().item(),
            # keep compatibility: return some extras as None if your old API did
            None, None, None)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-checkpt', required=True)
    parser.add_argument('-save', type=str, default='.')
    parser.add_argument('-gpu', type=int, default=0)
    args = parser.parse_args()

    def load_model_and_dataset(checkpt_filename):
        checkpt = torch.load(checkpt_filename)
        args = checkpt['args']
        state_dict = checkpt['state_dict']

        # backwards compatibility
        if not hasattr(args, 'conv'):
            args.conv = False

        from vae_quant import VAE, setup_data_loaders

        # model
        if args.dist == 'normal':
            prior_dist = dist.Normal()
            q_dist = dist.Normal()
        elif args.dist == 'laplace':
            prior_dist = dist.Laplace()
            q_dist = dist.Laplace()
        elif args.dist == 'flow':
            prior_dist = flows.FactorialNormalizingFlow(dim=args.latent_dim, nsteps=32)
            q_dist = dist.Normal()
        vae = VAE(z_dim=args.latent_dim, use_cuda=True, prior_dist=prior_dist, q_dist=q_dist, conv=args.conv)
        vae.load_state_dict(state_dict, strict=False)
        vae.eval()

        # dataset loader
        loader = setup_data_loaders(args, use_cuda=True)
        return vae, loader

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
        pin_memory = True
        use_cuda_flag = True
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        pin_memory = False
        use_cuda_flag = False
    else:
        device = torch.device('cpu')
        pin_memory = False
        use_cuda_flag = False
        
    vae, dataset_loader = load_model_and_dataset(args.checkpt)
    logpx, dependence, information, dimwise_kl, analytical_cond_kl, marginal_entropies, joint_entropy = \
        elbo_decomposition(vae, dataset_loader)
    torch.save({
        'logpx': logpx,
        'dependence': dependence,
        'information': information,
        'dimwise_kl': dimwise_kl,
        'analytical_cond_kl': analytical_cond_kl,
        'marginal_entropies': marginal_entropies,
        'joint_entropy': joint_entropy
    }, os.path.join(args.save, 'elbo_decomposition.pth'))
