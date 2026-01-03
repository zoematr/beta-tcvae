import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import brewer2mpl
import logging
bmap = brewer2mpl.get_map('Set1', 'qualitative', 3)
colors = bmap.mpl_colors

plt.style.use('ggplot')

VAR_THRESHOLD = 1e-2


def plot_vs_gt_shapes(vae, dataset, out_path, z_inds=None):
    import torch
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    device = next(vae.parameters()).device
    loader = DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=0)

    nparams = vae.q_dist.nparams
    K = vae.z_dim
    qz_params = torch.empty(len(dataset), K, nparams)

    n = 0
    vae.eval()
    with torch.no_grad():
        for xs in loader:
            bs = xs.size(0)
            xs = xs.view(bs, 1, 64, 64).to(device)
            enc = vae.encoder(xs).view(bs, K, nparams).detach().cpu()
            qz_params[n:n + bs] = enc
            n += bs

    # Shapes dataset grid: (shape=3, scale=6, rotation=40, pos_x=32, pos_y=32)
    qz_params = qz_params.view(3, 6, 40, 32, 32, K, nparams)

    # z_j is inactive if Var_x(E[z_j|x]) < eps.
    qz_means = qz_params[..., 0]
    var = torch.std(qz_means.contiguous().view(len(dataset), K), dim=0).pow(2)
    active_units = torch.arange(0, K)[var > VAR_THRESHOLD].long()
    print('Active units: ' + ','.join(map(str, active_units.tolist())))
    print('Number of active units: {}/{}'.format(len(active_units), vae.z_dim))

    if z_inds is None:
        z_inds = active_units

    # GT factors vs latent means
    mean_scale = qz_means.mean(2).mean(2).mean(2)      # (shape, scale, latent)
    mean_rotation = qz_means.mean(1).mean(2).mean(2)   # (shape, rotation, latent)
    mean_pos = qz_means.mean(0).mean(0).mean(0)        # (pos_x, pos_y, latent)

    fig = plt.figure(figsize=(3, len(z_inds)))
    gs = gridspec.GridSpec(len(z_inds), 3)
    gs.update(wspace=0, hspace=0)

    vmin_pos = torch.min(mean_pos)
    vmax_pos = torch.max(mean_pos)
    for i, j in enumerate(z_inds):
        ax = fig.add_subplot(gs[i * 3])
        ax.imshow(mean_pos[:, :, j].numpy(), cmap=plt.get_cmap('coolwarm'), vmin=vmin_pos, vmax=vmax_pos)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(r'$z_' + str(j.item() if torch.is_tensor(j) else j) + r'$')
        if i == len(z_inds) - 1:
            ax.set_xlabel(r'pos')

    vmin_scale = torch.min(mean_scale)
    vmax_scale = torch.max(mean_scale)
    for i, j in enumerate(z_inds):
        ax = fig.add_subplot(gs[1 + i * 3])
        ax.plot(mean_scale[0, :, j].numpy(), color=colors[2])
        ax.plot(mean_scale[1, :, j].numpy(), color=colors[0])
        ax.plot(mean_scale[2, :, j].numpy(), color=colors[1])
        ax.set_ylim([vmin_scale, vmax_scale]); ax.set_xticks([]); ax.set_yticks([])
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
        if i == len(z_inds) - 1:
            ax.set_xlabel(r'scale')

    vmin_rotation = torch.min(mean_rotation)
    vmax_rotation = torch.max(mean_rotation)
    for i, j in enumerate(z_inds):
        ax = fig.add_subplot(gs[2 + i * 3])
        ax.plot(mean_rotation[0, :, j].numpy(), color=colors[2])
        ax.plot(mean_rotation[1, :, j].numpy(), color=colors[0])
        ax.plot(mean_rotation[2, :, j].numpy(), color=colors[1])
        ax.set_ylim([vmin_rotation, vmax_rotation]); ax.set_xticks([]); ax.set_yticks([])
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
        if i == len(z_inds) - 1:
            ax.set_xlabel(r'rotation')

    fig.text(0.5, 0.03, 'Ground Truth', ha='center')
    fig.text(0.01, 0.5, 'Learned Latent Variables ', va='center', rotation='vertical')
    plt.savefig(out_path)
    plt.close()


def plot_vs_gt_faces(vae, dataset, out_path, z_inds=None):
    import torch
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    device = next(vae.parameters()).device
    loader = DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=0)

    nparams = vae.q_dist.nparams
    K = vae.z_dim
    qz_params = torch.empty(len(dataset), K, nparams)

    n = 0
    vae.eval()
    with torch.no_grad():
        for xs in loader:
            bs = xs.size(0)
            xs = xs.view(bs, 1, 64, 64).to(device)
            enc = vae.encoder(xs).view(bs, K, nparams).detach().cpu()
            qz_params[n:n + bs] = enc
            n += bs

    # Faces grid: (pose_az=50, pose_el=21, light_az=11, light_el=11)
    qz_params = qz_params.view(50, 21, 11, 11, K, nparams)

    qz_means = qz_params[..., 0]
    var = torch.std(qz_means.contiguous().view(len(dataset), K), dim=0).pow(2)
    active_units = torch.arange(0, K)[var > VAR_THRESHOLD].long()
    print('Active units: ' + ','.join(map(str, active_units.tolist())))
    print('Number of active units: {}/{}'.format(len(active_units), vae.z_dim))

    if z_inds is None:
        z_inds = active_units

    mean_pose_az = qz_means.mean(3).mean(2).mean(0)  # (pose_az, latent)
    mean_pose_el = qz_means.mean(3).mean(1).mean(0)  # (pose_el, latent)
    mean_light_az = qz_means.mean(2).mean(1).mean(0)  # (light_az, latent)

    fig = plt.figure(figsize=(len(z_inds), 3))
    gs = gridspec.GridSpec(3, len(z_inds))
    gs.update(wspace=0, hspace=0)

    vmin_scale = torch.min(mean_pose_az); vmax_scale = torch.max(mean_pose_az)
    for i, j in enumerate(z_inds):
        ax = fig.add_subplot(gs[i])
        ax.plot(mean_pose_az[:, j].numpy())
        ax.set_ylim([vmin_scale, vmax_scale]); ax.set_xticks([]); ax.set_yticks([])
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
        if i == 0:
            ax.set_ylabel(r'azimuth')

    vmin_scale = torch.min(mean_pose_el); vmax_scale = torch.max(mean_pose_el)
    for i, j in enumerate(z_inds):
        ax = fig.add_subplot(gs[len(z_inds) + i])
        ax.plot(mean_pose_el[:, j].numpy())
        ax.set_ylim([vmin_scale, vmax_scale]); ax.set_xticks([]); ax.set_yticks([])
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
        if i == 0:
            ax.set_ylabel(r'elevation')

    vmin_scale = torch.min(mean_light_az); vmax_scale = torch.max(mean_light_az)
    for i, j in enumerate(z_inds):
        ax = fig.add_subplot(gs[2 * len(z_inds) + i])
        ax.plot(mean_light_az[:, j].numpy())
        ax.set_ylim([vmin_scale, vmax_scale]); ax.set_xticks([]); ax.set_yticks([])
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
        if i == 0:
            ax.set_ylabel(r'lighting')

    plt.suptitle('GT Factors vs. Latent Variables')
    plt.savefig(out_path)
    plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-checkpt', required=True)
    parser.add_argument('-zs', type=str, default=None)
    parser.add_argument('-gpu', type=int, default=0)
    parser.add_argument('-save', type=str, default='latent_vs_gt.pdf')
    parser.add_argument('-elbo_decomp', action='store_true')
    args = parser.parse_args()

    from elbo_decomposition import elbo_decomposition
    import lib.dist as dist
    import lib.flows as flows
    from vae_quant import VAE, setup_data_loaders

    def load_model_and_dataset(checkpt_filename):
        print('Loading model and dataset.')
        checkpt = torch.load(checkpt_filename, map_location=lambda storage, loc: storage)
        args = checkpt['args']
        state_dict = checkpt['state_dict']

        # model
        if not hasattr(args, 'dist') or args.dist == 'normal':
            prior_dist = dist.Normal()
            q_dist = dist.Normal()
        elif args.dist == 'laplace':
            prior_dist = dist.Laplace()
            q_dist = dist.Laplace()
        elif args.dist == 'flow':
            prior_dist = flows.FactorialNormalizingFlow(dim=args.latent_dim, nsteps=4)
            q_dist = dist.Normal()
        vae = VAE(z_dim=args.latent_dim, use_cuda=True, prior_dist=prior_dist, q_dist=q_dist, conv=args.conv)
        vae.load_state_dict(state_dict, strict=False)

        # dataset loader
        loader = setup_data_loaders(args)
        return vae, loader, args

    z_inds = list(map(int, args.zs.split(','))) if args.zs is not None else None
    
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

    vae, dataset_loader, cpargs = load_model_and_dataset(args.checkpt)
    if args.elbo_decomp:
        elbo_decomposition(vae, dataset_loader)
    eval('plot_vs_gt_' + cpargs.dataset)(vae, dataset_loader.dataset, args.save, z_inds)
