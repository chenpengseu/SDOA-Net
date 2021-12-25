import time
import numpy as np
import torch
import torch.nn as nn
import scipy.signal
import math
import matplotlib.pyplot as plt


def gen_doa(target_num, doa, ant_num):
    doa_min = -45
    doa_max = 45
    min_sep = 102.0 / (ant_num - 1) * 2.0
    for n in range(target_num):
        condition = True
        doa_new = 0
        while condition:
            doa_new = np.random.rand() * (doa_max - doa_min) + doa_min
            condition = (np.min(np.abs(doa - doa_new)) < min_sep)
        doa[n] = doa_new


def steer_vec(doa_deg, d, ant_num, d_per):
    st = np.exp(1j * 2 * np.pi * (d * np.arange(ant_num).T + d_per) * np.sin(np.deg2rad(doa_deg)))
    return st


def gen_signal(data_num, args):
    target_num = np.random.randint(1, args.max_target_num + 1, data_num)
    doa = np.ones((data_num, args.max_target_num)) * np.inf
    s = np.zeros((data_num, 2, args.ant_num))
    for n in range(data_num):
        gen_doa(target_num[n], doa[n], args.ant_num)
        # perturbation
        d_per = np.random.randn(args.ant_num).T * np.random.rand(1) * args.max_per_std
        # phase and amplitude
        amp = np.ones(args.ant_num).T + np.random.randn(args.ant_num).T * np.random.rand(1) * args.max_amp_std
        pha = np.random.randn(args.ant_num).T * np.random.rand(1) * args.max_phase_std
        amp_phase = amp * np.exp(1j * pha)
        for m in range(target_num[n]):
            st = amp_phase * steer_vec(doa[n, m], args.d, args.ant_num, d_per)
            s[n, 0] = s[n, 0] + st.real
            s[n, 1] = s[n, 1] + st.imag
        s[n] = s[n] / np.sqrt(np.mean(np.power(s[n], 2)))
        s_comp = s[n, 0] + 1j * s[n, 1]
        # mutual coupling
        max_mc_power = np.power(args.max_mc, np.arange(args.ant_num))
        mc_mat = np.zeros((args.ant_num, args.ant_num), dtype=complex)
        for idx_ant1 in range(args.ant_num):
            for idx_ant2 in range(args.ant_num):
                if idx_ant1 == idx_ant2:
                    mc_mat[idx_ant1, idx_ant2] = 1
                else:
                    mc_power = np.random.rand(1) * max_mc_power[np.abs(idx_ant2 - idx_ant1)]
                    mc_mat[idx_ant1, idx_ant2] = np.sqrt(mc_power) * np.exp(1j * np.random.rand(1) * 2 * np.pi)
        s_comp_mc = np.matmul(mc_mat, s_comp)
        s[n, 0] = s_comp_mc.real
        s[n, 1] = s_comp_mc.imag
        s[n] = s[n] / np.sqrt(np.mean(np.power(s[n], 2)))

        # non linear function
        if args.is_nonlinear:
            s[n] = np.tanh(args.nonlinear * s[n])
            s[n] = s[n] / np.sqrt(np.mean(np.power(s[n], 2)))

    doa[doa == float('inf')] = -100
    doa = np.sort(doa, axis=1)
    return s.astype('float32'), doa.astype('float32'), target_num


def gen_refsp(doa, doa_grid, sigma):
    ref_sp = np.zeros((doa.shape[0], doa_grid.shape[0]))
    for i in range(doa.shape[1]):
        dist = np.abs(doa_grid[None, :] - doa[:, i][:, None])
        ref_sp += np.exp(- dist ** 2 / sigma ** 2)
    return ref_sp


def noise_torch(s, snr):
    bsz, _, signal_dim = s.size()
    s = s.view(bsz, -1)
    sigma_max = np.sqrt(1. / snr)
    sigmas = sigma_max * torch.rand(bsz, device=s.device, dtype=s.dtype)
    noise = torch.randn(s.size(), device=s.device, dtype=s.dtype)
    mult = sigmas * torch.norm(s, 2, dim=1) / (torch.norm(noise, 2, dim=1))
    noise = noise * mult[:, None]
    return (s + noise).view(bsz, -1, signal_dim)


class spectrumModule(nn.Module):
    def __init__(self, signal_dim=8, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3):
        super().__init__()
        self.n_filters = n_filters
        self.in_layer = nn.Linear(2 * signal_dim, inner_dim * n_filters, bias=False)
        mod = []
        for n in range(n_layers):
            mod += [
                nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size - 1, bias=False,
                          padding_mode='circular'),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
            ]
        self.mod = nn.Sequential(*mod)
        self.out_layer = nn.Linear(inner_dim * n_filters, 2 * signal_dim, bias=False)

    def forward(self, inp):
        bsz = inp.size(0)
        inp = inp.view(bsz, -1)
        x = self.in_layer(inp).view(bsz, self.n_filters, -1)
        x = self.mod(x)
        x = x.view(bsz, -1)
        x = self.out_layer(x).view(bsz, -1)
        return x


class DeepFreq(nn.Module):
    def __init__(self, signal_dim=8, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, upsampling=8, kernel_out=25):
        super().__init__()
        self.fr_size = inner_dim * upsampling
        self.n_filters = n_filters
        self.in_layer = nn.Linear(2 * signal_dim, inner_dim * n_filters, bias=False)
        mod = []
        for n in range(n_layers):
            mod += [
                nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size - 1, bias=False,
                          padding_mode='circular'),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
            ]
        self.mod = nn.Sequential(*mod)
        self.out_layer = nn.ConvTranspose1d(n_filters, 1, kernel_out, stride=upsampling,
                                            padding=(kernel_out - upsampling + 1) // 2, output_padding=1, bias=False)
    def forward(self, inp):
        bsz = inp.size(0)
        inp = inp.view(bsz, -1)
        x = self.in_layer(inp).view(bsz, self.n_filters, -1)
        x = self.mod(x)
        x = self.out_layer(x).view(bsz, -1)
        return x


def get_doa(sp, doa_num, doa_grid, max_target_num, ref_doa):
    est_doa = -100 * np.ones((doa_num.shape[0], max_target_num))
    for n in range(len(doa_num)):
        find_peaks_out = scipy.signal.find_peaks(sp[n], height=(None, None))
        num_spikes = min(len(find_peaks_out[0]), int(doa_num[n]))
        idx = np.argpartition(find_peaks_out[1]['peak_heights'], -num_spikes)[-num_spikes:]
        est_doa[n, :num_spikes] = np.sort(doa_grid[find_peaks_out[0][idx]])
        tmp = est_doa[n].copy()
        for idx_tmp in range(len(ref_doa[n])):
            est_doa[n, idx_tmp] = tmp[np.argmin(np.abs(ref_doa[n, idx_tmp] - tmp))]
    return est_doa


def train_net(args, net, optimizer, criterion, train_loader, val_loader,
              doa_grid, epoch, train_num, train_type, net_type):
    epoch_start_time = time.time()
    net.train()
    loss_train = 0
    dic_mat = np.zeros((doa_grid.size, 2, args.ant_num))
    if net_type == 0:
        for n in range(doa_grid.size):
            tmp = steer_vec(doa_grid[n], args.d, args.ant_num, np.zeros(args.ant_num).T)
            dic_mat[n, 0] = tmp.real
            dic_mat[n, 1] = tmp.imag
        dic_mat_torch = torch.from_numpy(dic_mat).float()
        if args.use_cuda:
            dic_mat_torch = dic_mat_torch.cuda()

    for batch_idx, (clean_signal, target_sp, doa) in enumerate(train_loader):
        if args.use_cuda:
            clean_signal, target_sp = clean_signal.cuda(), target_sp.cuda()
        noisy_signal = noise_torch(clean_signal, args.snr)
        optimizer.zero_grad()
        output_net = net(noisy_signal).view(args.batch_size, 2, -1)
        if net_type == 0:
            mm_real = torch.mm(output_net[:, 0, :], dic_mat_torch[:, 0, :].T) + torch.mm(output_net[:, 1, :],
                                                                                         dic_mat_torch[:, 1, :].T)
            mm_imag = torch.mm(output_net[:, 0, :], dic_mat_torch[:, 1, :].T) - torch.mm(output_net[:, 1, :],
                                                                                         dic_mat_torch[:, 0, :].T)
        else:
            mm_real = output_net[:, 0, :]
            mm_imag = output_net[:, 1, :]
        sp = torch.pow(mm_real, 2) + torch.pow(mm_imag, 2)
        loss = criterion(sp, target_sp)

        loss.backward()
        optimizer.step()
        loss_train += loss.data.item()
    net.eval()
    loss_val, fnr_val = 0, 0
    for batch_idx, (noisy_signal, _, target_sp, doa) in enumerate(val_loader):
        if args.use_cuda:
            noisy_signal, target_sp = noisy_signal.cuda(), target_sp.cuda()
        with torch.no_grad():
            output_net = net(noisy_signal).view(args.batch_size, 2, -1)

        if net_type == 0:
            mm_real = torch.mm(output_net[:, 0, :], dic_mat_torch[:, 0, :].T) + torch.mm(output_net[:, 1, :],
                                                                                         dic_mat_torch[:, 1, :].T)
            mm_imag = torch.mm(output_net[:, 0, :], dic_mat_torch[:, 1, :].T) - torch.mm(output_net[:, 1, :],
                                                                                         dic_mat_torch[:, 0, :].T)
        else:
            mm_real = output_net[:, 0, :]
            mm_imag = output_net[:, 1, :]
        sp = torch.pow(mm_real, 2) + torch.pow(mm_imag, 2)
        loss = criterion(sp, target_sp)
        loss_val += loss.data.item()

        doa_num = (doa >= -90).sum(dim=1)
        est_doa = get_doa(sp.cpu().detach().numpy(), doa_num, doa_grid, args.max_target_num, doa)

    loss_train /= args.n_training
    loss_val /= args.n_validation

    print("Train_Num: %d, Train_Type: %d, Epochs: %d / %d, Time: %.1f, Training Loss: %.2f, Validation Loss:  %.2f" % (
        train_num,
        train_type,
        epoch, args.n_epochs,
        time.time() - epoch_start_time,
        loss_train,
        loss_val))
    return net
