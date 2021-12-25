import time
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
import math
import doasys

if __name__ == '__main__':

    torch.nn.Module.dump_patches = True

    is_fig = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--grid_size', type=int, default=10000, help='the size of grids')

    # array parameters
    parser.add_argument('--ant_num', type=int, default=16, help='the number of antennas')
    parser.add_argument('--max_target_num', type=int, default=3, help='the maximum number of targets')
    parser.add_argument('--d', type=float, default=0.5, help='the distance between antennas')

    # imperfect parameters 0.15 0.5 0.2
    parser.add_argument('--max_per_std', type=float, default=0.15, help='the maximum std of the position perturbation')
    parser.add_argument('--max_amp_std', type=float, default=0.5, help='the maximum std of the amplitude')
    parser.add_argument('--max_phase_std', type=float, default=0.2, help='the maximum std of the phase')
    parser.add_argument('--max_mc', type=float, default=0.06, help='the maximum mutual coupling (0.1->-10dB)')
    parser.add_argument('--nonlinear', type=float, default=1.0, help='the nonlinear parameter')
    parser.add_argument('--is_nonlinear', type=int, default=1, help='nonlinear effect')

    parser.add_argument('--is_music', type=int, default=0, help='compared with MUSIC method')
    parser.add_argument('--is_anm', type=int, default=0, help='compared with ANM method')
    parser.add_argument('--data_len', type=int, default=1000, help='compared with ANM method')

    parser.add_argument('--net_file', type=str, default='net.pkl', help='the file of trained network')

    args = parser.parse_args()

    is_music = args.is_music
    is_anm = args.is_anm

    if torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.use_cuda = False

    doa_grid = np.linspace(-50, 50, args.grid_size, endpoint=False)

    if args.use_cuda:
        net = torch.load(args.net_file)
    else:
        net = torch.load(args.net_file, map_location=torch.device('cpu'))

    if args.use_cuda:
        net.cuda()

    dic_mat = np.zeros((doa_grid.size, 2, args.ant_num))
    dic_mat_comp = np.zeros((doa_grid.size, args.ant_num), dtype=complex)
    for n in range(doa_grid.size):
        tmp = doasys.steer_vec(doa_grid[n], args.d, args.ant_num, np.zeros(args.ant_num).T)
        tmp = tmp / np.sqrt(np.sum(np.power(np.abs(tmp), 2)))
        dic_mat[n, 0] = tmp.real
        dic_mat[n, 1] = tmp.imag
        dic_mat_comp[n] = tmp
    dic_mat_torch = torch.from_numpy(dic_mat).float()
    if args.use_cuda:
        dic_mat_torch = dic_mat_torch.cuda()

    # generate the validation data
    SNR_range = np.linspace(0, 30, 7)
    RMSE = np.zeros((SNR_range.size, 1))
    RMSE_FFT = np.zeros((SNR_range.size, 1))
    RMSE_MUSIC = np.zeros((SNR_range.size, 1))
    RMSE_OMP = np.zeros((SNR_range.size, 1))
    RMSE_ANM = np.zeros((SNR_range.size, 1))

    if is_music or is_anm:
        import matlab.engine
        eng = matlab.engine.start_matlab()

    data_len = args.data_len
    if data_len < 200:
        n_test = 1
        test_len = data_len
    else:
        test_len = 200
        n_test = int(data_len / test_len)

    for n in range(SNR_range.size):
        RMSE[n] = 0
        RMSE_FFT[n] = 0
        RMSE_MUSIC[n] = 0
        RMSE_OMP[n] = 0
        RMSE_ANM[n] = 0
        for n1 in range(n_test):
            epoch_start_time = time.time()
            signal, doa, target_num = doasys.gen_signal(test_len, args)
            signal = torch.from_numpy(signal).float()
            SNR_dB = SNR_range[n]
            noisy_signals = doasys.noise_torch(signal, math.pow(10.0, SNR_dB / 10.0))

            # proposed method
            if args.use_cuda:
                noisy_signals = noisy_signals.cuda()
            with torch.no_grad():
                output_net = net(noisy_signals).view(test_len, 2, -1)

            mm_real = torch.mm(output_net[:, 0, :], dic_mat_torch[:, 0, :].T) + torch.mm(output_net[:, 1, :],
                                                                                         dic_mat_torch[:, 1, :].T)
            mm_imag = torch.mm(output_net[:, 0, :], dic_mat_torch[:, 1, :].T) - torch.mm(output_net[:, 1, :],
                                                                                         dic_mat_torch[:, 0, :].T)
            sp = torch.pow(mm_real, 2) + torch.pow(mm_imag, 2)
            sp_np = sp.cpu().detach().numpy()
            for idx_sp in range(sp_np.shape[0]):
                sp_np[idx_sp] = sp_np[idx_sp] / np.max(sp_np[idx_sp])

            doa_num = (doa >= -90).sum(axis=1)
            est_doa = doasys.get_doa(sp_np, doa_num, doa_grid, args.max_target_num, doa)
            RMSE[n] = RMSE[n] + np.sum(np.power(np.abs(est_doa - doa), 2))

            # FFT method
            if args.use_cuda:
                r = noisy_signals.cpu().detach().numpy()
            else:
                r = noisy_signals.detach().numpy()
            r_c = r[:, 0, :] + 1j * r[:, 1, :]
            sp_FFT = np.power(np.abs(np.matmul(dic_mat_comp, np.conj(r_c).T)), 2).T
            for idx_sp in range(sp_FFT.shape[0]):
                sp_FFT[idx_sp] = sp_FFT[idx_sp] / np.max(sp_FFT[idx_sp])
            doa_num = (doa >= -90).sum(axis=1)
            est_doa = doasys.get_doa(sp_FFT, doa_num, doa_grid, args.max_target_num, doa)
            RMSE_FFT[n] = RMSE_FFT[n] + np.sum(np.power(np.abs(est_doa - doa), 2))

            # MUSIC alg
            if is_music:
                music_num = min(n_test, 1000)
                if args.use_cuda:
                    r = noisy_signals.cpu().detach().numpy()
                else:
                    r = noisy_signals.detach().numpy()

                r_c = r[0:music_num, 0, :] + 1j * r[0:music_num, 1, :]
                sp_MUSIC = np.zeros((r_c.shape[0], args.grid_size))
                for idx_r in range(r_c.shape[0]):
                    x_tmp = eng.MUSIConesnapshot(matlab.double(list(r_c[idx_r]), is_complex=True),
                                                 int(target_num[idx_r]),
                                                 matlab.double(list(doa_grid), is_complex=False))
                    sp_MUSIC[idx_r] = np.squeeze(np.asarray(x_tmp))

                doa_num = (doa >= -90).sum(axis=1)
                est_doa = doasys.get_doa(sp_MUSIC, doa_num[0:music_num], doa_grid, args.max_target_num, doa)
                RMSE_MUSIC[n] = RMSE_MUSIC[n] + np.sum(np.power(np.abs(est_doa - doa[0:music_num]), 2))

            # OMP alg
            if args.use_cuda:
                r = noisy_signals.cpu().detach().numpy()
            else:
                r = noisy_signals.detach().numpy()
            r_c = r[:, 0, :] + 1j * r[:, 1, :]
            est_doa_omp = -100 * np.ones((r_c.shape[0], args.max_target_num))
            for idx1 in range(r_c.shape[0]):
                r_tmp0 = np.expand_dims(r_c[idx1], axis=0)
                r_tmp1 = r_tmp0
                max_idx = np.zeros(target_num[idx1], dtype=int)
                for idx2 in range(target_num[idx1]):
                    max_idx_tmp = np.argmax(np.abs(np.matmul(dic_mat_comp, np.conj(r_tmp1).T)))
                    max_idx[idx2] = max_idx_tmp
                    dic_tmp = dic_mat_comp[max_idx[0:idx2 + 1]]
                    r_tmp1 = r_tmp0 - np.matmul(np.matmul(r_tmp0, np.linalg.pinv(dic_tmp)), dic_tmp)
                    est_doa_omp[idx1, idx2] = doa_grid[max_idx_tmp]
                est_doa_omp[idx1] = np.sort(est_doa_omp[idx1])
            RMSE_OMP[n] = RMSE_OMP[n] + np.sum(np.power(np.abs(est_doa_omp - doa), 2))

            # atomic norm minimization alg
            if is_anm:
                anm_num = min(n_test, 1000)
                if args.use_cuda:
                    r = noisy_signals.cpu().detach().numpy()
                else:
                    r = noisy_signals.detach().numpy()
                r_c = r[0:anm_num, 0, :] + 1j * r[0:anm_num, 1, :]
                x = np.zeros((r_c.shape[0], args.ant_num), dtype=complex)
                for idx_r in range(r_c.shape[0]):
                    x_tmp = eng.ANM(matlab.double(list(r_c[idx_r]), is_complex=True))
                    x[idx_r] = np.squeeze(np.asarray(x_tmp))

                sp_ANM = np.power(np.abs(np.matmul(dic_mat_comp, np.conj(x).T)), 2).T

                for idx_sp in range(sp_ANM.shape[0]):
                    sp_ANM[idx_sp] = sp_ANM[idx_sp] / np.max(sp_ANM[idx_sp])
                doa_num = (doa >= -90).sum(axis=1)
                est_doa = doasys.get_doa(sp_ANM, doa_num[0:anm_num], doa_grid, args.max_target_num, doa)
                RMSE_ANM[n] = RMSE_ANM[n] + np.sum(np.power(np.abs(est_doa - doa[0:anm_num]), 2))

            if n == 0 and is_fig:
                plt.figure()
                plt.plot(doa_grid, sp_np[0], label='Proposed method')
                plt.plot(doa_grid, sp_FFT[0], label='FFT method')
                if is_anm:
                    plt.plot(doa_grid, sp_ANM[0], label='ANM method')
                if is_music:
                    plt.plot(doa_grid, sp_MUSIC[0], label='MUSIC method')
                tmp_doa = est_doa_omp[0][np.argwhere(est_doa_omp[0] > -90)]
                plt.stem(tmp_doa, np.ones((tmp_doa.size, 1)), markerfmt='x', use_line_collection=True,
                         label='OMP method')
                tmp_doa = doa[0][np.argwhere(doa[0] > -90)]
                plt.stem(tmp_doa, np.ones((tmp_doa.size, 1)), use_line_collection=True, label='Ground-truth DOA')
                plt.xlabel('Spatial angle (deg)')
                plt.ylabel('Spatial spectrum')
                plt.legend()
                plt.grid()
                plt.show()

            print("SNR: %.2f dB, Test: %d/%d, Time: %.2f" % (SNR_dB, n1, n_test, time.time() - epoch_start_time))
        RMSE[n] = np.sqrt(RMSE[n] / (doa.size * n_test))
        RMSE_FFT[n] = np.sqrt(RMSE_FFT[n] / (doa.size * n_test))
        if is_music:
            RMSE_MUSIC[n] = np.sqrt(RMSE_MUSIC[n] / (music_num * args.max_target_num * n_test))
        RMSE_OMP[n] = np.sqrt(RMSE_OMP[n] / (doa.size * n_test))
        if is_anm:
            RMSE_ANM[n] = np.sqrt(RMSE_ANM[n] / (anm_num * args.max_target_num * n_test))
        print(
            "SNR (dB): %.2f dB, RMSE (deg): %.2f, RMSE_FFT (deg): %.2f, RMSE_OMP (deg): %.2f, RMSE_ANM (deg): %.2f, RMSE_MUSIC (deg): %.2f" % (
                SNR_dB, RMSE[n], RMSE_FFT[n], RMSE_OMP[n], RMSE_ANM[n], RMSE_MUSIC[n]))

    plt.figure()
    plt.semilogy(SNR_range, RMSE, linestyle='-', marker='o', linewidth=2, markersize=8, label='Proposed method')
    plt.semilogy(SNR_range, RMSE_FFT, linestyle='-', marker='v', linewidth=2, markersize=8, label='FFT method')
    if is_music:
        plt.semilogy(SNR_range, RMSE_MUSIC, linestyle='-', marker='x', linewidth=2, markersize=8, label='MUSIC method')
    plt.semilogy(SNR_range, RMSE_OMP, linestyle='-', marker='+', linewidth=2, markersize=8, label='OMP method')
    if is_anm:
        plt.semilogy(SNR_range, RMSE_ANM, linestyle='-', marker='s', linewidth=2, markersize=8, label='ANM method')
    plt.xlabel('SNR (dB)')
    plt.ylabel('RMSE (deg)')
    plt.legend()
    plt.grid()
    plt.show()
