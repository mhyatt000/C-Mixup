import numpy as np
from sklearn.neighbors import KernelDensity
import torch
from utils import stats_values


def get_mixup_sample_rate(args, data_packet, device="cuda", use_kde=False):
    mix_idx = []
    _, y_list = data_packet["x_train"], data_packet["y_train"]
    is_np = isinstance(y_list, np.ndarray)
    if is_np:
        data_list = torch.tensor(y_list, dtype=torch.float32)
    else:
        data_list = y_list

    N = len(data_list)

    ######## use kde rate or uniform rate #######
    for i in range(N):
        if args.mixtype == "kde" or use_kde:  # kde
            data_i = data_list[i]
            data_i = data_i.reshape(-1, data_i.shape[0])  # get 2D

            if args.show_process:
                if i % (N // 10) == 0:
                    print("Mixup sample prepare {:.2f}%".format(i * 100.0 / N))
                # if i == 0: print(f'data_list.shape = {data_list.shape}, std(data_list) = {torch.std(data_list)}')#, data_i = {data_i}' + f'data_i.shape = {data_i.shape}')

            ######### get kde sample rate ##########
            kd = KernelDensity(kernel=args.kde_type, bandwidth=args.kde_bandwidth).fit(data_i)  # should be 2D
            each_rate = np.exp(kd.score_samples(data_list))
            each_rate /= np.sum(each_rate)  # norm
        else:
            each_rate = np.ones(y_list.shape[0]) * 1.0 / y_list.shape[0]

        ####### visualization: observe relative rate distribution shot #######
        if args.show_process and i == 0:
            print(f"bw = {args.kde_bandwidth}")
            print(f"each_rate[:10] = {each_rate[:10]}")
            stats_values(each_rate)

        mix_idx.append(each_rate)

    mix_idx = np.array(mix_idx)

    self_rate = [mix_idx[i][i] for i in range(len(mix_idx))]

    if args.show_process:
        print(
            f"len(y_list) = {len(y_list)}, len(mix_idx) = {len(mix_idx)}, np.mean(self_rate) = {np.mean(self_rate)}, np.std(self_rate) = {np.std(self_rate)},  np.min(self_rate) = {np.min(self_rate)}, np.max(self_rate) = {np.max(self_rate)}"
        )

    return mix_idx


def get_batch_kde_mixup_idx(args, batch_X, batch_Y, device):
    assert batch_X.shape[0] % 2 == 0
    batch_packet = {}
    batch_packet["x_train"] = batch_X.cpu()
    batch_packet["y_train"] = batch_Y.cpu()

    batch_rate = get_mixup_sample_rate(args, batch_packet, device, use_kde=True)  # batch -> kde
    if args.show_process:
        stats_values(batch_rate[0])
        # print(f'batch_rate[0][:20] = {batch_rate[0][:20]}')
    idx2 = [
        np.random.choice(np.arange(batch_X.shape[0]), p=batch_rate[sel_idx])
        for sel_idx in np.arange(batch_X.shape[0] // 2)
    ]
    return idx2


def get_batch_kde_mixup_batch(args, batch_X1, batch_X2, batch_Y1, batch_Y2, device):
    batch_X = torch.cat([batch_X1, batch_X2], dim=0)
    batch_Y = torch.cat([batch_Y1, batch_Y2], dim=0)

    idx2 = get_batch_kde_mixup_idx(args, batch_X, batch_Y, device)

    new_batch_X2 = batch_X[idx2]
    new_batch_Y2 = batch_Y[idx2]
    return new_batch_X2, new_batch_Y2
