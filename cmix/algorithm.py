import copy
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from cmix.mix import get_batch_kde_mixup_batch, get_batch_kde_mixup_idx, get_mixup_sample_rate  # noqa: F401


def cal_worst_acc(args, data_packet, best_model_rmse, best_result_dict_rmse, all_begin, ts_data, device):
    #### worst group acc ---> rmse ####
    if args.is_ood:
        x_test_assay_list = data_packet["x_test_assay_list"]
        y_test_assay_list = data_packet["y_test_assay_list"]
        worst_acc = 0.0 if args.metrics == "rmse" else 1e10

        for i in range(len(x_test_assay_list)):
            result_dic = test(
                args, best_model_rmse, x_test_assay_list[i], y_test_assay_list[i], "", False, all_begin, device
            )
            acc = result_dic[args.metrics]
            if args.metrics == "rmse":
                if acc > worst_acc:
                    worst_acc = acc
            else:  # r
                if np.abs(acc) < np.abs(worst_acc):
                    worst_acc = acc
        print("worst {} = {:.3f}".format(args.metrics, worst_acc))
        best_result_dict_rmse["worst_" + args.metrics] = worst_acc


def test(args, model, x_list, y_list, name, need_verbose, epoch_start_time, device):
    model.eval()
    with torch.no_grad():
        if args.dataset == "Dti_dg":
            val_iter = x_list.shape[0] // args.batch_size
            val_len = args.batch_size
            y_list = y_list[: val_iter * val_len]
        else:  # read in the whole test data
            val_iter = 1
            val_len = x_list.shape[0]
        y_list_pred = []
        assert val_iter >= 1  #  easy test

        for ith in range(val_iter):
            if isinstance(x_list, np.ndarray):
                x_list_torch = torch.tensor(x_list[ith * val_len : (ith + 1) * val_len], dtype=torch.float32).to(device)
            else:
                x_list_torch = x_list[ith * val_len : (ith + 1) * val_len].to(device)

            model = model.to(device)
            pred_y = model(x_list_torch).cpu().numpy()
            y_list_pred.append(pred_y)

        y_list_pred = np.concatenate(y_list_pred, axis=0)
        y_list = y_list.squeeze()
        y_list_pred = y_list_pred.squeeze()

        if not isinstance(y_list, np.ndarray):
            y_list = y_list.numpy()

        ###### calculate metrics ######

        mean_p = y_list_pred.mean(axis=0)
        sigma_p = y_list_pred.std(axis=0)
        mean_g = y_list.mean(axis=0)
        sigma_g = y_list.std(axis=0)

        index = sigma_g != 0
        corr = ((y_list_pred - mean_p) * (y_list - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
        corr = (corr[index]).mean()

        mse = (np.square(y_list_pred - y_list)).mean()
        result_dict = {"mse": mse, "r": corr, "r^2": corr**2, "rmse": np.sqrt(mse)}

        not_zero_idx = y_list != 0.0
        mape = (np.fabs(y_list_pred[not_zero_idx] - y_list[not_zero_idx]) / np.fabs(y_list[not_zero_idx])).mean() * 100
        result_dict["mape"] = mape

    ### verbose ###
    if need_verbose:
        epoch_use_time = time.time() - epoch_start_time
        # valid -> interval time; final test -> all time
        print(
            name
            + "corr = {:.4f}, rmse = {:.4f}, mape = {:.4f} %".format(corr, np.sqrt(mse), mape)
            + ", time = {:.4f} s".format(epoch_use_time)
        )

    return result_dict


def train(args, model, data_packet, is_mixup=True, mixup_idx_sample_rate=None, ts_data=None, device="cuda"):
    ######### model prepare ########
    model.train(True)
    optimizer = Adam(model.parameters(), **args.optimiser_args)
    loss_fun = nn.MSELoss(reduction="mean").to(device)

    best_mse = 1e10  # for best update
    best_r2 = 0.0
    repr_flag = 1  # for batch kde visualize training process

    scheduler = None

    x_train = data_packet["x_train"]
    y_train = data_packet["y_train"]
    x_valid = data_packet["x_valid"]
    y_valid = data_packet["y_valid"]

    iteration = len(x_train) // args.batch_size
    steps_per_epoch = iteration

    result_dict, best_mse_model = {}, None
    step_print_num = 30  # for dti

    need_shuffle = not args.is_ood

    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        model.train()
        shuffle_idx = np.random.permutation(np.arange(len(x_train)))

        if need_shuffle:  # id
            x_train_input = x_train[shuffle_idx]
            y_train_input = y_train[shuffle_idx]
        else:  # ood
            x_train_input = x_train
            y_train_input = y_train

        if not is_mixup:
            # iteration for each batch
            for idx in range(iteration):
                # select batch
                x_input_tmp = x_train_input[idx * args.batch_size : (idx + 1) * args.batch_size]
                y_input_tmp = y_train_input[idx * args.batch_size : (idx + 1) * args.batch_size]

                # -> tensor
                x_input = torch.tensor(x_input_tmp, dtype=torch.float32).to(device)
                y_input = torch.tensor(y_input_tmp, dtype=torch.float32).to(device)

                # forward
                pred_Y = model(x_input)
                loss = loss_fun(pred_Y, y_input)

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:  # backward (without scheduler)
                    scheduler.step()

                # validation
                if args.dataset == "Dti_dg" and (idx - 1) % (iteration // step_print_num) == 0:
                    result_dict = test(
                        args,
                        model,
                        x_valid,
                        y_valid,
                        "Train epoch " + str(epoch) + ", step = {} ".format((epoch * steps_per_epoch + idx)) + ":\t",
                        args.show_process,
                        epoch_start_time,
                        device,
                    )

                    # save best model
                    if result_dict["mse"] <= best_mse:
                        best_mse = result_dict["mse"]
                        best_mse_model = copy.deepcopy(model)
                    if result_dict["r"] ** 2 >= best_r2:
                        best_r2 = result_dict["r"] ** 2
                        best_r2_model = copy.deepcopy(model)

        else:  # mix up
            for idx in range(iteration):
                lambd = np.random.beta(args.mix_alpha, args.mix_alpha)

                if need_shuffle:  # get batch idx
                    idx_1 = shuffle_idx[idx * args.batch_size : (idx + 1) * args.batch_size]
                else:
                    idx_1 = np.arange(len(x_train))[idx * args.batch_size : (idx + 1) * args.batch_size]

                if args.mixtype == "kde":
                    idx_2 = np.array(
                        [
                            np.random.choice(np.arange(x_train.shape[0]), p=mixup_idx_sample_rate[sel_idx])
                            for sel_idx in idx_1
                        ]
                    )
                else:  # random mix
                    idx_2 = np.array([np.random.choice(np.arange(x_train.shape[0])) for sel_idx in idx_1])

                if isinstance(x_train, np.ndarray):
                    X1 = torch.tensor(x_train[idx_1], dtype=torch.float32).to(device)
                    Y1 = torch.tensor(y_train[idx_1], dtype=torch.float32).to(device)

                    X2 = torch.tensor(x_train[idx_2], dtype=torch.float32).to(device)
                    Y2 = torch.tensor(y_train[idx_2], dtype=torch.float32).to(device)
                else:
                    X1 = x_train[idx_1].to(device)
                    Y1 = y_train[idx_1].to(device)

                    X2 = x_train[idx_2].to(device)
                    Y2 = y_train[idx_2].to(device)

                if args.batch_type == 1:  # sample from batch
                    assert args.mixtype == "random"
                    if not repr_flag:  # show the sample status once
                        args.show_process = 0
                    else:
                        repr_flag = 0
                    X2, Y2 = get_batch_kde_mixup_batch(args, X1, X2, Y1, Y2, device)
                    args.show_process = 1

                X1 = X1.to(device)
                X2 = X2.to(device)
                Y1 = Y1.to(device)
                Y2 = Y2.to(device)

                # mixup
                mixup_Y = Y1 * lambd + Y2 * (1 - lambd)
                mixup_X = X1 * lambd + X2 * (1 - lambd)

                # forward
                if args.use_manifold:
                    pred_Y = model.forward_mixup(X1, X2, lambd)
                else:
                    pred_Y = model.forward(mixup_X)

                if args.dataset == "TimeSeires":  # time series loss need scale
                    scale = ts_data.scale.expand(pred_Y.size(0), ts_data.m)
                    loss = loss_fun(pred_Y * scale, mixup_Y * scale)
                else:
                    loss = loss_fun(pred_Y, mixup_Y)

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (
                    args.dataset == "Dti_dg" and (idx - 1) % (iteration // step_print_num) == 0
                ):  # dti has small epoch number, so verbose multiple times at 1 iteration
                    result_dict = test(
                        args,
                        model,
                        x_valid,
                        y_valid,
                        "Train epoch " + str(epoch) + ",  step = {} ".format((epoch * steps_per_epoch + idx)) + ":\t",
                        args.show_process,
                        epoch_start_time,
                        device,
                    )
                    # save best model
                    if result_dict["mse"] <= best_mse:
                        best_mse = result_dict["mse"]
                        best_mse_model = copy.deepcopy(model)
                    if result_dict["r"] ** 2 >= best_r2:
                        best_r2 = result_dict["r"] ** 2
                        best_r2_model = copy.deepcopy(model)

        # validation
        result_dict = test(
            args,
            model,
            x_valid,
            y_valid,
            "Train epoch " + str(epoch) + ":\t",
            args.show_process,
            epoch_start_time,
            device,
        )

        # if args.is_ood:
        #     cal_worst_acc(args,data_packet,model,result_dict,epoch_start_time,ts_data,device)
        #     worst_test_loss_log.append(result_dict['worst_rmse']**2)

        if result_dict["mse"] <= best_mse:
            best_mse = result_dict["mse"]
            best_mse_model = copy.deepcopy(model)
            print(f"update best mse! epoch = {epoch}")

        if result_dict["r"] ** 2 >= best_r2:
            best_r2 = result_dict["r"] ** 2
            best_r2_model = copy.deepcopy(model)

    return best_mse_model, best_r2_model
