import pickle
import random

import numpy as np
import torch


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stats_values(targets):
    ######### get status information ###########
    mean = np.mean(targets)
    min = np.min(targets)
    max = np.max(targets)
    std = np.std(targets)
    print(f"y stats: mean = {mean}, max = {max}, min = {min}, std = {std}")
    return mean, min, max, std


def get_unique_file_name(args, extra_str2="", profix=".txt"):
    ######### get file name ###########
    if args.dataset == "TimeSeries":
        extra_str = "_" + args.ts_name
    elif args.is_ood:
        extra_str = "_OOD"
    else:
        extra_str = ""

    if extra_str2 != "":
        extra_str += "_" + extra_str2

    if args.dataset == "Dti_dg" and args.sub_sample_batch_max_num != -1:
        extra_str += f"_sub{args.sub_sample_batch_max_num}"

    if args.mix_alpha != 2.0:
        extra_str += "_MixAlpha_" + str(args.mix_alpha)

    if args.batch_type != 0:
        extra_str += f"_BatchType{args.batch_type}"

    if args.seed != 0:
        extra_str += "_Seed" + str(args.seed)

    if args.dataset == "RotateFashionMNIST":
        if args.construct_color_data:
            extra_str += "_Color"
        if args.construct_no_color_data:
            extra_str += "_NoColor"

    if args.mixtype == "erm":
        fn = f"{args.dataset}{extra_str}_{args.mixtype}"
    else:
        fn = f"{args.dataset}{extra_str}_{args.mixtype}_{'UseManifold' if args.use_manifold else 'NotUseManifold'}"

    fn += profix
    return fn


def write_result(args, bw, data, result_path, extra_str=""):
    #### write result and model #####
    full_path = result_path + get_unique_file_name(args, extra_str, ".txt")
    if args.show_process:
        print(f"write result into path: {full_path}")
    with open(full_path, "a+") as f:  # >>
        # f.write(f'{args.seed}:{r}\n')
        if isinstance(data, list):
            for i in range(len(data)):
                f.write(f"{data[i]}\t")
            f.write("\n")
        elif isinstance(data, dict):  # write result dict
            f.write(f"bw = {bw}, seed = {args.seed}\n")
            for k in data.keys():
                f.write(f"{k}\t\t")
            f.write("\n")
            for k in data.keys():
                f.write("{:.7f}\t".format(data[k]))
            f.write("\n")
        else:
            f.write(f"{data}\n")


def write_model(args, model, result_path, extra_str=""):
    if model is not None:
        pt_full_path = result_path + get_unique_file_name(args, extra_str, ".pickle")
        if args.show_process:
            print(f"write model into path: {pt_full_path}")
        ##### store best model #####
        s = pickle.dumps(model)
        with open(pt_full_path, "wb+") as f:
            f.write(s)
