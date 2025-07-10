# ContinuousMIXUP
from dataclasses import dataclass, field
import os
import pickle
import time
from typing import Optional

import torch
import tyro

from cmix import algorithm
from cmix.data_generate import load_data
from cmix.models import Learner, Learner_Dti_dg, Learner_RCF_MNIST, Learner_TimeSeries
from cmix.utils import get_unique_file_name, set_seed, write_model, write_result


@dataclass
class KDEConfig:
    use: bool = True
    bandwidth: float = 1.0
    model: str = "gaussian"  # gaussian or epanechnikov or linear


@dataclass
class MixUp:
    kde: KDEConfig = field(default_factory=KDEConfig)
    use_manifold: bool = True
    mode: str = "random"


@dataclass
class Config:
    result_root_path: str = "../../result/"
    dataset: str = "NO2"
    mixtype: str = "random"  # random or kde or erm
    use_manifold: bool = True
    seed: int = 0
    gpu: int = 0

    # kde parameter
    kde_bandwidth: float = 1.0
    kde_type: str = "gaussian"
    batch_type: int = 0  # 1 for y batch and 2 for x batch and 3 for representation

    # verbose
    show_process: bool = True
    show_setting: bool = True

    # model read & write
    read_best_model: bool = False
    store_model: bool = True

    # data path, for RCF_MNIST and TimeSeries
    data_dir: Optional[str] = None
    ts_name: str = ""


def load_model(cfg, ts_data):
    if cfg.dataset == "TimeSeries":
        model = Learner_TimeSeries(args=cfg, data=ts_data).to(device)
    elif cfg.dataset == "Dti_dg":
        model = Learner_Dti_dg(hparams=None).to(device)
    elif cfg.dataset == "RCF_MNIST":
        model = Learner_RCF_MNIST(args=cfg).to(device)
    else:
        model = Learner(args=cfg).to(device)

    if cfg.show_setting:
        nParams = sum([p.nelement() for p in model.parameters()])
        print("Number of parameters: %d" % nParams)
    return model


def set_device(cfg: Config) -> None:
    if torch.cuda.is_available() and cfg.gpu != -1:
        torch.cuda.set_device("cuda:" + str(cfg.gpu))
        device = torch.device("cuda:" + str(cfg.gpu))
        if cfg.show_setting:
            print(device)
    else:
        device = torch.device("cpu")
        if cfg.show_setting:
            print("use cpu")
    return device


def main(cfg: Config) -> None:
    """
    cfg.cuda = torch.cuda.is_available()
    cfg_dict = asdict(cfg)
    dict_name = cfg.dataset
    if cfg.dataset == "TimeSeries":
        dict_name += "-" + cfg.ts_name
    cfg_dict.update(dataset_defaults[dict_name])
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)
    """

    device = set_device(cfg)
    set_seed(cfg.seed)

    # prepare result directories
    result_root = cfg.result_root_path
    if not os.path.exists(result_root):
        os.mkdir(result_root)
    result_path = result_root + f"{cfg.dataset}/"
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    t1 = time.time()
    best_model_dict: dict[str, torch.nn.Module] = {}
    data_packet, ts_data = load_data(cfg)
    if cfg.show_setting:
        print("load dataset success, use time = {:.4f}".format(time.time() - t1))
        print(f"args.mixtype = {cfg.mixtype}, Use_manifold = {cfg.use_manifold}")

    set_seed(cfg.seed)

    if cfg.read_best_model == 0:
        model = load_model(cfg, ts_data)
        if cfg.show_setting:
            print("load untrained model done")
            print(cfg)

        all_begin = time.time()

        if cfg.mixtype == "kde":
            mixup_idx_sample_rate = algorithm.get_mixup_sample_rate(cfg, data_packet)
        else:
            mixup_idx_sample_rate = None

        sample_use_time = time.time() - all_begin
        print("sample use time = {:.4f}".format(sample_use_time))

        best_model_dict["rmse"], best_model_dict["r"] = algorithm.train(
            cfg,
            model,
            data_packet,
            cfg.mixtype != "erm",
            mixup_idx_sample_rate,
            ts_data,
            device,
        )

        print("=" * 30 + " single experiment result " + "=" * 30)
        result_dict_best = algorithm.test(
            cfg,
            best_model_dict[cfg.metrics],
            data_packet["x_test"],
            data_packet["y_test"],
            "seed = "
            + str(cfg.seed)
            + ": Final test for best "
            + cfg.metrics
            + " model: "
            + cfg.mixtype
            + ", use_manifold = "
            + str(cfg.use_manifold)
            + ", kde_bandwidth = "
            + str(cfg.kde_bandwidth)
            + ":\n",
            cfg.show_process,
            all_begin,
            device,
        )

        algorithm.cal_worst_acc(
            cfg,
            data_packet,
            best_model_dict[cfg.metrics],
            result_dict_best,
            all_begin,
            ts_data,
            device,
        )

        write_result(cfg, cfg.kde_bandwidth, result_dict_best, result_path)
        if cfg.store_model:
            write_model(cfg, best_model_dict[cfg.metrics], result_path)

    else:
        assert cfg.read_best_model == 1
        pt_full_path = result_path + get_unique_file_name(cfg, "", ".pickle")

        with open(pt_full_path, "rb") as f:
            s = f.read()
            read_model = pickle.loads(s)
        print(f"load best model success from {pt_full_path}!")

        all_begin = time.time()

        print("=" * 30 + " read best model and verify result " + "=" * 30)
        read_result_dic = algorithm.test(
            cfg,
            read_model,
            data_packet["x_test"],
            data_packet["y_test"],
            ("seed = " + str(cfg.seed) + ": Final test for read model: " + pt_full_path + ":\n"),
            True,
            all_begin,
            device,
        )

        algorithm.cal_worst_acc(cfg, data_packet, read_model, read_result_dic, all_begin, ts_data, device)

        write_result(cfg, "read", read_result_dic, result_path, "")


if __name__ == "__main__":
    main(tyro.cli(Config))
