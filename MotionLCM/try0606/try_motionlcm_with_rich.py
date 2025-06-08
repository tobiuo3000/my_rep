import os
import pickle
import sys
import datetime
import logging
import os.path as osp

from omegaconf import OmegaConf

import torch
import numpy as np
from mld.config import parse_args
from mld.data.get_data import get_dataset
from mld.models.modeltype.mld import MLD
from mld.models.modeltype.vae import VAE
from mld.utils.utils import set_seed, move_batch_to_device
from mld.data.humanml.utils.plot_script import plot_3d_motion
from mld.utils.temos_utils import remove_padding

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

NPY_PATH = "/home/masa/Documents/MotionLCM/rich_sample_263.npy"
np263 = np.load(NPY_PATH)[:200]
print(np263.shape)

tmp_path = "/home/masa/Documents/MotionLCM/datasets/humanml_spatial_norm/Mean_raw.npy"
tmp_path2 = "/home/masa/Documents/MotionLCM/datasets/humanml_spatial_norm/Std_raw.npy"
print(np.load(tmp_path).shape)
print(np.load(tmp_path2).shape)

# Python で直接 MotionLCM.forward を呼ぶ場合のイメージ
batch = {}
batch["motion"] = torch.zeros((1, 200, 263))      # ダミー(殆ど使わない)
batch["mask"] = torch.ones((1, 200), dtype=torch.bool)
batch["hint"] = torch.from_numpy(np263).unsqueeze(0)  # (1,200,263)
hint_mask = torch.ones((1,200), dtype=torch.bool)  # 全部 True (=全部 PAD)
hint_mask[:, :100] = False                         # 最初100フレームだけ非PAD (=hint)
batch["hint_mask"] = hint_mask.to(device)
# text_emb はテキストエンコーダから得る

model = MLD()
recon_motion = model.forward(batch, return_loss=False, text=text_emb)

