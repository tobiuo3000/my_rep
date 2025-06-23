# infer_rich.py

import os, sys
repo_root = os.path.dirname(__file__)
motionlcm_dir = os.path.join(repo_root, "MotionLCM")
sys.path.insert(0, motionlcm_dir)

import os
import numpy as np
import torch

# ① モデル定義・ロード
from MotionLCM.mld.models.modeltype.mld import MotionLCMModule

# GPU/CPU の選択
device = "cuda" if torch.cuda.is_available() else "cpu"

# ② チェックポイントパス（適宜ご自分のパスに合わせてください）
CKPT = os.path.expanduser("checkpoints/motionlcm.ckpt")

# モデルをロード
model = MotionLCMModule.load_from_checkpoint(
    CKPT,
    map_location=device
)
model.eval().to(device)

# ③ Rich データから hint motion を準備
#    （最後の 10 フレームを使う例）
RICH_DIR = os.path.expanduser("MotionLCM/try0606/sample")
# rich_sample_263.npy のようなファイル名に合わせる
np_files = sorted([f for f in os.listdir(RICH_DIR) if f.endswith(".npy")])
# 例えば最初のサンプルを使うなら:
hint_path = os.path.join(RICH_DIR, np_files[0])
data = np.load(hint_path)  # shape (T, J, C)

# 切り出すフレーム数
K = 10
hint_np = data[-K:]                   # (K, J, C)
# モデル入力形状に合わせて (1, K, J*C) へ reshape or unsqueeze
# MotionLCMModule.sample が受け取る形状に応じて reshape を調整してください。
# たとえば内部で (B, T, NFEATS) を期待している場合：
hint_feats = torch.from_numpy(hint_np.reshape(1, K, -1)).to(device).float()

# ④ 推論
with torch.no_grad():
    # cond=None, text_cond=None で純粋に hint → 生成
    gen = model.sample(
        hint_motion=hint_feats,
        num_frames=30,
        cond=None,
        text_cond=None
    )  # 返り値は Tensor shape (1, out_len, NFEATS)

# ⑤ 結果を保存
out = gen.squeeze(0).cpu().numpy()  # (out_len, NFEATS)
np.save("generated_from_rich.npy", out)
print(f"Generated motion saved to generated_from_rich.npy, shape={out.shape}")

