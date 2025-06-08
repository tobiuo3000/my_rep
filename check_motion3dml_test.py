import os
import numpy as np

# (1) new_joint_vecs フォルダへのパス
data_dir = "/home/masa/Documents/myRTMP/MotionLCM/datasets/humanml3d/new_joint_vecs"

# フォルダ内の .npy ファイル一覧を取得
files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
if not files:
    raise FileNotFoundError(f"No .npy files found in {data_dir}")

# 試しに最初のファイルを読む
sample_file = files[0]
sample_path = os.path.join(data_dir, sample_file)
motion_data = np.load(sample_path)

# (2) 形状・データ型・簡単な統計情報を表示
print(f"=== Sample: {sample_file} ===")
print(f"Shape      : {motion_data.shape}")
print(f"Dtype      : {motion_data.dtype}")
print(f"Min value  : {np.min(motion_data):.6f}")
print(f"Max value  : {np.max(motion_data):.6f}")
print(f"Mean value : {np.mean(motion_data):.6f}")
print()

# (3) 最初の数フレームを見てみる（行方向がフレーム、列方向が特徴次元）
print("First 5 frames:")
print(motion_data[:5])
print()

# (4) 特徴次元ごとの統計（最初の 5 フレームずつ）
sub = motion_data[:5]
print("Per-feature stats for first 5 frames:")
print("  Mean per feature:", np.mean(sub, axis=0))
print("  Std per feature : ", np.std(sub, axis=0))
print()

# ---------------------------------------
# (5) nfeats 次元 → (frames, joints, 3) に復元したい場合
#     もし MotionLCM の recover_from_ric 関数が利用できるなら以下のように：
#
# from humanml.scripts.motion_process import recover_from_ric
# 
# # nfeats = dataset_map[name][0]（例: 263）
# # njoints = dataset_map[name][1]（例: 22）
# restored_joints = recover_from_ric(motion_data, njoints)
# # restored_joints の shape は (frames, njoints, 3) になるはず
# print("Restored shape (frames, joints, 3):", restored_joints.shape)
# print("First frame joint positions:\n", restored_joints[0])
# ---------------------------------------

