import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy.io as sio
import os

# ===================== 路径配置 =====================
data_root = r"D:\vscode_code\bci\dataset\BCICIV_2a_gdf"
label_root = r"D:\vscode_code\bci\dataset\BCICIV_2a_gdf\labels"
save_path = r"D:\vscode_code\bci\dataset\bciciv_2a_preprocessed_test_A09T.npz"
# ====================================================

sub = "A09T"
print(f"\n========== 处理：{sub} ==========")

# 文件路径
gdf_path = os.path.join(data_root, f"{sub}.gdf")
mat_path = os.path.join(label_root, f"{sub}.mat")

# 1. 读取GDF
raw_gdf = mne.io.read_raw_gdf(
    gdf_path, stim_channel="auto", verbose='ERROR',
    exclude=["EOG-left", "EOG-central", "EOG-right"]
)

# 2. 重命名通道
raw_gdf.rename_channels({
    'EEG-Fz': 'Fz', 'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz', 'EEG-3': 'FC2', 'EEG-4': 'FC4',
    'EEG-5': 'C5', 'EEG-C3': 'C3', 'EEG-6': 'C1', 'EEG-Cz': 'Cz', 'EEG-7': 'C2', 'EEG-C4': 'C4', 'EEG-8': 'C6',
    'EEG-9': 'CP3', 'EEG-10': 'CP1', 'EEG-11': 'CPz', 'EEG-12': 'CP2', 'EEG-13': 'CP4',
    'EEG-14': 'P1', 'EEG-15': 'Pz', 'EEG-16': 'P2', 'EEG-Pz': 'POz'
})

# 3. 去NaN
raw_gdf.load_data()
data = raw_gdf.get_data()
for i_chan in range(data.shape[0]):
    this_chan = data[i_chan]
    data[i_chan] = np.where(this_chan == np.min(this_chan), np.nan, this_chan)
    mask = np.isnan(data[i_chan])
    chan_mean = np.nanmean(data[i_chan])
    data[i_chan, mask] = chan_mean

raw_gdf = mne.io.RawArray(data, raw_gdf.info, verbose="ERROR")

# 4. 读取官方标签
mat = sio.loadmat(mat_path)
labels_sub = mat["classlabel"].squeeze()  # (288,)
event_id_map = {1:7, 2:8, 3:9, 4:10}
labels_sub = np.array([event_id_map[l] for l in labels_sub])

# 5. 手动构建事件（修复核心）
n_trials = len(labels_sub)
sfreq = raw_gdf.info['sfreq']  # 250Hz
cue_times = np.arange(0, n_trials * 5, 5)  # 每个试次间隔5秒
events = np.zeros((n_trials, 3), dtype=int)
events[:, 0] = (cue_times * sfreq).astype(int)
events[:, 2] = labels_sub

# 6. 截取1~4秒
tmin, tmax = 1.0, 4.0
event_id = {'769':7, '770':8, '771':9, '772':10}

epochs = mne.Epochs(
    raw_gdf, events, event_id, tmin, tmax,
    baseline=None, preload=True, verbose=False
)

# 7. 获取数据和标签
sub_data = epochs.get_data()
sub_labels = epochs.events[:, -1]

# 8. 保存
np.savez_compressed(save_path, data=sub_data, labels=sub_labels)

print(f"\n✅ A09T 预处理完成！")
print(f"数据形状: {sub_data.shape}")
print(f"标签形状: {sub_labels.shape}")
print(f"文件保存至: {save_path}")