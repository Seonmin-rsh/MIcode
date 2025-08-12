import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import os
import nibabel as nib
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes, gaussian_filter
from skimage import morphology
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------
# Constants & Device
GAMMA       = 2.67502e8
DELTA_CHI_0 = 0.264e-6
HCT         = 0.34
B0          = 3.0
BATCH_SIZE  = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device : {device}")

# --------------------------------------------------------
# PINN model (train과 동일 구조/스케일)
class PINN(nn.Module):
    def __init__(self, gamma, delta_chi0, Hct, B0):
        super().__init__()
        self.gamma      = torch.tensor(gamma)
        self.delta_chi0 = torch.tensor(delta_chi0)
        self.Hct        = torch.tensor(Hct)
        self.B0         = torch.tensor(B0)

        self.layers = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 2)
        )
        self.sigmoid  = nn.Sigmoid()
        self.softplus = nn.Softplus()

        # SO2 범위 (train과 동일)
        self.so2_L = 0.40
        self.so2_U = 0.85

    def forward(self, R2, BVf, TE):
        x = torch.cat([R2, BVf], dim=1)
        raw = self.layers(x)
        SO2   = self.so2_L + (self.so2_U - self.so2_L) * self.sigmoid(raw[:,0:1])
        CteFt = self.softplus(raw[:,1:2])

        if TE.ndim == 1:
            TE = TE.unsqueeze(0).repeat(R2.shape[0], 1)

        exponent = -R2 * TE - BVf * self.gamma * (4/3) * np.pi * self.delta_chi0 * \
                   self.Hct * (1 - SO2) * self.B0 * TE
        pred_s = CteFt * torch.exp(exponent)  # [B,E]
        return pred_s, SO2, CteFt

# ---------------------------------------------------------
# 마스크 (train과 동일 로직)
def keep_largest_component(binary_mask):
    labeled = morphology.label(binary_mask)
    if labeled.max() == 0:
        return binary_mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return (labeled == sizes.argmax())

def create_mask(original_signal, slice_idx, erosion_iterations=2, dilation_iterations=2):
    """
    original_signal: pre7meGRE (H,W,Z,Frame) 또는 (H,W,Z)
    slice_idx: 사용할 z-슬라이스 리스트 (유효 슬라이스만)
    TE는 0:7 고정, TE 평균 후 2D Otsu + morphology
    """
    if original_signal.ndim == 3:
        original_signal = original_signal[..., np.newaxis]  # 4D 보장

    signal = original_signal[:, :, slice_idx, 0:7]          # (H,W,len(sel),7)
    image  = np.mean(signal, axis=3)                        # (H,W,Z)

    H, W, Z = image.shape
    brain_mask = np.zeros((H, W, Z), dtype=bool)

    for i in range(Z):
        slc = image[:, :, i]
        if slc.max() > slc.min():
            slc = (slc - slc.min()) / (slc.max() - slc.min())
        smoothed = gaussian_filter(slc, sigma=1)
        thr = threshold_otsu(smoothed)
        m = smoothed > thr
        m = binary_erosion(m, iterations=erosion_iterations)
        m = binary_fill_holes(m)
        m = binary_dilation(m, iterations=dilation_iterations)
        m = keep_largest_component(m)
        brain_mask[:, :, i] = m
    return brain_mask

# --------------------------------------------------------
# 환자 수집 (원본만 선택: masked/brainmask/결과물 제외)
def collect_patients(root_dir):
    patients = {}
    for pid in os.listdir(root_dir):
        pth = os.path.join(root_dir, pid)
        if not os.path.isdir(pth):
            continue
        files = sorted(os.listdir(pth))
        paths = {}
        for f in files:
            if ("R2_map" in f) and ("masked" not in f) and ("brainmask" not in f):
                paths["R2"] = os.path.join(pth, f); break
        for f in files:
            if ("Bvf_map" in f) and ("masked" not in f) and ("brainmask" not in f):
                paths["Bvf"] = os.path.join(pth, f); break
        for f in files:
            if ("pre7meGRE" in f) and ("masked" not in f) and ("brainmask" not in f):
                paths["pre7meGRE"] = os.path.join(pth, f); break
        patients[pid] = paths
    return patients

# 시각화 + SO2 NIfTI 저장 (결과는 test_results_min2/<pid>/ 아래로만 저장)
def visualize_and_save_so2(so2_flat, mask3d, affine, out_dir, pid, slices_to_show):
    os.makedirs(out_dir, exist_ok=True)
    H, W, Z = mask3d.shape
    so2_vol = np.zeros((H, W, Z), dtype=np.float32)
    so2_vol[mask3d] = so2_flat.squeeze()

    # PNG
    fig, axes = plt.subplots(1, len(slices_to_show), figsize=(5*len(slices_to_show), 5))
    if len(slices_to_show) == 1:
        axes = [axes]
    for i, s in enumerate(slices_to_show):
        im = axes[i].imshow(so2_vol[:,:,s], cmap='hot', vmin=0, vmax=1)
        axes[i].axis('off'); axes[i].set_title(f"Slice {s}")
    fig.colorbar(im, ax=axes, orientation='vertical', shrink=0.7)
    plt.savefig(os.path.join(out_dir, f"{pid}_so2_map_min.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # NIfTI
    nib.save(nib.Nifti1Image(so2_vol, affine), os.path.join(out_dir, f"{pid}_so2_map_min.nii"))

# --------------------------------------------------------
if __name__=="__main__":
    # 설정
    TE         = np.array([2,12,22,32,42,52,62]) / 1000
    root_dir   = '/home/ailive/jiwon/oxygenation/processed_data'
    test_dir   = os.path.join(root_dir, 'test_data')
    ckpt       = "pinn_trained_min2.pth"   # <- train에서 저장한 이름과 맞추기
    slice_idx  = [10,24,36]                # 요청 슬라이스 (유효 범위 자동 필터)
    show_slices= None                      # None이면 유효 슬라이스 전부 표시

    save_dir   = "./test_results_min2"
    os.makedirs(save_dir, exist_ok=True)

    # (1) 모델
    model = PINN(GAMMA, DELTA_CHI_0, HCT, B0).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # (2) 환자 목록(원본만)
    test_pats = collect_patients(test_dir)
    patient_ids = sorted(test_pats.keys())

    for pid in patient_ids:
        paths = test_pats[pid]
        if set(["R2","Bvf","pre7meGRE"]) - set(paths.keys()):
            print(f"[WARN] {pid}: missing required files")
            continue

        # (3) 데이터 로드
        r2  = nib.load(paths["R2"]).get_fdata()    # (H,W,Z)
        bvf = nib.load(paths["Bvf"]).get_fdata()   # (H,W,Z)
        pre_img = nib.load(paths["pre7meGRE"])
        pre = pre_img.get_fdata()                  # (H,W,Z,F) or (H,W,Z)

        if pre.ndim == 3:
            pre = pre[..., np.newaxis]
        if pre.shape[3] < 7:
            print(f"[WARN] {pid}: pre7meGRE has only {pre.shape[3]} frames (<7). Skip.")
            continue

        # 유효 슬라이스 선택
        Z = pre.shape[2]
        sel = [s for s in slice_idx if 0 <= s < Z] or list(range(Z))
        sig4 = pre[..., sel, 0:7]  # (H,W,len(sel),7)

        # (4) 마스크 (train과 동일 방식)
        mask3d = create_mask(pre, slice_idx=sel, erosion_iterations=2, dilation_iterations=2)  # (H,W,len(sel))

        # 안전성 체크 (형상 일치)
        assert r2.shape[:2] == bvf.shape[:2] == mask3d.shape[:2], f"{pid}: XY mismatch"
        assert r2.shape[2] >= max(sel)+1 and bvf.shape[2] >= max(sel)+1, f"{pid}: Z smaller than sel"

        # (5) 마스킹된 벡터로 추론
        R2m  = r2[:, :, sel][mask3d].reshape(-1,1).astype(np.float32)
        BVfm = bvf[:, :, sel][mask3d].reshape(-1,1).astype(np.float32)

        ds = TensorDataset(torch.from_numpy(R2m), torch.from_numpy(BVfm))
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

        so2_list = []
        with torch.no_grad():
            for r2b, bvfb in loader:
                r2b  = r2b.to(device)
                bvfb = bvfb.to(device)
                _, so2b, _ = model(r2b, bvfb, torch.tensor(TE, dtype=torch.float32, device=device))
                so2_list.append(so2b.cpu().numpy())
        so2_flat = np.concatenate(so2_list, axis=0)  # (N_vox, 1)

        # (6) 저장: 환자별 전용 폴더에만 저장(섞임 방지)
        pid_out = os.path.join(save_dir, pid)
        if show_slices is None:
            slices_to_show = list(range(mask3d.shape[2]))
        else:
            slices_to_show = [s for s in show_slices if 0 <= s < mask3d.shape[2]]

        visualize_and_save_so2(
            so2_flat=so2_flat,
            mask3d=mask3d,
            affine=pre_img.affine,
            out_dir=pid_out,
            pid=pid,
            slices_to_show=slices_to_show
        )
