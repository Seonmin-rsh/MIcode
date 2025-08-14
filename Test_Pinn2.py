import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import os
import nibabel as nib
from skimage.filters import threshold_otsu
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")            # ─── 수정: 백엔드 변경
import matplotlib.pyplot as plt
# Remove unused LabelEncoder / pickle imports

#--------------------------------------------------------
## Constants & Device
GAMMA       = 2.67502e8
DELTA_CHI_0 = 0.264e-6
HCT         = 0.34
B0          = 3.0

BATCH_SIZE  = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device : {device}")

#--------------------------------------------------------
# PINN model (same as train)
class PINN(nn.Module):
    def __init__(self, gamma, delta_chi0, Hct, B0):
        super().__init__()
        self.gamma      = gamma
        self.delta_chi0 = delta_chi0
        self.Hct        = Hct
        self.B0         = B0
        self.layers     = nn.Sequential(
            nn.Linear(2,64), nn.Tanh(),
            nn.Linear(64,64), nn.Tanh(),
            nn.Linear(64,2)
        )
        self.sigmoid   = nn.Sigmoid()
        self.softplus  = nn.Softplus()

    def forward(self, R2, BVf, TE):
        x = torch.cat([R2, BVf], dim=1)
        raw = self.layers(x)
        SO2   = self.sigmoid(raw[:,0:1])
        CteFt = self.softplus(raw[:,1:2])
        if TE.ndim == 1:
            TE = TE.unsqueeze(0).repeat(R2.shape[0],1)
        exp_term = -R2*TE - BVf*self.gamma*(4/3)*np.pi*self.delta_chi0*self.Hct*(1-SO2)*self.B0*TE
        return CteFt * torch.exp(exp_term), SO2, CteFt

# masking
def otsu_mask(image_4d):
    img = np.mean(image_4d, axis=3)
    m   = np.zeros_like(img, bool)
    for s in range(img.shape[2]):
        thr = threshold_otsu(img[:,:,s])
        m[:,:,s] = img[:,:,s] > thr
    return m

# collect + load (same as train)
def collect_patients(root_dir):
    patients = {}
    for pid in os.listdir(root_dir):
        pth   = os.path.join(root_dir, pid)
        files = os.listdir(pth)
        paths = {}
        for f in files:
            if "R2" in f:
                paths["R2"] = os.path.join(pth, f)
            elif "Bvf" in f:
                paths["Bvf"] = os.path.join(pth, f)
            elif "pre7meGRE" in f:
                paths["pre7meGRE"] = os.path.join(pth, f)
        patients[pid] = paths
    return patients

def load_all_patients(patients, TE, slice_idx=[10,24,36]):
    R2_list, BVf_list, Sig_list, ID_list = [], [], [], []
    for pid, paths in tqdm(patients.items(), desc="Loading patients"):
        if set(["R2","Bvf","pre7meGRE"]) - set(paths.keys()):
            print(f"{pid} missing"); continue
        r2   = nib.load(paths["R2"]).get_fdata()
        bvf  = nib.load(paths["Bvf"]).get_fdata()
        sig4 = nib.load(paths["pre7meGRE"]).get_fdata()[..., slice_idx, 0:7]
        mask = otsu_mask(sig4)
        R2m  = r2[mask].reshape(-1,1)
        BVfm = bvf[mask].reshape(-1,1)
        Sm   = sig4[mask].reshape(-1, len(TE))
        N    = R2m.shape[0]
        R2_list.append(R2m); BVf_list.append(BVfm)
        Sig_list.append(Sm);   ID_list.append(np.full((N,), pid))
    return (
        torch.tensor(np.concatenate(R2_list), dtype=torch.float32),
        torch.tensor(np.concatenate(BVf_list), dtype=torch.float32),
        torch.tensor(np.concatenate(Sig_list), dtype=torch.float32),
        np.concatenate(ID_list)
    )

# 시각화
def visualize_so2_slices_by_patient(SO2_pred_flat, mask, shape, slice_idx, save_path, pid, nii_affine):
    H, W, S = shape
    so2_map = np.zeros((H, W, S), dtype=np.float32)

    # 1) SO₂를 %로 변환
    so2_map[mask] = SO2_pred_flat.squeeze() * 100  

    fig, axes = plt.subplots(1, len(slice_idx), figsize=(5*len(slice_idx), 5))
    for i, s in enumerate(slice_idx):
        im = axes[i].imshow(so2_map[:, :, s], cmap='hot', vmin=0, vmax=100)  # 2) vmin/vmax를 % 기준으로
        axes[i].axis('off')
        axes[i].set_title(f"Slice {s}")
    fig.suptitle(f"SO2 Map for {pid} (%)")
    fig.colorbar(im, ax=axes, orientation='vertical', shrink=0.7, label='SO₂ (%)')

    plt.savefig(os.path.join(save_path, f"{pid}_so2_map.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ─── NIfTI 저장 (값은 %)
    nii_img = nib.Nifti1Image(so2_map, affine=nii_affine)
    nib.save(nii_img, os.path.join(save_path, f"{pid}_so2_map.nii"))


#--------------------------------------------------------
if __name__=="__main__":
    TE         = np.array([2,12,22,32,42,52,62]) / 1000
    root_dir   = '/home/ailive/jiwon/oxygenation/processed_data'
    test_dir   = os.path.join(root_dir, 'test_data')
    ckpt       = "pinn_trained.pth"
    slice_idx  = [10,24,36]
    vis_slices = [0,1,2]
    save_dir   = "./test_results"
    os.makedirs(save_dir, exist_ok=True)

    # (1) 모델 불러오기
    model = PINN(GAMMA, DELTA_CHI_0, HCT, B0).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # (2) 환자 리스트 순회 ─── 수정
    test_pats   = collect_patients(test_dir)
    patient_ids = sorted(test_pats.keys())

    for pid in patient_ids:
        paths = test_pats[pid]

        # (3) 한 환자 데이터 로드
        R2_t, BVf_t, Sig_t, _ = load_all_patients({pid: paths}, TE, slice_idx)

        # (4) DataLoader without IDs
        ds     = TensorDataset(R2_t, BVf_t, Sig_t)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

        # (5) 추론
        preds, so2s, ctefs = [], [], []
        with torch.no_grad():
            for r2, bvf, sig in loader:
                r2, bvf, sig = r2.to(device), bvf.to(device), sig.to(device)
                p, s, c = model(r2, bvf, torch.tensor(TE, dtype=torch.float32).to(device))
                preds.append(p.cpu().numpy())
                so2s.append(s.cpu().numpy())
                ctefs.append(c.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        so2s  = np.concatenate(so2s, axis=0)
        ctefs = np.concatenate(ctefs, axis=0)

        # (6) 저장
        np.save(os.path.join(save_dir, f"{pid}_pred_signal.npy"), preds)
        np.save(os.path.join(save_dir, f"{pid}_so2_pred.npy"),     so2s)
        np.save(os.path.join(save_dir, f"{pid}_cteft_pred.npy"),   ctefs)

        # (7) 시각화 & NIfTI 저장 ─── 수정
        full   = nib.load(paths["pre7meGRE"]).get_fdata()[..., slice_idx, 0:7]
        mask3d = otsu_mask(full)
        affine = nib.load(paths["pre7meGRE"]).affine
        visualize_so2_slices_by_patient(
            SO2_pred_flat=so2s,
            mask=mask3d,
            shape=(512,512,len(slice_idx)),
            slice_idx=vis_slices,
            save_path=save_dir,
            pid=pid,
            nii_affine=affine
        )
