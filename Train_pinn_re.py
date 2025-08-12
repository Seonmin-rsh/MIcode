import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import random
import nibabel as nib
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes
from scipy.ndimage import gaussian_filter
from skimage import morphology
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#--------------------------------------------------------
## SEED 고정 (재현성 확보)
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

## Constants
GAMMA = 2.67502e8  # Hz/T
DELTA_CHI_0 = 0.264e-6
HCT = 0.34
B0 = 3.0  # Tesla

## Hyperparameters
EPOCHS = 100
BATCH_SIZE = 256 # pixel로 batch 설정
LEARNING_RATE = 5e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device : {device}")

#--------------------------------------------------------
#** PINN model
class PINN(nn.Module):
    def __init__(self, gamma, delta_chi0, Hct, B0):
        super(PINN, self).__init__()
        self.gamma = torch.tensor(gamma)
        self.delta_chi0 = torch.tensor(delta_chi0)
        self.Hct = torch.tensor(Hct)
        self.B0 = torch.tensor(B0)

        self.layers = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 2)
        )
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        # SO2 범위
        self.so2_L = 0.40
        self.so2_U = 0.85

    def forward(self, R2, BVf, TE):
        x = torch.cat([R2, BVf], dim=1)
        params_raw = self.layers(x)

        # scaled sigmoid: [0.40, 0.85]
        SO2 = self.so2_L + (self.so2_U - self.so2_L) * self.sigmoid(params_raw[:, 0:1])
        CteFt = self.softplus(params_raw[:, 1:2])

        if TE.ndim == 1:
            TE = TE.unsqueeze(0).repeat(R2.shape[0], 1)

        exponent = -R2 * TE - BVf * self.gamma * (4/3) * np.pi * self.delta_chi0 * \
                   self.Hct * (1 - SO2) * self.B0 * TE
        pred_s = CteFt * torch.exp(exponent)
        return pred_s, SO2, CteFt

#---------------------------------------------------------
# masking image -> main modification (skull-stripping style)

def keep_largest_component(binary_mask):
    """Keep only the largest connected component."""
    labeled = morphology.label(binary_mask)
    if labeled.max() == 0:
        return binary_mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # ignore background
    largest = sizes.argmax()
    return (labeled == largest)

def create_brain_mask_from_sig4(sig4_xyz_te, erosion_iterations=1, dilation_iterations=1,
                                sigma=5, min_object_size=5000, border=20):
    """
    Create brain-only mask (skull excluded) from pre7meGRE subset.
    Steps:
    - TE-average
    - Otsu threshold
    - binary erosion / fill holes / binary dilation
    - keep largest component
    - remove small objects (min_object_size)
    - border trimming (remove ring near skull)
    Return: mask (bool) with shape (X, Y, Z)
    """
    # TE-average -> (X,Y,Z)
    img_mean = np.mean(sig4_xyz_te, axis=3)

    # global normalization for stability
    vmin, vmax = img_mean.min(), img_mean.max()
    img_norm = (img_mean - vmin) / (vmax - vmin) if vmax > vmin else img_mean.copy()

    # smooth to suppress skull rim
    img_smooth = gaussian_filter(img_norm, sigma=sigma)

    H, W, Z = img_smooth.shape
    mask = np.zeros_like(img_smooth, dtype=bool)

    for s in range(Z):
        slc = img_smooth[:, :, s]
        thr = threshold_otsu(slc) if slc.max() > slc.min() else 0.0

        m = slc > thr
        m = binary_erosion(m, iterations=erosion_iterations)
        m = binary_fill_holes(m)
        m = binary_dilation(m, iterations=dilation_iterations)
        m = keep_largest_component(m)
        m = morphology.remove_small_objects(m, min_size=min_object_size)  # remove small objects

        # border trimming
        if border > 0:
            bm = np.ones_like(m, dtype=bool)
            bm[:border, :] = False; bm[-border:, :] = False
            bm[:, :border] = False; bm[:, -border:] = False
            m = m & bm

        mask[:, :, s] = m

    return mask  # (X,Y,Z) bool

def save_mask_nifti(mask_xyz_bool, reference_img, out_path):
    """Save boolean mask to NIfTI using reference affine/header."""
    out_img = nib.Nifti1Image(mask_xyz_bool.astype(np.uint8), reference_img.affine, reference_img.header)
    nib.save(out_img, out_path)

def apply_mask_3d(vol_xyz, mask_xyz_bool):
    """Apply boolean mask to 3D volume (X,Y,Z). Outside mask -> 0."""
    out = np.zeros_like(vol_xyz, dtype=vol_xyz.dtype)
    out[mask_xyz_bool] = vol_xyz[mask_xyz_bool]
    return out

def save_nifti_like(data, reference_img, out_path):
    """Save NIfTI with reference affine/header."""
    img = nib.Nifti1Image(data, reference_img.affine, reference_img.header)
    nib.save(img, out_path)

#--------------------------------------------------------
## collect patients
def collect_patients(root_dir):
    patients = {}
    for pid in os.listdir(root_dir):
        pth = os.path.join(root_dir, pid)
        files = os.listdir(pth)
        paths = {}
        for f in files:
            if "R2" in f:       paths["R2"] = os.path.join(pth, f)
            elif "Bvf" in f:    paths["Bvf"] = os.path.join(pth, f)
            elif "pre7meGRE" in f:
                                paths["pre7meGRE"] = os.path.join(pth, f)
        patients[pid] = paths
    return patients

## data load 
def load_all_patients(patients, TE, slice_idx=[10,24,36]):
    R2_list, BVf_list, Sig_list, ID_list = [], [], [], []
    for pid, paths in tqdm(patients.items(), desc="Loading patients"):
        if set(["R2","Bvf","pre7meGRE"]) - set(paths.keys()):
            print(f"{pid}'s data is missing"); continue

        # --- pre7meGRE NIfTI를 참조로 로드(affine/header 저장용) ---
        pre_img = nib.load(paths["pre7meGRE"])
        pre_all = pre_img.get_fdata()  # 기대: (X,Y,Z,Frame)

        # --- R2/BVf, 신호 로드 ---
        r2   = nib.load(paths["R2"]).get_fdata()
        bvf  = nib.load(paths["Bvf"]).get_fdata()
        sig4 = pre_all[..., slice_idx, 0:7]  # (X,Y,3,7)

        # --- skull 제외 뇌 마스크 생성 + 저장 ---
        mask = create_brain_mask_from_sig4(
            sig4,
            erosion_iterations=1,
            dilation_iterations=1,
            sigma=5,              # 스컬 림 억제
            min_object_size=5000, # 작은 연결 성분 제거
            border=20             # 가장자리 링 제거
        )

        # 저장 경로: <PID>/<PID>_brainmask_from_pre7meGRE.nii.gz
        try:
            out_mask_path = os.path.join(os.path.dirname(paths["pre7meGRE"]),
                                         f"{pid}_brainmask_from_pre7meGRE.nii.gz")
            save_mask_nifti(mask, pre_img, out_mask_path)
        except Exception as e:
            print(f"[WARN] {pid}: mask save failed -> {e}")

        # === masked R2/BVf/pre7meGRE 저장 (QC용) ===
        try:
            # 1) R2 masked (3D)
            r2_masked = apply_mask_3d(r2, mask)
            r2_img = nib.load(paths["R2"])
            out_r2_path = os.path.join(os.path.dirname(paths["R2"]),
                                       f"{pid}_R2_masked.nii.gz")
            save_nifti_like(r2_masked.astype(r2.dtype), r2_img, out_r2_path)

            # 2) BVf masked (3D)
            bvf_masked = apply_mask_3d(bvf, mask)
            bvf_img = nib.load(paths["Bvf"])
            out_bvf_path = os.path.join(os.path.dirname(paths["Bvf"]),
                                        f"{pid}_Bvf_masked.nii.gz")
            save_nifti_like(bvf_masked.astype(bvf.dtype), bvf_img, out_bvf_path)

            # 3) pre7meGRE masked (4D: Z=3, TE=7)
            sig4_masked = np.zeros_like(sig4)
            for te in range(sig4.shape[3]):
                sl = sig4[..., te]          # (X,Y,Z)
                sl_m = apply_mask_3d(sl, mask)
                sig4_masked[..., te] = sl_m
            out_sig4_path = os.path.join(os.path.dirname(paths["pre7meGRE"]),
                                         f"{pid}_pre7meGRE_masked_Z3_TE7.nii.gz")
            sig4_img = nib.Nifti1Image(sig4_masked.astype(sig4.dtype), pre_img.affine)
            nib.save(sig4_img, out_sig4_path)

        except Exception as e:
            print(f"[WARN] {pid}: masked map save failed -> {e}")

        # --- 마스크 적용 (학습용 벡터화) ---
        try:
            R2m = r2[mask].reshape(-1,1)
            BVfm= bvf[mask].reshape(-1,1)
            Sm  = sig4[mask].reshape(-1, len(TE))
        except Exception as e:
            print(f"[ERROR] {pid}: masking index failed ({e}). Check shape/affine/slice_idx.")
            print("      shapes:", r2.shape, bvf.shape, sig4.shape, mask.shape)
            continue

        N = R2m.shape[0]
        R2_list.append(R2m); BVf_list.append(BVfm)
        Sig_list.append(Sm);   ID_list.append(np.full((N,), pid))

    R2_all = torch.tensor(np.concatenate(R2_list), dtype=torch.float32)
    BVf_all= torch.tensor(np.concatenate(BVf_list), dtype=torch.float32)
    Sig_all= torch.tensor(np.concatenate(Sig_list), dtype=torch.float32)
    ID_all = np.concatenate(ID_list)
    return R2_all, BVf_all, Sig_all, ID_all

#--------------------------------------------------------
## Training
def train(model, dataloader, TE):
    opt  = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=0)
    crit = nn.MSELoss()
    hist = []

    # 에폭 루프에 tqdm 적용
    for ep in tqdm(range(1, EPOCHS+1), desc="Epochs"):
        model.train()
        total = 0
        # 각 에폭마다 배치 진행상황도 표시
        for r2, bvf, sig, _ in tqdm(dataloader, desc=f" Epoch {ep}", leave=False):
            r2, bvf, sig = r2.to(device), bvf.to(device), sig.to(device)
            pred, _, _ = model(r2, bvf, TE.to(device))
            loss = crit(pred, sig)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        scheduler.step()

        avg = total / len(dataloader)
        hist.append(avg)
        if ep % 10 == 0:
            print(f"Epoch {ep}/{EPOCHS} Loss: {avg:.6f}")

    # 학습 곡선
    plt.plot(range(1, EPOCHS+1), hist)
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Train loss")
    plt.savefig("train_loss.png", dpi=200)
    plt.close()
    return model

#--------------------------------------------------------
if __name__=="__main__":
    # (1) 준비
    TE = np.array([2,12,22,32,42,52,62]) / 1000
    root_dir = '/home/ailive/jiwon/oxygenation/processed_data'
    train_dir= os.path.join(root_dir, 'train_data')

    # (2) 데이터 로드
    train_pats = collect_patients(train_dir)
    R2_tr, BVf_tr, Sig_tr, ID_tr = load_all_patients(train_pats, TE)
    le = LabelEncoder().fit(ID_tr)
    ID_tensor = torch.tensor(le.transform(ID_tr), dtype=torch.long)

    ds = TensorDataset(R2_tr, BVf_tr, Sig_tr, ID_tensor)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    # (3) 학습
    model = PINN(GAMMA, DELTA_CHI_0, HCT, B0).to(device)
    trained = train(model, loader, torch.tensor(TE, dtype=torch.float32))

    # (4) 저장
    torch.save(trained.state_dict(), "pinn_trained.pth")
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    print(" TRAIN DONE, model saved to pinn_trained.pth")
