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
import re  # <-- 추가: 파일명 정리용

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
EPOCHS = 50
BATCH_SIZE = 256 # pixel로 batch 설정
LEARNING_RATE = 3e-4

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
            nn.Linear(64, 2)
        )
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        # # SO2 범위
        # self.so2_L = 0.40
        # self.so2_U = 0.85

    def forward(self, R2, BVf, TE):
        x = torch.cat([R2, BVf], dim=1)
        params_raw = self.layers(x)

        SO2 = self.sigmoid(params_raw[:, 0:1])
        CteFt = self.softplus(params_raw[:, 1:2])

        # # scaled sigmoid: [0.40, 0.85]
        # SO2 = self.so2_L + (self.so2_U - self.so2_L) * self.sigmoid(params_raw[:, 0:1])
        # CteFt = self.softplus(params_raw[:, 1:2])

        if TE.ndim == 1:
            TE = TE.unsqueeze(0).repeat(R2.shape[0], 1)

        exponent = -R2 * TE - BVf * self.gamma * (4/3) * np.pi * self.delta_chi0 * \
                   self.Hct * (1 - SO2) * self.B0 * TE
        pred_s = CteFt * torch.exp(exponent)
        return pred_s, SO2, CteFt

#---------------------------------------------------------
def create_mask(original_signal, slice_idx=[10,24,36], erosion_iterations = 2, dilation_iterations = 2):
    # FIX: original_signal은 pre7meGRE 4D 전체로 받고, 여기서 슬라이스+TE(0:7)를 선택
    if original_signal.ndim == 3:
        original_signal = original_signal[..., np.newaxis]      # 4D 보장
    signal = original_signal[:, :, slice_idx, 0:7]              # (H,W,len(slices),7)

    image = np.mean(signal, axis=3)                             # (H,W,Z)
    H, W, Z = image.shape
    brain_mask = np.zeros((H, W, Z), dtype=bool)                # FIX: bool 마스크

    for i in range(Z):
        slc = image[:, :, i]                                    # (H,W)
        if slc.max() > slc.min():
            normalized_slice = (slc - slc.min()) / (slc.max() - slc.min())
        else:
            normalized_slice = slc

        smoothed = gaussian_filter(normalized_slice, sigma = 1)
        thr = threshold_otsu(smoothed)                          # 2D Otsu (경고 없음)
        m = smoothed > thr

        m = binary_erosion(m, iterations=erosion_iterations)
        m = binary_fill_holes(m)
        m = binary_dilation(m, iterations=dilation_iterations)
        m = keep_largest_component(m)

        brain_mask[:, :, i] = m

    return brain_mask

def keep_largest_component(binary_mask):
    labeled_mask = morphology.label(binary_mask)
    if labeled_mask.max() == 0:
        return binary_mask
    component_sizes = np.bincount(labeled_mask.ravel())
    component_sizes[0] = 0  # 배경 제외
    largest_component = component_sizes.argmax()
    final_mask = (labeled_mask == largest_component)
    return final_mask

# ---------- 추가: 안전한 파일명/경로 생성 유틸 ----------
def _clean_base(path):
    """원본 파일명에서 .nii/.nii.gz와 기존 마스크 접미사 제거"""
    base = os.path.basename(path)
    base = re.sub(r'\.nii(\.gz)?$', '', base)                  # 확장자 제거
    base = re.sub(r'(_masked_min|_masked)$', '', base)         # 중복 접미사 제거
    return base

def _masked_dir_of(path):
    """원본과 같은 폴더 내 masked/ 서브폴더 경로"""
    d = os.path.join(os.path.dirname(path), "masked")
    os.makedirs(d, exist_ok=True)
    return d
# --------------------------------------------------------

## collect patients
def collect_patients(root_dir):
    patients = {}
    for pid in os.listdir(root_dir):
        pth = os.path.join(root_dir, pid)
        if not os.path.isdir(pth):
            continue
        files = sorted(os.listdir(pth))  # 결정적 선택
        paths = {}

        # 원본만 선택: masked/brainmask/결과물 제외
        for f in files:
            if ("R2_map" in f) and ("masked" not in f) and ("brainmask" not in f):
                paths["R2"] = os.path.join(pth, f)
                break
        for f in files:
            if ("Bvf_map" in f) and ("masked" not in f) and ("brainmask" not in f):
                paths["Bvf"] = os.path.join(pth, f)
                break
        for f in files:
            if ("pre7meGRE" in f) and ("masked" not in f) and ("brainmask" not in f):
                paths["pre7meGRE"] = os.path.join(pth, f)
                break

        patients[pid] = paths
    return patients

## data load 
def load_all_patients(patients, TE, slice_idx=[10,24,36]):
    R2_list, BVf_list, Sig_list, ID_list = [], [], [], []
    for pid, paths in tqdm(patients.items(), desc="Loading patients"):
        if set(["R2","Bvf","pre7meGRE"]) - set(paths.keys()):
            print(f"{pid}'s data is missing"); continue
        r2   = nib.load(paths["R2"]).get_fdata()
        bvf  = nib.load(paths["Bvf"]).get_fdata()

        # FIX: pre7meGRE를 먼저 4D로 로드하고 유효 슬라이스로 자른 뒤 TE 0:7 지정
        pre = nib.load(paths["pre7meGRE"]).get_fdata()
        if pre.ndim == 3:
            pre = pre[..., np.newaxis]                          # 4D 보장
        if pre.shape[3] < 7:
            print(f"[WARN] {pid}: pre7meGRE has only {pre.shape[3]} frames (<7). Skipping."); 
            continue

        Z = pre.shape[2]
        sel = [s for s in slice_idx if 0 <= s < Z] or list(range(Z))  # 유효 슬라이스만
        sig4 = pre[..., sel, 0:7]                                     # (H,W,len(sel),7)

        # FIX: 마스크는 pre 전체와 동일 슬라이스(sel)로 생성 (함수 내부에서 0:7 사용)
        mask = create_mask(pre, slice_idx=sel, erosion_iterations = 2, dilation_iterations = 2)

        # === (추가) 마스킹 저장 ===
        try:
            # 저장 폴더: masked/ (원본과 분리)
            out_dir = _masked_dir_of(paths["R2"])

            # R2_map masked 저장
            r2_img     = nib.load(paths["R2"])
            r2_masked  = np.zeros_like(r2, dtype=r2.dtype)
            r2_masked[mask] = r2[mask]
            r2_base = _clean_base(paths["R2"])
            r2_out  = os.path.join(out_dir, f"{r2_base}_masked_min.nii.gz")
            nib.save(nib.Nifti1Image(r2_masked, r2_img.affine, r2_img.header), r2_out)

            # Bvf_map masked 저장
            bvf_img     = nib.load(paths["Bvf"])
            bvf_masked  = np.zeros_like(bvf, dtype=bvf.dtype)
            bvf_masked[mask] = bvf[mask]
            bvf_base = _clean_base(paths["Bvf"])
            bvf_out  = os.path.join(out_dir, f"{bvf_base}_masked_min.nii.gz")
            nib.save(nib.Nifti1Image(bvf_masked, bvf_img.affine, bvf_img.header), bvf_out)

        except Exception as e:
            print(f"[WARN] {pid}: saving masked maps failed -> {e}")

        R2m = r2[mask].reshape(-1,1)
        BVfm= bvf[mask].reshape(-1,1)
        Sm  = sig4[mask].reshape(-1, len(TE))
        N   = R2m.shape[0]

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
    plt.savefig("train_loss_min.png", dpi=200)
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
    torch.save(trained.state_dict(), "pinn_trained_min2.pth")
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    print(" TRAIN DONE, model saved to pinn_trained_min2.pth")