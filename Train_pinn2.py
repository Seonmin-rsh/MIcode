import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import random
import nibabel as nib
from skimage.filters import threshold_otsu
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
# masking image -> main modification
def otsu_mask(image_4d):
    image = np.mean(image_4d, axis=3)
    mask  = np.zeros_like(image, dtype=bool)
    for s in range(image.shape[2]):
        thr = threshold_otsu(image[:,:,s])
        mask[:,:,s] = image[:,:,s] > thr
    return mask

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
        r2   = nib.load(paths["R2"]).get_fdata()
        bvf  = nib.load(paths["Bvf"]).get_fdata()
        sig4 = nib.load(paths["pre7meGRE"]).get_fdata()[..., slice_idx, 0:7]
        mask = otsu_mask(sig4)

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
