import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import matplotlib.pyplot as plt
import random
import nibabel as nib
from skimage.filters import threshold_otsu
from tqdm import tqdm

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
LEARNING_RATE = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device : {device}")

#--------------------------------------------------------
#** PINN model
import torch
import torch.nn as nn
import numpy as np

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
# masking image
def otsu_mask(image_4d):
    image = np.mean(image_4d, axis=3)  # shape: (H, W, S) #3D이므로 마스크도 동일하게 3D shape
    mask = np.zeros_like(image, dtype=bool)  # shape: (H, W, S)
    for s in range(image.shape[2]):
        slice_img = image[:, :, s]
        threshold = threshold_otsu(slice_img)
        mask[:, :, s] = slice_img > threshold
    return mask  # shape: (H, W, S)

#--------------------------------------------------------
## collect patients
# 수정 1: 조건문 정확히 수정
def collect_patients(root_dir):
    patients = {}
    for patient_id in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient_id)
        nii_files = os.listdir(patient_path)
        paths = {}

        for f in nii_files:
            if "R2" in f:
                paths["R2"] = os.path.join(patient_path, f)
            elif "Bvf" in f:
                paths["Bvf"] = os.path.join(patient_path, f)
            elif "pre7meGRE" in f:  # 수정된 부분
                paths["pre7meGRE"] = os.path.join(patient_path, f)

        patients[patient_id] = paths
    return patients

## data load 
# 수정 2: patients_name → pid 하나만 사용
def load_all_patients(patients, TE, slice_idx=[10, 24, 36]):
    R2_list, BVf_list, Signal_list, ID_list = [], [], [], []
    for pid, paths in tqdm(patients.items(), total=len(patients), desc="Loading patients"):  # 수정된 부분
        if "R2" not in paths or "Bvf" not in paths or "pre7meGRE" not in paths:
            print(f"{pid}'s data is missing")
            continue

        r2_map = nib.load(paths["R2"]).get_fdata()
        bvf_map = nib.load(paths["Bvf"]).get_fdata()
        signal = nib.load(paths["pre7meGRE"]).get_fdata()
        signal = signal[:, :, slice_idx, 0:7]

        mask = otsu_mask(signal)

        R2_masked = r2_map[mask].reshape(-1, 1)
        BVf_masked = bvf_map[mask].reshape(-1, 1)
        Signal_masked = signal[mask].reshape(-1, len(TE))
        N = R2_masked.shape[0]

        R2_list.append(R2_masked)
        BVf_list.append(BVf_masked)
        Signal_list.append(Signal_masked)
        ID_list.append(np.full((N,), pid))

    R2_all = torch.tensor(np.concatenate(R2_list), dtype=torch.float32)
    BVf_all = torch.tensor(np.concatenate(BVf_list), dtype=torch.float32)
    Signal_all = torch.tensor(np.concatenate(Signal_list), dtype=torch.float32)
    ID_all = np.concatenate(ID_list)  # 문자열이므로 torch.tensor 변환 생략

    TE_tensor = torch.tensor(TE, dtype=torch.float32)
    return R2_all, BVf_all, Signal_all, ID_all, TE_tensor

#--------------------------------------------------------
## Training
def train(model, dataloader, TE_data):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    loss_history = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for R2_batch, Bvf_batch, Signal_batch, _ in loop:
            R2_batch = R2_batch.to(device)
            Bvf_batch = Bvf_batch.to(device)
            Signal_batch = Signal_batch.to(device)

            pred_S, pred_SO2, pred_CteFt = model(R2_batch, Bvf_batch, TE_data.to(device))
            loss = criterion(pred_S, Signal_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.6f}, '
                  f'SO2 mean: {pred_SO2.mean().item():.4f}, CteFt mean: {pred_CteFt.mean().item():.4f}')

    plt.plot(range(1, EPOCHS+1), loss_history)
    plt.xlabel("epochs"); plt.ylabel("loss"); plt.title("Train loss")
    plt.show()
    return model

#--------------------------------------------------------
## Test
def test(model, dataloader, TE_data, ID_all, label_encoder):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0

    results = {}  # 각 환자별 저장용

    with torch.no_grad():
        for R2_batch, Bvf_batch, Signal_batch, ID_batch in dataloader:
            R2_batch = R2_batch.to(device)
            Bvf_batch = Bvf_batch.to(device)
            Signal_batch = Signal_batch.to(device)

            pred_S, pred_SO2, pred_Cteft = model(R2_batch, Bvf_batch, TE_data.to(device))
            loss = criterion(pred_S, Signal_batch)
            total_loss += loss.item()

            # ID별로 결과 저장
            for i in range(R2_batch.size(0)):
                pid_encoded = ID_batch[i].item()
                pid = label_encoder.inverse_transform([pid_encoded])[0]
                if pid not in results:
                    results[pid] = {"signal": [], "so2": [], "cteft": []}
                results[pid]["signal"].append(pred_S[i].cpu().numpy())
                results[pid]["so2"].append(pred_SO2[i].cpu().numpy())
                results[pid]["cteft"].append(pred_Cteft[i].cpu().numpy())

    print(f"Test Loss: {total_loss/len(dataloader):.6f}")
    return results

#--------------------------------------------------------
# (1) SO2 맵 시각화
def visualize_so2_slices(SO2_pred, shape, slice_idx=[10, 24, 36]):
    H, W, S = shape
    so2_map = SO2_pred.reshape(S, H, W).transpose(1, 2, 0)
    fig, axes = plt.subplots(1, len(slice_idx), figsize=(15, 5))
    for i in range(len(slice_idx)):
        axes[i].imshow(so2_map[:, :, i], cmap='hot', vmin=0, vmax=1)
        axes[i].set_title(f"SO2 Map (Slice {slice_idx[i]})")
        axes[i].axis('off')
    fig.colorbar(axes[-1].imshow(so2_map[:, :, 0], cmap='hot', vmin=0, vmax=1), ax=axes, location='right', shrink=0.7)
    plt.suptitle("SO2 Maps for Selected Slices")
    plt.savefig("so2_maps.png", dpi=300, bbox_inches='tight')
    plt.show()

# (2) Residual 분석
def plot_residual_hist(Signal, pred_signal):
    residual = (Signal - pred_signal).numpy()
    rmse = np.sqrt(np.mean(residual**2, axis=1))
    plt.hist(rmse, bins=50)
    plt.title("Pixel-wise RMSE Distribution")
    plt.xlabel("RMSE")
    plt.ylabel("Count")
    plt.show()

#--------------------------------------------------------
## Main 실행
if __name__ == "__main__":
    TE = np.array([2, 12, 22, 32, 42, 52, 62]) / 1000
    root_dir = '/home/ailive/jiwon/oxygenation/processed_data'
    train_dir = os.path.join(root_dir, 'train_data')
    test_dir = os.path.join(root_dir, 'test_data')

    train_patients = collect_patients(train_dir)
    test_patients = collect_patients(test_dir)

    # 수정 3: all_patients 관련 블록 제거 (삭제)
    # train_patients = {pid: paths for pid, paths in all_patients.items() if pid.startswith('train')}
    # test_patients  = {pid: paths for pid, paths in all_patients.items() if pid.startswith('test_patient')}

    # (3) Train 환자 데이터 로드
    R2_train, BVf_train, Signal_train, ID_train, TE_tensor = load_all_patients(train_patients, TE)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    ID_train_tensor = torch.tensor(le.fit_transform(ID_train), dtype=torch.long)

    # 여기 수정됨: ID_train → ID_train_tensor
    train_dataset = TensorDataset(R2_train, BVf_train, Signal_train, ID_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 모델 학습
    model = PINN(GAMMA, DELTA_CHI_0, HCT, B0).to(device)
    final_model = train(model, train_loader, TE_tensor)

    # (4) Test 환자 데이터 로드
    R2_test, BVf_test, Signal_test, ID_test, TE_tensor = load_all_patients(test_patients, TE)
    ID_test_tensor = torch.tensor(le.transform(ID_test), dtype=torch.long)

    # 여기도 동일하게 수정됨: ID_test → ID_test_tensor
    test_dataset = TensorDataset(R2_test, BVf_test, Signal_test, ID_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Test 실행 및 환자별 결과 분리 저장
    results_by_patient = test(final_model, test_loader, TE_tensor, ID_test_tensor, le)
    print("Test is completed")

    # 결과 저장 (피험자별로)
    save_dir = "./test_results"
    os.makedirs(save_dir, exist_ok=True)

    for pid, data in results_by_patient.items():
        signal_array = np.stack(data["signal"], axis=0)
        so2_array = np.stack(data["so2"], axis=0)
        cteft_array = np.stack(data["cteft"], axis=0)

        np.save(os.path.join(save_dir, f"{pid}_pred_signal.npy"), signal_array)
        np.save(os.path.join(save_dir, f"{pid}_so2_pred.npy"), so2_array)
        np.save(os.path.join(save_dir, f"{pid}_cteft_pred.npy"), cteft_array)

    print(f"Saved individual prediction results to {save_dir}")

    # SO2 visualize (예시 환자 하나로 시각화)
    main_shape = (512, 512, 3)
    example_pid = list(results_by_patient.keys())[0]
    example_so2 = np.stack(results_by_patient[example_pid]["so2"], axis=0)
    visualize_so2_slices(example_so2, main_shape, slice_idx=[10, 24, 36])
