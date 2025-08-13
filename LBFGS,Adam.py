import torch
import torch.nn as nn
import torch.optim as optim
from skimage.transform import resize
import numpy as np
import tqdm
import nibabel as nib
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes
from scipy.ndimage import gaussian_filter
from skimage import morphology
import matplotlib.pyplot as plt
import time


# constant
B0=3.0
delta_chi0=0.264e-6
Hct=0.34
gamma=2.675e8
max_iter = 500 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

## LBFGS or Adam
def compute_so2_map(signal, R2_map, Bvf_map, TE):

    H, W ,S, E = signal.shape #(512,512,3,7)
    TE = torch.tensor(TE, dtype=torch.float32).to(device) / 1000 # second

    const = gamma * (4 / 3) * np.pi * delta_chi0 * Hct * B0

    so2_map = torch.zeros((H, W, S), dtype=torch.float32)
    cteFt_map = torch.zeros((H, W, S), dtype=torch.float32)

    criterion = nn.MSELoss()

    start_time = time.time()

    for z in tqdm(range(S), desc = "slices"):
        signal_slice = signal[:,:,z,0:7]
        R2_slice = R2_map[:,:,z]
        Bvf_slice = Bvf_map[:,:,z]
        mask_slice = mask[:,:,z]

        indices = np.where(mask_slice)
        i_list, j_list = indices

        for idx in range(len(i_list)):
            i = i_list[idx]
            j = j_list[idx]

            signal_val = torch.tensor(signal_slice[i, j, :], dtype=torch.float32, device=device)
            R2_val = torch.tensor(R2_slice[i, j], dtype=torch.float32, device=device)
            BVf_val = torch.tensor(Bvf_slice[i, j], dtype=torch.float32, device=device)

            # 파라미터 초기화
            cteFt = torch.tensor([1000.0], dtype=torch.float32, requires_grad=True, device=device)
            so2 = torch.tensor([0.9], dtype=torch.float32, requires_grad=True, device=device)
            optimizer = optim.LBFGS([cteFt, so2], lr=1e-5, max_iter=max_iter, line_search_fn="strong_wolfe") #strong_wolfe -> step마다 lr수정

            def closure():
                optimizer.zero_grad()
                pred_s = cteFt * torch.exp(-R2_val * TE_sec - BVf_val * const * (1 - so2) * TE_sec)
                loss = criterion(pred_s, signal_val)
                loss.backward()
                return loss

            optimizer.step(closure)

            # 값 저장
            cteFt_map[i, j, z] = cteFt.detach()
            so2_map[i, j, z] = so2.detach()

    end_time = time.time()
    print(f"fitting time: {end_time - start_time:.4f}초")

    return so2_map.numpy(), cteFt_map.numpy()

## masking image
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

## data load
def load_data(r2_dir, bvf_dir, signal_dir, slice_idx = [10,24,36]):
    R2_map = nib.load(r2_dir).get_fdata() #(512,512,3)
    Bvf_map = nib.load(bvf_dir).get_fdata()
    signal = nib.load(signal_dir).get_fdata() #(512,512,45,16)
    signal = signal[:, :, slice_idx, 0:7] 

    brain_mask = create_mask(signal, slice_idx=None, erosion_iterations = 2, dilation_iterations = 2)

    R2_masked = np.where(brain_mask, R2_map, np.nan)
    Bvf_masked = np.where(brain_mask, Bvf_map, np.nan)
    signal_masked = np.where(brain_mask[...,None], signal, np.nan) # (512,512,3,16)

    return R2_masked, Bvf_masked, signal_masked

## So2 visualize
def So2_visualize(so2_map):

    # H, W, S = so2_map.shape

    if so2_map.shape[2] != 3:
        raise ValueError(f"slice 수가 {so2_map.shape[0]}개 입니다.")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx in range(3):
        axes[idx].imshow(so2_map[:, :, idx], cmap="hot", vmin=0, vmax=1)
        axes[idx].set_title(f"Slice {idx}")
        axes[idx].axis('off')
    fig.colorbar(axes[-1].imshow(so2_map[:, :, 0], cmap="hot", vmin=0, vmax=1),
                 ax=axes, location='right')
    plt.suptitle("SO2 Maps")
    plt.show()


## main 실행
if __name__ == "__main__":
    TE = np.array([2,12,22,32,42,52,62]) # ms
    r2_dir = r"C:\Users\user\Desktop\assistant\my_project\processed_data\test_data\07511225YDS\07511225YDS_R2_map.nii"
    bvf_dir = r"C:\Users\user\Desktop\assistant\my_project\processed_data\test_data\07511225YDS\07511225YDS_Bvf_map.nii"
    signal_dir = r"C:\Users\user\Desktop\assistant\my_project\processed_data\test_data\07511225YDS\07511225_YDS_VSI15_WIP_iadVSI05b_pre7meGRE_9_1.nii"
    R2_map, Bvf_map, signal, mask = load_data(r2_dir, bvf_dir, signal_dir, slice_idx=[10,24,36])

    so2_map, cteFt_map = compute_so2_map(signal, TE, R2_map, Bvf_map, mask)
    So2_visualize(so2_map)