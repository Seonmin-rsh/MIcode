import numpy as np
from scipy.optimize import curve_fit
from skimage.transform import resize
from tqdm import tqdm
import nibabel as nib
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes
from scipy.ndimage import gaussian_filter
from skimage import morphology

# -------------------------------
# Constants
# -------------------------------
B0 = 3.0
delta_chi0 = 0.264e-6
Hct = 0.34
gamma = 2.675e8


# -------------------------------
# Masking (그대로 사용하되 약간 보강)
# -------------------------------
def create_mask(original_signal, slice_idx=[10, 24, 36], erosion_iterations=2, dilation_iterations=2):
    # original_signal: 4D (H,W,S,E) 기대. 3D면 E축 추가
    if original_signal.ndim == 3:
        original_signal = original_signal[..., np.newaxis]  # (H,W,S,1)

    # 관심 슬라이스 + 0:7 Echo만 사용 (시각/노이즈 안정)
    sig = original_signal[:, :, slice_idx, 0:7]  # (H,W,len(slice_idx),7)

    image = np.mean(sig, axis=3)  # (H,W,Z)
    H, W, Z = image.shape
    brain_mask = np.zeros((H, W, Z), dtype=bool)

    for z in range(Z):
        slc = image[:, :, z]
        if slc.max() > slc.min():
            slc = (slc - slc.min()) / (slc.max() - slc.min())
        # smoothing 후 Otsu
        smoothed = gaussian_filter(slc, sigma=1)
        thr = threshold_otsu(smoothed)
        m = smoothed > thr

        # post-processing
        m = binary_erosion(m, iterations=erosion_iterations)
        m = binary_fill_holes(m)
        m = binary_dilation(m, iterations=dilation_iterations)
        m = keep_largest_component(m)

        brain_mask[:, :, z] = m

    return brain_mask  # (H,W,Z)

def keep_largest_component(binary_mask):
    labeled = morphology.label(binary_mask)
    if labeled.max() == 0:
        return binary_mask
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest = counts.argmax()
    return (labeled == largest)

# -------------------------------
# Data Load (+ masking 적용)
# -------------------------------
def data_load(R2_dir, Bvf_dir, signal_dir, slice_idx=[10, 24, 36]):
    R2_map = nib.load(R2_dir).get_fdata().astype(np.float32)        # 예상: (H,W,Z)
    BVf_map = nib.load(Bvf_dir).get_fdata().astype(np.float32)      # 예상: (H,W,Z) 혹은 (H,W)
    signal  = nib.load(signal_dir).get_fdata().astype(np.float32)   # (H,W,S,E)

    # 관심 슬라이스만 추출(신호만), R2/BVf는 아래에서 마스크 시점에 크기 맞춰줌
    sig_sel = signal[:, :, slice_idx, :]   # (H,W,Z,E)
    mask = create_mask(signal, slice_idx=slice_idx, erosion_iterations=2, dilation_iterations=2)  # (H,W,Z)

    # 마스크 적용 (마스크 밖은 NaN)
    R2_masked = np.where(mask, R2_map, np.nan)                # (H,W,Z)
    BVf_masked = np.where(mask, BVf_map, np.nan)              # (H,W,Z)
    signal_masked = np.where(mask[..., None], sig_sel, np.nan)  # (H,W,Z,E)

    return R2_masked, BVf_masked, signal_masked

# -------------------------------
# Fitting (마스크 영역만, slice 루프)
# -------------------------------
def compute_so2_map(signal_masked, R2_masked, BVf_masked, TE_ms, intensity_threshold=30.0):
    """
    signal_masked: (H,W,Z,E) 마스크 밖은 NaN
    R2_masked:     (H,W,Z)   [1/s]
    BVf_masked:    (H,W,Z)   [unitless]
    TE_ms:         list/array of Echo times [ms]
    """
    # Echo 0~6만 사용할 경우, 바깥에서 이미 슬라이싱해둔 상태라고 가정 (signal_masked[..., 0:7])
    H, W, Z, E = signal_masked.shape
    TE_sec = np.asarray(TE_ms, dtype=np.float64) / 1000.0
    if len(TE_sec) != E:
        raise ValueError(f"TE 길이({len(TE_sec)})와 echo 수({E})가 다릅니다.")

    # 출력 초기화 (NaN으로 채움)
    so2_map = np.full((H, W, Z), np.nan, dtype=np.float32)
    cteFt_map = np.full((H, W, Z), np.nan, dtype=np.float32)

    # 상수 (감마 등)
    CONST = (gamma * (4.0/3.0) * np.pi * delta_chi0 * Hct * B0)

    # 슬라이스 루프
    for z in tqdm(range(Z), desc="Slice"):
        # 현재 슬라이스 2D
        R2_z  = R2_masked[:, :, z]
        BVf_z = BVf_masked[:, :, z]
        Sig_z = signal_masked[:, :, z, :]  # (H,W,E)

        # 픽셀 루프
        for i in range(H):
            r2_row = R2_z[i]
            bv_row = BVf_z[i]
            sig_row = Sig_z[i]  # (W,E)

            for j in range(W):
                vox = sig_row[j].astype(np.float64)  # (E,)
                r2  = r2_row[j]
                bv  = bv_row[j]

                # 마스크 밖(NaN) 또는 유효하지 않은 경우 스킵
                if (not np.isfinite(r2)) or (not np.isfinite(bv)):
                    continue
                if (not np.all(np.isfinite(vox))):
                    continue
                if vox[0] <= intensity_threshold:
                    continue

                # x_data 구성: [R2, TE, BVf]
                x_data = np.stack([
                    np.full(E, r2, dtype=np.float64),
                    TE_sec,
                    np.full(E, bv, dtype=np.float64)
                ], axis=1)  # (E,3)

                # 모델: C * exp( -R2*TE - BVf*CONST*(1 - so2)*TE )
                def model_func(x, C, so2):
                    r2_term   = -x[:, 0] * x[:, 1]
                    susc_term = -x[:, 2] * CONST * (1.0 - so2) * x[:, 1]
                    return C * np.exp(r2_term + susc_term)

                # 초기값/경계
                C0 = float(np.nanmax(vox))  # 첫 에코/최대값 사용
                C0 = 1.0 if not np.isfinite(C0) or C0 <= 0 else C0
                p0 = (C0, 0.9)
                bounds = ([0.0, 0.0], [1e4, 1.0])

                try:
                    popt, _ = curve_fit(model_func, x_data, vox, p0=p0, bounds=bounds, maxfev=10000)
                    C_fit, so2_fit = popt
                    cteFt_map[i, j, z] = np.float32(C_fit)
                    so2_map[i, j, z]   = np.float32(so2_fit)
                except Exception:
                    # 실패 시 NaN 유지
                    pass

    return so2_map, cteFt_map

# -------------------------------
# (선택) 시각화 도우미
# -------------------------------
def visualize_so2_slices(so2_map, slice_titles=None, vmin=0.0, vmax=1.0, cmap='hot'):
    import matplotlib.pyplot as plt
    H, W, Z = so2_map.shape
    n = Z
    fig, axes = plt.subplots(1, n, figsize=(4.5*n, 4.5))
    if n == 1:
        axes = [axes]
    for k in range(n):
        im = axes[k].imshow(so2_map[:, :, k], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[k].axis('off')
        axes[k].set_title(slice_titles[k] if slice_titles else f"Slice {k}")
    fig.colorbar(im, ax=axes, orientation='vertical', shrink=0.75)
    plt.tight_layout()
    plt.show()

# -------------------------------
# Example main
# -------------------------------
if __name__ == "__main__":
    # 예시 세팅
    TE_ms = np.array([2, 12, 22, 32, 42, 52, 62], dtype=np.float32)  # ms
    slice_idx = [10, 24, 36]

    R2_dir = r"...\R2_map.nii.gz"
    Bvf_dir = r"...\BVf_map.nii.gz"
    signal_dir = r"...\pre7meGRE.nii.gz"   # (H,W,S,E=16) 같은 형태

    # 로드 + 마스킹 (관심 슬라이스만)
    R2_masked, BVf_masked, signal_masked = data_load(
        R2_dir, Bvf_dir, signal_dir, slice_idx=slice_idx
    )

    # Echo 0~6만 사용해서 fitting (signal_masked는 전체 E를 갖고 있으므로 여기서 슬라이싱)
    signal_for_fit = signal_masked[..., 0:7]  # (H,W,Z,7)

    # Fitting
    so2_map, cteFt_map = compute_so2_map(
        signal_for_fit,
        R2_masked,
        BVf_masked,
        TE_ms,
        intensity_threshold=30.0
    )

    # (선택) 시각화
    visualize_so2_slices(so2_map, slice_titles=[str(s) for s in slice_idx])
