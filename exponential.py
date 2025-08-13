import numpy as np
from scipy.optimize import curve_fit
from skimage.transform import resize
from tqdm import tqdm
import nibabel as nib
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes
from scipy.ndimage import gaussian_filter
from skimage import morphology
import tqdm

## constant 
B0=3.0
delta_chi0=0.264e-6
Hct=0.34
gamma=2.675e8


def compute_so2_map(sig, R2_map, BVf_map, TE):

    signal = sig[:,:,:,0:7]

    H, W, S, E = signal.shape
    TE_sec = np.array(TE) / 1000  # convert to seconds

    so2_map = np.full((H, W, S), dtype=np.float32)
    cteFt_map = np.full((H, W, S), dtype=np.float32)

    const = (gamma * (4 / 3) * np.pi * delta_chi0 * Hct * B0)

    for z in tqdm(range(S), desc="Slice"):
        # Resize R2* and BVf to match MEGRE resolution (if needed)
        R2_resized = resize(R2_map[:, :, z], (H, W), preserve_range=True)
        BVf_resized = resize(BVf_map[:, :], (H, W), preserve_range=True)

        for i in range(H):
            for j in range(W):
                signal = MEgre_data[i, j, z, :]
                if signal[0] > 30 and np.all(np.isfinite(signal)) and np.isfinite(R2_resized[i, j]) and np.isfinite(BVf_resized[i, j]):
                    x_data = np.stack([
                        R2_resized[i, j] * np.ones(E),
                        TE_sec,
                        BVf_resized[i, j] * np.ones(E)
                    ], axis=1)

                    def model_func(x, C, so2):
                        r2_term = -x[:, 0] * x[:, 1]
                        susc_term = -x[:, 2] * const * (1 - so2) * x[:, 1]
                        return C * np.exp(r2_term + susc_term)

                    try:
                        popt, _ = curve_fit(model_func, x_data, signal, p0=(1000, 0.9), bounds=([0, 0], [1e4, 1]))
                        cteFt_map[i, j, z], so2_map[i, j, z] = popt
                    except:
                        continue

    return so2_map, cteFt_map

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

def data_load(R2_dir, Bvf_dir, signal_dir):
    R2_map = nib.load(R2_dir).get_fdata() # (512,512,3)
    Bvf_map = nib.load(Bvf_dir).get_fdata() 
    signal = nib.load(signal_dir).get_fdata() # (512,512,45,16)

    brain_mask = create_mask(signal, slice_idx=[10, 24, 36], erosion_iterations = 2, dilation_iterations = 2)

    R2_masked = np.where(brain_mask, R2_map, np.nan)
    Bvf_masked = np.where(brain_mask, Bvf_map, np.nan)
    signal_masked = np.where(brain_mask[...,None], signal, np.nan) # (512,512,3,16)

    return R2_masked, Bvf_masked, signal_masked

## So2 visualizes


if __name__ == "__main__"
    TE = np.array([2,12,22,32,42,52,62]) # ms
    R2_dir = ""
    Bvf_dir = ""
    signal_dir = ""

    R2_masked, Bvf_masked, signal_masked = data_load(R2_dir, Bvf_dir, signal_dir)
    so2_map, cteFt_map = compute_so2_map(signal_masked, TE, R2_masked, Bvf_masked, B0=3.0, delta_chi0=0.264e-6, Hct=0.34, gamma=2.675e8)
