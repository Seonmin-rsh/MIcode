import numpy as np
from scipy.optimize import curve_fit
from skimage.transform import resize
from tqdm import tqdm

def compute_so2_map(MEgre_data, TE_values_ms, R2_map, BVf_map,
                           B0=3.0, delta_chi0=0.264e-6, Hct=0.34, gamma=2.675e8):

    H, W, S, E = MEgre_data.shape
    TE_sec = np.array(TE_values_ms) / 1000  # convert to seconds

    so2_map = np.zeros((H, W, S), dtype=np.float32)
    cteFt_map = np.zeros((H, W, S), dtype=np.float32)

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

so2_map, cteFt_map = compute_so2_map(pre7meGRE_data[:,:,44:45,0:7], TE_list_t2star, preR2_map, BVf_map,
                    B0=3.0, delta_chi0=0.264e-6, Hct=0.34, gamma=2.675e8)

