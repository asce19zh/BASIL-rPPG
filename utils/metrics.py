import torch
import numpy as np
from scipy.signal import butter, filtfilt

def compute_power_spectrum(signal, Fs, zero_pad=None):
    if isinstance(signal, torch.Tensor):
        signal = signal.numpy()

    if zero_pad is not None:
        L = signal.shape[-1]
        pad_len = int(zero_pad / 2 * L)
        signal = np.pad(signal, ((0, 0), (pad_len, pad_len)), mode='constant')

    freqs = np.fft.fftfreq(signal.shape[-1], 1 / Fs) * 60  # in bpm
    ps = np.abs(np.fft.fft(signal, axis=-1)) ** 2
    cutoff = len(freqs) // 2
    return freqs[:cutoff], ps[:, :cutoff]

def predict_heart_rate(signal, Fs=30, min_hr=40., max_hr=180.):
    if isinstance(signal, torch.Tensor):
        signal = signal.numpy()

    signal -= np.mean(signal, axis=-1, keepdims=True)
    freqs, ps = compute_power_spectrum(signal, Fs, zero_pad=100)

    mask = (freqs >= min_hr) & (freqs <= max_hr)
    freqs = freqs[mask]
    ps = ps[:, mask]

    max_ind = np.argmax(ps, axis=-1)
    max_bpm = np.zeros(signal.shape[0])

    for i in range(signal.shape[0]):
        if 0 < max_ind[i] < len(freqs) - 1:
            inds = max_ind[i] + np.array([-1, 0, 1])
            x = ps[i][inds]
            f = freqs[inds]
            d1 = x[1] - x[0]
            d2 = x[1] - x[2]
            offset = (1 - min(d1, d2) / max(d1, d2)) * (f[1] - f[0])
            if d2 > d1:
                offset *= -1
            max_bpm[i] = f[1] + offset
        elif max_ind[i] == 0:
            max_bpm[i] = freqs[0]
        elif max_ind[i] == len(freqs) - 1:
            max_bpm[i] = freqs[-1]

    return max_bpm


def butter_bandpass(sig_list, lowcut, highcut, fs, order=2):
    # butterworth bandpass filter (batch version)
    # signals are in the sig_list

    y_list = []

    for sig in sig_list:
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, sig)
        y_list.append(y)
    return np.array(y_list)


class HeartRateEvaluator:
    def __init__(self, Fs=30, min_hr=40., max_hr=250.):
        self.Fs = Fs
        self.min_hr = min_hr
        self.max_hr = max_hr

    def __call__(self, pred_signal, gt_siganl):

        if isinstance(pred_signal, torch.Tensor):
            pred_signal = pred_signal.detach().cpu().numpy()
        if isinstance(gt_siganl, torch.Tensor):
            gt_siganl = gt_siganl.detach().cpu().numpy()

        pred_signal = butter_bandpass(pred_signal, self.min_hr / 60, self.max_hr / 60, self.Fs)
        gt_siganl = butter_bandpass(gt_siganl, self.min_hr / 60, self.max_hr / 60, self.Fs)

        pred_hr = predict_heart_rate(pred_signal, self.Fs, self.min_hr, self.max_hr)
        gt_hr = predict_heart_rate(gt_siganl, self.Fs, self.min_hr, self.max_hr)

        results = {}
        results['MAE'] = np.mean(np.abs(pred_hr - gt_hr))
        results['MSE'] = np.mean((pred_hr - gt_hr) ** 2)
        results['RMSE'] = np.sqrt(results['MSE'])

        ## Pearson correlation
        results['R'] = np.corrcoef(pred_hr, gt_hr)[0, 1]
        return pred_hr, gt_hr, results

