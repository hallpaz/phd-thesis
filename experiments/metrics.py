import torch
from scipy.signal import correlate
from scipy.fft import fft, fftfreq
from mrnet.logs.handler import ResultHandler


class MetricsHandler(ResultHandler):
    def log_metrics(self, gt, pred):
        super().log_metrics(gt, pred)
        self.log_maxae(gt, pred)
        self.log_correlation(gt, pred)
        self.log_spectral_distortion(gt, pred)


    def log_PSNR(self, gt, pred):
        return super().log_PSNR(gt, pred)

    def log_maxae(self, gt, pred):
        maxae = torch.max(torch.abs(gt - pred))
        label = f"Stage {self.hyper['stage']}"
        self.logger.log_metric("psnr", maxae, label)
    
    def log_correlation(self, gt, pred):
        cross_corr = correlate(gt, pred, mode='full')
        label = f"Stage {self.hyper['stage']}"
        self.logger.log_metric("correlation", cross_corr, label)

    def log_spectral_distortion(self, gt, pred):
        # Compute the Fourier Transform of both signals
        fft_signal1 = fft(gt)
        fft_signal2 = fft(pred)
        
        # Compute the L2 norm (Euclidean distance) between the two Fourier Transforms
        spectral_error = torch.linalg.norm(fft_signal1 - fft_signal2)
        label = f"Stage {self.hyper['stage']}"
        self.logger.log_metric("spectral_mse", spectral_error, label)


if __name__ == '__main__':
    import numpy as np
    from numpy.fft import fft, fftfreq

    def spectral_distortion_by_band(signal1, signal2, omega, sampling_rate):
        N = len(signal1)
        
        # Compute the Fourier Transform of both signals
        fft_signal1 = fft(signal1)
        fft_signal2 = fft(signal2)
        
        # Get the frequencies corresponding to the FFT components
        frequencies = fftfreq(N, 1 / sampling_rate)
        
        # Initialize an empty list to store spectral errors by bands
        spectral_errors_by_band = []
        
        # We will calculate distortion by bands [-omega, omega], [-2*omega, -omega] U [omega, 2*omega], etc.
        band = 0
        while band * omega < sampling_rate / 2:  # Ensure we're within the Nyquist limit
            # Define the frequency range for this band
            lower_bound1 = -(band + 1) * omega
            upper_bound1 = -band * omega
            lower_bound2 = band * omega
            upper_bound2 = (band + 1) * omega
            
            # Get the indices where the frequencies are within this band
            indices_band1 = np.where((frequencies >= lower_bound1) & (frequencies < upper_bound1))
            indices_band2 = np.where((frequencies >= lower_bound2) & (frequencies < upper_bound2))
            
            # Calculate the spectral distortion for this band
            error_band1 = np.linalg.norm(fft_signal1[indices_band1] - fft_signal2[indices_band1])
            error_band2 = np.linalg.norm(fft_signal1[indices_band2] - fft_signal2[indices_band2])
            
            # Sum the errors from both negative and positive parts of the band
            spectral_error = error_band1 + error_band2
            spectral_errors_by_band.append(spectral_error)
            
            # Move to the next band
            band += 1
        
        return spectral_errors_by_band

    # Example usage:
    sampling_rate = 1000  # in Hz, adjust as per your signals
    omega = 50  # Example band width, adjust as per your needs

    # Create some example signals (signal1 as input, signal2 as reconstruction)
    signal1 = np.random.rand(1000)
    signal2 = signal1 + np.random.normal(0, 0.1, 1000)  # Slightly noisy reconstruction

    spectral_errors = spectral_distortion_by_band(signal1, signal2, omega, sampling_rate)
    print(f'Spectral Distortion by Bands: {spectral_errors}')
