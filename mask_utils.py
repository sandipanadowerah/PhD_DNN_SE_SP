import numpy as np
import librosa as lb
import scipy.linalg
import sys


eps = sys.float_info.epsilon
eta = 1e6


def intern_filter(Rxx, Rnn, mu=1, type='r1-mwf', rank='Full'):
    """
    Computes a filter according to references [1] (SDW-MWF) or [2] (GEVD-MWF).

    :param Rxx: Speech covariance matrix
    :param Rnn: Noise covariance matrix
    :param mu: Speech distortion constant [default 1]
    :param type: (string) Type of filter to compute (SDW-MWF (see [1], equation (4)) or GEVD (see [2])). [default 'mwf']
    :param rank: ('Full' or 1) Rank-1 approximation of Rxx ? (see [2]). [default 'Full']
    :return:    - Wint: Filter coefficients
                - t1: (for GEVD case) vector selecting signals in GEVD fashion, to get correct reference signal
    """
    t1 = np.vstack((1, np.zeros((np.shape(Rxx)[0] - 1, 1))))    # Default is e1, selecting first column
    sort_index = None
    if type == 'r1-mwf':
        # # ----- Make Rxx rank 1  -----
        D, X = np.linalg.eig(Rxx)
        D = np.real(D)  # Must be real (imaginary part due to numerical noise)
        Dmax, maxind = D.max(), D.argmax()  # Find maximal eigenvalue
        Rxx = np.outer(np.abs(Dmax) * X[:, maxind],
                       np.conjugate(X[:, maxind]).T)  # Rxx is assumed to be rank 1
        # -----------------------------
        P = np.linalg.lstsq(Rnn, Rxx, rcond=None)[0]
        Wint = 1 / (mu + np.trace(P)) * P[:, 0]  # Rank1-SDWMWF (see [1])

    elif type == 'gevd':
        # TODO: inquire wether scipy.linalg.eig is much slower than scipy.linag.eigh
        D, Q = scipy.linalg.eig(Rxx, Rnn)               # GEV decomposition of Rnn^(-1)*Ryy
        D = np.maximum(D,
                       eps * np.ones(np.shape(D)))      # Prevent negative eigenvalues
        D = np.minimum(D,
                       eta * np.ones(np.shape(D)))      # Prevent infinite eigenvalues
        sort_index = np.argsort(D)                      # Sort the array to put GEV in descending order in the diagonal
        D = np.diag(D[sort_index[::-1]])                # Diagonal matrix of descending-order sorted GEV
        Q = Q[:, sort_index[::-1]]                      # Sorted matrix of generalized eigenvectors
        if rank != 'Full':                              # Rank-1 matrix of GEVD;
            D[rank:, :] = 0                             # Force zero values for all GEV but the highest
        # Filter
        Wint = np.matmul(Q,
                         np.matmul(D,
                                   np.matmul(np.linalg.inv(D + mu * np.eye(len(D))),
                                             np.linalg.inv(Q))))[:, 0]
        t1 = Q[:, 0] * np.linalg.inv(Q)[0, 0]
    elif type == 'basic':
        P = np.linalg.lstsq(Rnn + Rxx, Rxx, rcond=None)[0]
        Wint = P[:, 0]

    else:
        raise AttributeError('Unknown filter reference')

    return Wint, (t1, sort_index)


def spatial_correlation_matrix(Rxx, x, lambda_cor=0.95, M=None):
    """
    Return spatial correlation matrix computed as exponentially smoothing of :
            - if M is None: x*x.T
                            so Rxx = lambda * Rxx + (1 - lambda)x*x.T
              x should then be an estimation of the signal of which one wants the Rxx

            - if M is not None: M*x*x.T
              x is then the mixture
    :param Rxx:             Previous estimation of Rxx
    :param x:               Signal (estimation of noise/speech if M is none; mixture otherwise)
    :param lambda_cor:      Smoothing parameter
    :param M:               Mask. If None, x is the estimation of the signal of which one wants the Rxx.
    :return: Rxx            Current eximation of Rxx
    """
    if M is None:
        Rxx = lambda_cor * Rxx + (1 - lambda_cor) * np.outer(x, np.conjugate(x).T)
    else:
        Rxx = lambda_cor * Rxx + M * (1 - lambda_cor) * np.outer(x, np.conjugate(x).T)
    return Rxx



def truncated_eye(N, j, k=0):
    """
    Create a NxN matrix with k consecutive ones in the diagonal.
    :param N:   (int) Dimension of output matrix
    :param j:   (int) Number of ones in the diagonal
    :param k:   (int) Diagonal in question (k>0 shifts the diagonal to a sub-diagonal)
    :return: A truncated eye matrix
    """
    v1 = np.ones((j, ))
    v0 = np.zeros((N - j, ))

    return np.diag(np.concatenate((v1, v0), axis=0), k=k)






def wiener_mask(x, n, power=2):
    """Returns the ideal wiener mask.

    Arguments:
        - x:        speech spectrogram (real values)
        - n:        noise spectrogram (real values; same size as x)
        - power:    power of SNR in gain computation [2]
    Output:
        - wm: wiener mask values between 0 and 1
    """
    xi = (x / n) ** power
    wf = xi / (1 + xi)
    return wf


def masking_old(y, s, n, m, win_len=512, win_hop=256):
    y_stft = lb.core.stft(y, n_fft=win_len, hop_length=win_hop, center=True)
    s_stft = lb.core.stft(s, n_fft=win_len, hop_length=win_hop, center=True)
    n_stft = lb.core.stft(n, n_fft=win_len, hop_length=win_hop, center=True)

    m = np.pad(m, ((0, 0), (1, 1)), 'reflect')
    y_m = m*y_stft
    s_m = m*s_stft
    n_m = m*n_stft

    y_f = lb.core.istft(y_m, hop_length=win_hop, win_length=win_len, center=True, length=len(y))
    s_f = lb.core.istft(s_m, hop_length=win_hop, win_length=win_len, center=True, length=len(s))
    n_f = lb.core.istft(n_m, hop_length=win_hop, win_length=win_len, center=True, length=len(n))

    return y_f, s_f, n_f

def masking(y, m, win_len=512, win_hop=256):
    y_stft = lb.core.stft(y, n_fft=win_len, hop_length=win_hop, center=True)

    m = np.pad(m, ((0, 0), (1, 1)), 'reflect')
    y_m = m*y_stft

    y_f = lb.core.istft(y_m, hop_length=win_hop, win_length=win_len, center=True, length=len(y))

    return y_f, s_f, n_f


def multichannel_weiner_filter_previous(y, ms, win_len, win_hop, mu, lambda_cor):
    # Input data parameters
    n_freq = int(win_len / 2 + 1)
    n_frames = int(1 + np.floor((len(y) - win_len) / win_hop))
    n_ch = y.shape[1]
    # Initialize variables
    y_stft = np.zeros((n_freq, n_frames, n_ch), 'complex')
    s_stft_hat = np.zeros((n_freq, n_frames, n_ch), 'complex')
    n_stft_hat = np.zeros((n_freq, n_frames, n_ch), 'complex')
    Rxx = np.zeros((n_freq, n_ch, n_ch), 'complex')
    Rnn = np.zeros((n_freq, n_ch, n_ch), 'complex')
    w = np.zeros((n_freq, n_frames, n_ch), 'complex')
    y_filt = np.zeros((n_freq, n_frames, n_ch), 'complex')
    y_out = np.zeros(y.shape)
    #print(y_stft.shape, ms.shape)
    
    # multichannel stft calculation, y_tf, s_tf, n_tf
    for i_ch in range(n_ch):
        y_stft[:, :, i_ch] = lb.core.stft(np.ascontiguousarray(y[:, i_ch]), n_fft=win_len, hop_length=win_hop, center=False)
        # Input estimation of signal and noise as s^ and n^
        s_stft_hat[:, :, i_ch] = ms[i_ch] * y_stft[:, :, i_ch]
        n_stft_hat[:, :, i_ch] = (1 - ms[i_ch]) * y_stft[:, :, i_ch]
    
    for i_frame in np.arange(n_frames):
        for i_freq in np.arange(n_freq):
            lambda_cor_ = np.minimum(lambda_cor, 1 - 1 / (i_frame + 1))
            Rxx[i_freq, :, :] = spatial_correlation_matrix(Rxx[i_freq, :, :], s_stft_hat[i_freq, i_frame, :],
                                                               lambda_cor=lambda_cor_, M=None)
            Rnn[i_freq, :, :] = spatial_correlation_matrix(Rnn[i_freq, :, :], n_stft_hat[i_freq, i_frame, :],
                                                               lambda_cor=lambda_cor_, M=None)

            try:
                w[i_freq, i_frame, :], _ = intern_filter(Rxx[i_freq, :, :], Rnn[i_freq, :, :],
                                                             mu=mu, type='gevd', rank=1)
            except np.linalg.linalg.LinAlgError:
                pass

            y_filt[i_freq, i_frame, :] = np.matmul(np.conjugate(w[i_freq, i_frame, :]), y_stft[i_freq, i_frame, :])


    for i_ch in range(n_ch):
        y_out[:, i_ch] = lb.core.istft(np.pad(y_filt[:, :, i_ch], ((0, 0), (1, 1)), 'reflect'),
                                           hop_length=win_hop, win_length=win_len, center=True, length=len(y))
    
    return y_out


def multichannel_weiner_filter_current(y, s, n, m, win_len=512, win_hop=256, compute_mask=None):
    """
    Batch Multichannel Wiener filter, i.e. the covariance matrices are computed from the whole signal.
    Inputs are signal arrays, with one column per channel
    :param y:               Array of mixture signals
    :param s:
    :param n:
    :param m:               Masks list. One per signal
    :param win_len:
    :param win_hop:
    :param recompute_mask:  Whether to recompute mask as wiener(S, N). If false, input is kep [False]
    :return:
    """
    # Input data parameters
    STFT_MIN = 1e-6
    STFT_MAX = 1e3
    n_freq = int(win_len / 2 + 1)
    n_frames = int(1 + np.floor((len(y) - win_len) / win_hop))
    n_ch = y.shape[1]
    # Initialize variables
    y_stft = np.zeros((n_freq, n_frames, n_ch), 'complex')
    s_stft = np.zeros((n_freq, n_frames, n_ch), 'complex')
    n_stft = np.zeros((n_freq, n_frames, n_ch), 'complex')
    s_stft_hat = np.zeros((n_freq, n_frames, n_ch), 'complex')
    n_stft_hat = np.zeros((n_freq, n_frames, n_ch), 'complex')
    r_ss = np.zeros((n_freq, n_ch, n_ch), 'complex')
    r_nn = np.zeros((n_freq, n_ch, n_ch), 'complex')
    w = np.zeros((n_freq, n_frames, n_ch), 'complex')
    y_filt = np.zeros((n_freq, n_frames, n_ch), 'complex')
    s_filt = np.zeros((n_freq, n_frames, n_ch), 'complex')
    n_filt = np.zeros((n_freq, n_frames, n_ch), 'complex')
    y_out = np.zeros(y.shape)
    s_out = np.zeros(y.shape)
    n_out = np.zeros(y.shape)
    oracle_signal = np.zeros(y.shape)

    for i_ch in range(n_ch):
        y_stft[:, :, i_ch] = lb.core.stft(y[:, i_ch], n_fft=win_len, hop_length=win_hop, center=False)
        s_stft[:, :, i_ch] = lb.core.stft(s[:, i_ch], n_fft=win_len, hop_length=win_hop, center=False)
        n_stft[:, :, i_ch] = lb.core.stft(n[:, i_ch], n_fft=win_len, hop_length=win_hop, center=False)
        
        y_stft[:, :, i_ch] = np.clip(y_stft[:, :, i_ch], STFT_MIN, STFT_MAX)
        s_stft[:, :, i_ch] = np.clip(s_stft[:, :, i_ch], STFT_MIN, STFT_MAX)
        n_stft[:, :, i_ch] = np.clip(n_stft[:, :, i_ch], STFT_MIN, STFT_MAX)

        # Input estimation
        if compute_mask == 'wiener_mask':
            m[i_ch] = wiener_mask(abs(s_stft[:, :, i_ch]), abs(n_stft[:, :, i_ch]), power=1)
        elif compute_mask == 'oracle_mask':
            m[i_ch] = ideal_binary_mask(abs(s_stft[:, :, i_ch]), abs(n_stft[:, :, i_ch]))
        elif compute_mask == 'oracle_signal':
            oracle_signal = vad_oracle_batch(y[:, i_ch])
            
        s_stft_hat[:, :, i_ch] = m[i_ch] * y_stft[:, :, i_ch]
        n_stft_hat[:, :, i_ch] = (1 - m[i_ch]) * y_stft[:, :, i_ch]

    # Compute Rx, Rn with mask
    for f in range(n_freq):
        phi_s_f = [[] for it in range(n_frames)]  # Covariance matrix at every frame
        phi_n_f = [[] for it in range(n_frames)]  # Covariance matrix at every frame
        
        for t in range(n_frames):
            phi_s_f[t] = np.outer(s_stft_hat[f, t, :], np.conjugate(s_stft_hat[f, t, :]).T)
            phi_n_f[t] = np.outer(n_stft_hat[f, t, :], np.conjugate(n_stft_hat[f, t, :]).T)

        
        # computing global Rss and Rnn
        r_ss[f, :, :] = np.mean(np.array(phi_s_f), axis=0)
        r_nn[f, :, :] = np.mean(np.array(phi_n_f), axis=0)
        
        # filter estimation for each f
        w[f] = np.linalg.lstsq(r_nn[f, :, :] + r_ss[f, :, :], r_ss[f, :, :], rcond=None)[0][:, 0]
        
        # filtering
        for i_frame in range(n_frames):
            y_filt[f, i_frame, :] = np.matmul(np.conjugate(w[f, i_frame, :]), y_stft[f, i_frame, :])
            s_filt[f, i_frame, :] = np.matmul(np.conjugate(w[f, i_frame, :]), s_stft[f, i_frame, :])
            n_filt[f, i_frame, :] = np.matmul(np.conjugate(w[f, i_frame, :]), n_stft[f, i_frame, :])

    for i_ch in range(n_ch):
        y_out[:, i_ch] = lb.core.istft(np.pad(y_filt[:, :, i_ch], ((0, 0), (1, 1)), 'reflect'),
                                       hop_length=win_hop, win_length=win_len, center=True, length=len(y))
        s_out[:, i_ch] = lb.core.istft(np.pad(s_filt[:, :, i_ch], ((0, 0), (1, 1)), 'reflect'),
                                       hop_length=win_hop, win_length=win_len, center=True, length=len(y))
        n_out[:, i_ch] = lb.core.istft(np.pad(n_filt[:, :, i_ch], ((0, 0), (1, 1)), 'reflect'),
                                       hop_length=win_hop, win_length=win_len, center=True, length=len(y))

    return y_out	
