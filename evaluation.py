import torch
from utils.mog_utils import BatchGMM40

def eval_log_prob(samples, mean_scale, std):
    """
    Evaluate the average log-probability of samples under the batched GMM.
    
    Args:
        samples (Tensor): Tensor of shape [B, N, 2], where B is batch size, N is number of samples.
        mean_scale (Tensor): Tensor of shape [B, 1] or scalar, used to scale GMM means.
        std (Tensor): Tensor of shape [B, 1] or scalar, standard deviation of GMM components.
        
    Returns:
        avg_log_prob (Tensor): Scalar, average log probability across all batches and samples.
    """    
    # Create batched GMM
    gmm = BatchGMM40(mean_scale=mean_scale, std=std, device=samples.device)
    
    # Compute log-probabilities of samples
    log_probs = gmm.log_prob(samples)  # Shape: [B, N]
    
    # Average over samples
    avg_log_prob = log_probs.sum(dim=-1)  # Shape: [B]
    
    return avg_log_prob.mean()


def gaussian_kernel(x, y, sigma=1.0):
    x_norm = (x ** 2).sum(dim=1).view(-1, 1)
    y_norm = (y ** 2).sum(dim=1).view(1, -1)
    dist_sq = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
    kernel = torch.exp(-dist_sq / (2 * sigma ** 2))
    return kernel


def eval_mmd(x, y, sigma=1.0):
    m, n = x.size(0), y.size(0)
    K_xx = gaussian_kernel(x, x, sigma)
    K_yy = gaussian_kernel(y, y, sigma)
    K_xy = gaussian_kernel(x, y, sigma)
    mmd = K_xx.sum() / (m * m) + K_yy.sum() / (n * n) - 2 * K_xy.sum() / (m * n)
    return mmd


def eval_mean_mmd_bandwidths(x, y, sigma_list):
    """
    Computes mean MMD for multiple kernel bandwidths.

    Args:
        x (Tensor): Tensor of shape (n_samples, 2).
        y (Tensor): Tensor of shape (n_samples, 2).
        sigma_list (list or tensor): List of kernel bandwidths.

    Returns:
        float: The mean MMD value across all bandwidths.
    """
    mmd_values = []

    for sigma in sigma_list:
        mmd = eval_mmd(x, y, sigma)
        mmd_values.append(mmd)

    mean_mmd = torch.mean(torch.tensor(mmd_values))
    return mean_mmd.item()
