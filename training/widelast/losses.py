import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class DistillationLoss(nn.Module):

    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module, args):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert args.distillation_type in ['none', 'fft', 'soft', 'hard']
        self.distillation_type = args.distillation_type
        self.tau = args.distillation_tau

        self.layer_ids_s = args.s_id
        self.layer_ids_t = args.t_id
        self.alpha = args.distillation_alpha
        self.beta = args.distillation_beta
        self.w_fft = args.w_fft
        self.use_mse_loss = args.use_mse_loss

    def forward(self, inputs, outputs, labels):
        block_outs_s = outputs[1]
        student_outputs = outputs[0]
        base_loss = self.base_criterion(student_outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        # don't backprop through the teacher
        with torch.no_grad():
            teacher_outputs, block_outs_t = self.teacher_model(inputs, self.layer_ids_t)

        loss_base = (1-self.alpha) * base_loss

        # Choose between MSE loss or FFT loss for feature map distillation
        if self.use_mse_loss:
            loss_fft = mse_feature_loss(block_outs_s, block_outs_t)
        else:
            loss_fft = fft_loss(block_outs_s, block_outs_t)

        loss_fft = self.beta * self.w_fft * loss_fft

        if self.distillation_type == 'fft':
            return loss_base, loss_fft

        if self.distillation_type == 'soft':
            T = self.tau
            distillation_loss = F.kl_div(
                F.log_softmax(student_outputs / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='batchmean',
                log_target=True
            ) * (T * T)
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(student_outputs, teacher_outputs.argmax(dim=1))
        loss_dist = self.alpha * distillation_loss

        return loss_base, loss_fft, loss_dist


def fft_loss(block_outs_s, block_outs_t):
    losses = []
    for F_s, F_t in zip(block_outs_s, block_outs_t):
        loss_fft = layer_fft_loss(F_s, F_t)
        losses.append(loss_fft)

    loss_fft = sum(losses) / len(losses)

    return loss_fft


def layer_fft_loss(F_s, F_t):
    """
    Compute FFT-based spectral loss between student and teacher features.

    Args:
        F_s: Student features [B, C_s, H, W]
        F_t: Teacher features [B, C_t, H, W]

    Returns:
        loss_fft: MSE loss between FFT spectra

    Note:
        - If C_s < C_t: downsample teacher using adaptive pooling
        - If C_s == C_t: direct comparison (no pooling needed)
        - If C_s > C_t: Not expected in typical distillation setup
    """
    _, C_s, _, _ = F_s.shape
    _, C_t, _, _ = F_t.shape

    # Handle dimension mismatch
    if C_s < C_t:
        # Original behavior: downsample teacher
        F_t = F.adaptive_avg_pool3d(F_t, (C_s, None, None))
    elif C_s == C_t:
        # Dimensions match: no pooling needed
        # This is the case for modified student with channel expansion
        pass
    else:
        # C_s > C_t: Downsample student (unexpected but handle gracefully)
        print(f"Warning: Student channels ({C_s}) > Teacher channels ({C_t}). Downsampling student.")
        F_s = F.adaptive_avg_pool3d(F_s, (C_t, None, None))

    # Apply 2D FFT on spatial dimensions
    F_s_fft = torch.fft.rfft2(F_s.float(), norm="ortho")
    F_t_fft = torch.fft.rfft2(F_t.float(), norm="ortho")

    # Stack real and imaginary parts
    F_s_fft = torch.stack([F_s_fft.real, F_s_fft.imag], dim=-1)
    F_t_fft = torch.stack([F_t_fft.real, F_t_fft.imag], dim=-1)

    # Compute MSE loss
    loss_fft = F.mse_loss(F_s_fft, F_t_fft)

    return loss_fft


def mse_feature_loss(block_outs_s, block_outs_t):
    """
    Compute pure MSE loss between student and teacher feature maps.
    This is an alternative to FFT-based spectral loss.

    Args:
        block_outs_s: List of student features [B, C_s, H, W]
        block_outs_t: List of teacher features [B, C_t, H, W]

    Returns:
        loss_mse: Average MSE loss across all layers
    """
    losses = []
    for F_s, F_t in zip(block_outs_s, block_outs_t):
        loss_mse = layer_mse_loss(F_s, F_t)
        losses.append(loss_mse)

    loss_mse = sum(losses) / len(losses)

    return loss_mse


def layer_mse_loss(F_s, F_t):
    """
    Compute MSE loss between student and teacher features for a single layer.

    Args:
        F_s: Student features [B, C_s, H, W]
        F_t: Teacher features [B, C_t, H, W]

    Returns:
        loss_mse: MSE loss between feature maps

    Note:
        - If C_s < C_t: downsample teacher using adaptive pooling
        - If C_s == C_t: direct comparison (no pooling needed)
        - If C_s > C_t: Not expected in typical distillation setup
    """
    _, C_s, _, _ = F_s.shape
    _, C_t, _, _ = F_t.shape

    # Handle dimension mismatch
    if C_s < C_t:
        # Original behavior: downsample teacher
        F_t = F.adaptive_avg_pool3d(F_t, (C_s, None, None))
    elif C_s == C_t:
        # Dimensions match: no pooling needed
        # This is the case for modified student with channel expansion
        pass
    else:
        # C_s > C_t: Downsample student (unexpected but handle gracefully)
        print(f"Warning: Student channels ({C_s}) > Teacher channels ({C_t}). Downsampling student.")
        F_s = F.adaptive_avg_pool3d(F_s, (C_t, None, None))

    # Compute MSE loss directly on feature maps
    loss_mse = F.mse_loss(F_s, F_t)

    return loss_mse

