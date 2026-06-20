# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
Loss functions and SDE classes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models".
"""

import abc
import re
import torch as th
import numpy as np
from .sde_lib import _expand_dims
from utils import dist_util, logger
import torch.nn.functional as F

# Dictionary to hold registered SDE classes
registered_sdes = {}

def register_sde(cls=None, *, name=None):
    """Decorator to register SDE classes with a specific name."""
    if cls is None:
        # Return a decorator if the decorator is used with arguments
        return lambda cls: register_sde(cls, name=name)
    
    local_name = name or cls.__name__.lower()
    if local_name in registered_sdes:
        raise ValueError(f'Already registered SDE with name: {local_name}')
    registered_sdes[local_name] = cls
    return cls

# Abstract base class for SDEs
class BaseSDE(abc.ABC):
    def __init__(self, N):
        """Initialize the base SDE with the number of timesteps."""
        super().__init__()
        self.N = N

    # Abstract methods to be implemented by subclasses
    @abc.abstractmethod
    def sigma_sample(self, rnd):
        pass

    @abc.abstractmethod
    def sigma(self, t):
        pass

    @abc.abstractmethod
    def loss_weight(self, sigma):
        pass

    @abc.abstractmethod
    def scale_input(self, sigma):
        pass

    @abc.abstractmethod
    def scale_noise(self, sigma):
        pass

    @abc.abstractmethod
    def scale_output(self, sigma):
        pass

    @abc.abstractmethod
    def sigma_min_max(self):
        pass

    @abc.abstractmethod
    def round_sigma(self):
        pass

    @abc.abstractmethod
    def time_step(self, nd):
        pass
    
    @abc.abstractmethod
    def sigma_inv(self, sigma):
        pass
    
    @abc.abstractmethod
    def sigma_prime(self, t):
        pass
    
    @abc.abstractmethod
    def s(self, t):
        pass
    
    @abc.abstractmethod
    def s_prime(self, t):
        pass

    def noise_sample(self, batch, sigma):
        return th.randn_like(batch) * sigma
    
    def initial_sample(self, t0, shape, device, dtype):
        return th.randn(shape, device=device, dtype=dtype) * (self.sigma(t0) * self.s(t0))

# Variance Preserving SDE implementation
@register_sde(name='vpsde')
class VPSDE(BaseSDE):
    def __init__(self, beta_max=19.9, beta_min=0.1, eps=1e-5, N=1000):
        """Initialize VPSDE with beta and epsilon parameters."""
        super().__init__(N)
        self.beta_d = beta_max
        self.beta_min = beta_min
        self.eps = eps

    def sigma_sample(self, shape, device):
        """Calculate the time step based on random input."""
        rnd = th.rand(shape, device=device)
        return self.sigma(self.time_step(rnd))

    def sigma(self, t):
        """Compute the noise level based on time t."""
        t = th.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

    def loss_weight(self, sigma):
        """Weight for the loss function."""
        return 1 / sigma ** 2

    def scale_input(self, sigma):
        """Scale input for the model based on sigma."""
        return 1 / (sigma ** 2 + 1).sqrt()

    def scale_noise(self, sigma):
        """Scale the noise based on sigma."""
        return self.sigma_inv(sigma) * (self.N - 1)

    def scale_output(self, sigma):
        """Compute scaling factors for output."""
        return th.as_tensor(1.), -sigma
    
    def sigma_min_max(self):
        """Return the minimum and maximum sigma values."""
        return float(self.sigma(self.eps)), float(self.sigma(1.))
    
    def round_sigma(self, sigma):
        return th.as_tensor(sigma)
    
    def time_step(self, nd):
        return 1 + th.as_tensor(nd) * (self.eps - 1)
    
    def sigma_inv(self, sigma):
        return ((self.beta_min ** 2 + 2 * self.beta_d * (1 + sigma ** 2).log()).sqrt() - self.beta_min) / self.beta_d
    
    def sigma_prime(self, t):
        sigma = self.sigma(t)
        return 0.5 * (self.beta_min + self.beta_d * t) * (sigma + 1 / sigma)
    
    def s(self, t):
        return 1 / (1 + self.sigma(t) ** 2).sqrt()
    
    def s_prime(self, t):
        return - self.sigma(t) * self.sigma_prime(t) * (self.s(t) ** 3)
    

# Variance Exploding SDE implementation
@register_sde(name='vesde')
class VESDE(BaseSDE):
    def __init__(self, sigma_min=0.02, sigma_max=100, N=1000):
        """Initialize VESDE with minimum and maximum sigma values."""
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def sigma_sample(self, shape, device):
        """Calculate the time step based on random input."""
        rnd = th.rand(shape, device=device)
        return self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd)

    def sigma(self, t):
        """Compute the noise level based on time t."""
        return th.as_tensor(t).sqrt()

    def loss_weight(self, sigma):
        """Weight for the loss function."""
        return 1 / sigma ** 2

    def scale_input(self, sigma):
        """Scale input for the model."""
        return th.as_tensor(1.)

    def scale_noise(self, sigma):
        """Scale the noise based on sigma."""
        return (0.5 * sigma).log()

    def scale_output(self, sigma):
        """Compute scaling factors for output."""
        return th.as_tensor(1.), sigma
    
    def sigma_min_max(self):
        """Return the minimum and maximum sigma values."""
        return self.sigma_min, self.sigma_max
    
    def round_sigma(self, sigma):
        return th.as_tensor(sigma)
    
    def time_step(self, nd):
        return (self.sigma_max ** 2) * ((self.sigma_min ** 2 / self.sigma_max ** 2) ** th.as_tensor(nd))
    
    def sigma_inv(self, sigma):
        return sigma ** 2
    
    def sigma_prime(self, t):
        t = th.as_tensor(t)
        return 0.5 / t.sqrt()
    
    def s(self, t):
        return th.as_tensor(1.)
    
    def s_prime(self, t):
        return th.as_tensor(0.)
    

# EDM-specific SDE implementation
@register_sde(name='edmsde')
class EDMSDE(BaseSDE):
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_min=0.001, sigma_max=80, sigma_data=0.5, rho=7, N=1000):
        """Initialize EDMSDE with mean, std, and sigma data."""
        super().__init__(N)
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho

    def sigma_sample(self, shape, device):
        """Calculate the time step based on random input."""
        rnd = th.randn(shape, device=device)
        return (rnd * self.P_std + self.P_mean).exp()

    def sigma(self, t):
        """Compute the noise level based on time t."""
        return th.as_tensor(t)

    def loss_weight(self, sigma):
        """Weight for the loss function based on sigma."""
        return (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

    def scale_input(self, sigma):
        """Scale input for the model based on sigma."""
        return 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()

    def scale_noise(self, sigma):
        """Scale the noise based on sigma."""
        return sigma.log() / 4

    def scale_output(self, sigma):
        """Compute scaling factors for output."""
        coef_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        coef_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        return coef_skip, coef_out
    
    def sigma_min_max(self):
        """Return the minimum and maximum sigma values."""
        return self.sigma_min, self.sigma_max
    
    def round_sigma(self, sigma):
        return th.as_tensor(sigma)
    
    def time_step(self, nd):
        return (self.sigma_max ** (1 / self.rho) + nd * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
    
    def sigma_inv(self, sigma):
        return sigma
    
    def sigma_prime(self, t):
        return th.as_tensor(1.)
    
    def s(self, t):
        return th.as_tensor(1.)
    
    def s_prime(self, t):
        return th.as_tensor(0.)

@register_sde(name='pfgm')
class PFGM(EDMSDE):
    def __init__(self, D=2048, M=512**2, **kwargs):
        super().__init__(**kwargs)
        self.D = D
        self.M = M
        self.beta = th.distributions.Beta(M / 2., D / 2.)

    def radius(self, sigma):
        return sigma.double() * np.sqrt(self.D).astype(np.float64)

    def noise_sample(self, batch, sigma):
        samples_norm = self.beta.sample(sigma.shape).to(sigma.device).double()
        samples_norm = samples_norm.clamp(1e-3, 1 - 1e-3)
        inverse_beta = samples_norm / (1 - samples_norm + 1e-8)
        
        samples_norm = self.radius(sigma) * th.sqrt(inverse_beta + 1e-8)

        gaussian = th.randn_like(batch)
        gaussian_norm = th.norm(gaussian, p=2, dim=list(range(1, gaussian.ndimension())), keepdim=True)
        unit_gaussian = gaussian / gaussian_norm

        noise = unit_gaussian * samples_norm

        return noise.to(batch.dtype)
    
    def initial_sample(self, t0, shape, device, dtype):
        batch = th.zeros(shape, device=device, dtype=dtype)
        sigma_shape = [batch.shape[0]] + [1] * (batch.ndimension() - 1)
        sigma = th.ones(sigma_shape, device=device, dtype=dtype) * self.sigma_max
        return self.noise_sample(batch, sigma)
    
# EDMDiffusion class handles the diffusion loss computation
class EDMDiffusion:
    def __init__(self, sde, reduce_mean=True):
        """Initialize EDMDiffusion with an SDE and reduction method."""
        self.sde = sde
        self.reduce_mean = reduce_mean
        self.num_timesteps = sde.N
        self.reduce_op = th.mean if self.reduce_mean else lambda *args, **kwargs: 0.5 * th.sum(*args, **kwargs)

    def loss_fn(self, model, batch, model_kwargs=None):
        """Compute the loss for the diffusion process."""
        if model_kwargs is None:
            model_kwargs = {}

        shape = [batch.shape[0]] + [1] * (batch.ndimension() - 1)
        sigma = self.sde.sigma_sample(shape, batch.device)
        coef_in = self.sde.scale_input(sigma)
        c_noise = self.sde.scale_noise(sigma).reshape(batch.shape[0])
        weight = self.sde.loss_weight(sigma)
        coef_skip, coef_out = self.sde.scale_output(sigma)
        noise = self.sde.noise_sample(batch, sigma)

        # Perturb the input data with noise
        perturbed_batch = batch + noise
        c_in = coef_in * perturbed_batch
        out = model(c_in, c_noise, **model_kwargs)

        # Compute the final output and loss
        c_out = coef_skip * perturbed_batch + coef_out * out
        losses = self.reduce_op(weight * ((c_out - batch) ** 2))

        return {"loss": losses}

    def training_losses(self, model, x_start, t, model_kwargs=None):
        """Wrapper for loss computation during training."""
        if model_kwargs is None:
            model_kwargs = {}
        return self.loss_fn(model, x_start, model_kwargs)


def gradient_descent(x0, y, H, H_t, z, eta, max_iter=1, lr=1.0, W=None, tol=1e-4):
    """
    Solve:
        min_x ½ * ||Hx - y||²_W + ½ * η * xᵀ(x - z)
    
    Args:
        x0: Initial guess
        y: Observed sinogram
        H: Forward projection operator
        H_t: Backprojection operator
        z: Denoised estimate
        eta: Regularization strength
        lr: Learning rate
        W: Optional weighting tensor for WLS (same shape as Hx)
    """
    x = x0.clone()

    for i in range(max_iter):
        x_prev = x.clone()

        # Compute gradient
        residual = H(x) - y
        if W is not None:
            residual = residual * W  # Weighted residual

        grad_data = H_t(residual)
        grad_reg = eta * (x - z)
        grad = grad_data + grad_reg

        # Gradient step
        x = x - lr * grad

        # Early stopping
        error = ((x - x_prev) ** 2).sum() / x0.size(0)
        if error < tol:
            break

    return x

def cp_tv_aniso_denoise(x0, lambda_tv, n_iter=10, tau=0.25, sigma=0.25, theta=1.0):
    x = x0.clone()
    x_bar = x.clone()
    p_x = th.zeros_like(x)
    p_y = th.zeros_like(x)

    for _ in range(n_iter):
        grad_x = F.pad(x_bar[:, :, :, 1:] - x_bar[:, :, :, :-1], (0, 1, 0, 0))
        grad_y = F.pad(x_bar[:, :, 1:, :] - x_bar[:, :, :-1, :], (0, 0, 0, 1))

        p_x = th.clamp(p_x + sigma * grad_x, min=-lambda_tv, max=lambda_tv)
        p_y = th.clamp(p_y + sigma * grad_y, min=-lambda_tv, max=lambda_tv)

        div_p = F.pad(p_x[:, :, :, :-1] - p_x[:, :, :, 1:], (1, 0, 0, 0)) + \
                F.pad(p_y[:, :, :-1, :] - p_y[:, :, 1:, :], (0, 0, 1, 0))

        x_new = (x - tau * (div_p + x - x0)) / (1 + tau)
        x_bar = x_new + theta * (x_new - x)
        x = x_new

    return x

def grad_x(f):
    u = f.clone()
    u[...,:-1] = f[...,1:]
    return u - f

def grad_y(f):
    u = f.clone()
    u[...,:-1,:] = f[...,1:,:]
    return u - f

def div_x(f):
    u0 = f.clone()
    u0[...,-1] = 0.0
    u1 = th.zeros_like(f)
    u1[...,1:] = f[...,:-1]
    return u0 - u1

def div_y(f):
    u0 = f.clone()
    u0[...,-1,:] = 0.0
    u1 = th.zeros_like(f)
    u1[...,1:,:] = f[...,:-1,:]
    return u0 - u1  

def Chambolle_Pock_TV(u0, alpha, iter):
    step = 0.25
    px = th.zeros_like(u0)
    py = th.zeros_like(u0)
    for i in range(iter):
        tmp = div_x(px) + div_y(py) - u0 / alpha
        dx = grad_x(tmp)
        dy = grad_y(tmp)
        tv = (dx ** 2 + dy ** 2).sqrt()
        px = (px + step * dx) / (1 + step * tv)
        py = (py + step * dy) / (1 + step * tv)
    u = u0 - alpha * (div_x(px) + div_y(py))
    return u

def gradient(x):
    """
    Compute first-order derivatives (grad_y, grad_x).
    Args:
        x: tensor of shape (B, 1, H, W)
    Returns:
        grad_y, grad_x: tensors of shape (B, 1, H, W)
    """
    grad_y = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1))  # pad bottom
    grad_x = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0))  # pad right
    return grad_y, grad_x

def divergence(grad):
    """
    Compute divergence of a first-order gradient.
    Args:
        grad: list [grad_y, grad_x], each of shape (B, 1, H, W)
    Returns:
        div: tensor of shape (B, 1, H, W)
    """
    grad_y, grad_x = grad

    div = th.zeros_like(grad_y)

    div[:, :, :, 0] = grad_x[:, :, :, 0]
    div[:, :, :, 1:-1] = grad_x[:, :, :, 1:-1] - grad_x[:, :, :, 0:-2]
    div[:, :, :, -1] = -grad_x[:, :, :, -1]

    div[:, :, 0, :] += grad_y[:, :, 0, :]
    div[:, :, 1:-1, :] += grad_y[:, :, 1:-1, :] - grad_y[:, :, 0:-2, :]
    div[:, :, -1, :] += -grad_y[:, :, -1, :]

    return div

def symmetrized_second_gradient(grad):
    """
    Compute symmetrized second-order derivatives.
    Args:
        grad: list [grad_y, grad_x] (first derivatives)
    Returns:
        grad_yy, grad_yx, grad_xy, grad_xx
    """
    grad_y, grad_x = grad

    grad_yy, grad_yx = gradient(grad_y)
    grad_xy, grad_xx = gradient(grad_x)

    return grad_yy, grad_yx, grad_xy, grad_xx

def second_order_divergence(grad2):
    """
    Compute divergence of second-order symmetrized gradient.
    Args:
        grad2: list [grad_yy, grad_yx, grad_xy, grad_xx]
    Returns:
        div_y, div_x
    """
    grad_yy, grad_yx, grad_xy, grad_xx = grad2

    div_sec_x = th.zeros_like(grad_xx)
    div_sec_x[:, :, :, 0] = grad_xx[:, :, :, 0]
    div_sec_x[:, :, :, 1:-1] = grad_xx[:, :, :, 1:-1] - grad_xx[:, :, :, :-2]
    div_sec_x[:, :, :, -1] = -grad_xx[:, :, :, -1]

    div_sec_x[:, :, 0, :] += grad_xy[:, :, 0, :]
    div_sec_x[:, :, 1:-1, :] += grad_xy[:, :, 1:-1, :] - grad_xy[:, :, :-2, :]
    div_sec_x[:, :, -1, :] += -grad_xy[:, :, -1, :]

    div_sec_y = th.zeros_like(grad_yx)
    div_sec_y[:, :, :, 0] = grad_yx[:, :, :, 0]
    div_sec_y[:, :, :, 1:-1] = grad_yx[:, :, :, 1:-1] - grad_yx[:, :, :, :-2]
    div_sec_y[:, :, :, -1] = -grad_yx[:, :, :, -1]

    div_sec_y[:, :, 0, :] += grad_yy[:, :, 0, :]
    div_sec_y[:, :, 1:-1, :] += grad_yy[:, :, 1:-1, :] - grad_yy[:, :, :-2, :]
    div_sec_y[:, :, -1, :] += -grad_yy[:, :, -1, :]

    return div_sec_y, div_sec_x

def proj_l2(g, alpha=1.0):
    """
    Project onto l2 ball of radius alpha (per pixel).
    Args:
        g: tensor of shape (2, B, C, H, W)
        alpha: scalar
    Returns:
        projected tensor
    """
    abs_sum = th.sum(g.abs(), dim=0, keepdim=True)
    denorm = th.clamp(abs_sum / alpha, 1e-8, 1.0)
    return g / denorm

def proj_double_norm(u, f, lambda_tv=1.0, tau=1.0):
    return (lambda_tv * u + tau * f) / (lambda_tv + tau)

def tgv_denoise_pd(x0, lambda_tv=1.0, alpha=0.01, L=24, n_iter=100, device=None, verbose=False):
    """
    Primal-Dual Second-Order TGV denoising:
    min_u 0.5*‖u - x0‖² + λ * TGV²(u)

    Args:
        x0: input tensor, shape (B, 1, H, W)
        lambda_tv: TGV lambda (weight on total variation)
        alpha: alpha parameter (relative weight on second-order derivative)
        L: Lipschitz constant
        n_iter: number of iterations
    Returns:
        u: denoised image tensor
    """
    if device is None:
        device = x0.device

    B, C, H, W = x0.shape
    gamma = lambda_tv
    delta = alpha
    mu = 2 * (gamma * delta) ** 0.5 / L

    tau = mu / (2 * gamma)
    sigma = mu / (2 * delta)
    theta = 1 / (1 + mu)

    # Init variables
    u = th.zeros_like(x0)
    p = th.zeros(2, B, C, H, W, device=device)
    v = th.zeros_like(p)
    q = th.zeros(4, B, C, H, W, device=device)

    u_bar = x0.clone()
    v_bar = th.zeros_like(v)

    energy_list = []

    for _ in range(n_iter):
        # --- Dual updates ---
        grad_u_bar = th.stack(gradient(u_bar), dim=0)
        p = proj_l2(p + sigma * (grad_u_bar - v_bar), 2.0)

        sym_grad_v_bar = th.stack(symmetrized_second_gradient(v_bar), dim=0)
        q = proj_l2(q + sigma * sym_grad_v_bar, 1.0)

        # --- Primal updates ---
        div_p = divergence(p)
        u_old = u.clone()
        u = proj_double_norm(u + tau * div_p, x0, lambda_tv, tau)

        u_bar = u + theta * (u - u_old)

        div_q = th.stack(second_order_divergence(q), dim=0)
        v_old = v.clone()
        v = v + tau * (p + div_q)
        v_bar = v + theta * (v - v_old)

        if verbose:
            fidelity = 0.5 * th.sum((u - x0) ** 2)
            tv1 = th.sum(th.abs(gradient(u)))
            tv2 = th.sum(th.abs(symmetrized_second_gradient(v)))
            energy = fidelity + lambda_tv * (tv1 + tv2)
            energy_list.append(energy.item())

    return u

def CG(x0, y, H, H_t, z, eta, max_iter, W=None, tol=1e-4):
    """
    Solve:
        min_x ½ * ||Hx - y||^2_W + ½ * η * xᵀ(x - z)

    Args:
        W: element-wise weight tensor same shape as y (or broadcastable)
    """
    x = x0.clone()
    grad_old = None

    for i in range(max_iter):
        u = x.clone()
        r = H(x) - y  # residual
        if W is not None:
            r = r * W  # weighted residual

        grad = H_t(r) + eta * (x - z)

        if i == 0:
            d = -grad
        else:
            beta = (grad ** 2).sum((2, 3), keepdim=True) / (grad_old ** 2).sum((2, 3), keepdim=True).clamp(min=1e-8)
            d = -grad + beta * d

        grad_old = grad

        Hd = H(d)
        if W is not None:
            Hd = Hd * W

        Ad = H_t(Hd) + eta * d
        step = - (grad * d).sum((2, 3), keepdim=True) / (d * Ad).sum((2, 3), keepdim=True).clamp(min=1e-8)

        x = x + step * d
        error = ((x - u) ** 2).sum() / x0.size(0)
        if error < tol:
            break

    return x   

def CG_with_TV(x0, y, H, H_t, z, eta, lambda_tv, max_iter, W=None, tol=1e-4):
    def tv_grad(x, weight=1.0, eps=1e-8):
        dx = x[..., 1:, :] - x[..., :-1, :]
        dy = x[..., :, 1:] - x[..., :, :-1]
        dx_pad = th.nn.functional.pad(dx, (0, 0, 0, 1))
        dy_pad = th.nn.functional.pad(dy, (0, 1, 0, 0))
        grad_x = dx_pad - th.nn.functional.pad(dx, (0, 0, 1, 0))
        grad_y = dy_pad - th.nn.functional.pad(dy, (1, 0, 0, 0))
        return weight * (grad_x + grad_y)

    x = x0.clone()
    grad_old = None

    for i in range(max_iter):
        u = x.clone()
        r = H(x) - y
        if W is not None:
            r = r * W
        grad = H_t(r) + eta * (x - z)
        if lambda_tv > 0:
            grad += tv_grad(x, weight=lambda_tv)

        if i == 0:
            d = -grad
        else:
            beta = (grad ** 2).sum((2, 3), keepdim=True) / (grad_old ** 2).sum((2, 3), keepdim=True).clamp(min=1e-8)
            d = -grad + beta * d
        grad_old = grad

        Hd = H(d)
        if W is not None:
            Hd = Hd * W
        Ad = H_t(Hd) + eta * d
        if lambda_tv > 0:
            Ad += tv_grad(d, weight=lambda_tv)
        step = - (grad * d).sum((2, 3), keepdim=True) / (d * Ad).sum((2, 3), keepdim=True).clamp(min=1e-8)

        x = x + step * d
        error = ((x - u) ** 2).sum() / x0.size(0)
        if error < tol:
            break

    return x


def OS_SART(u0, p, H, H_t, views, w=1.0, max_iter=1, group=16, u_ones=None, p_ones=None):
    eps = th.finfo(u0.dtype).eps  # Numerical stability

    # Precompute normalization terms if not provided
    if u_ones is None:
        u_ones = H_t(th.ones_like(p), views) / group
    else:
        u_ones = u_ones / group

    if p_ones is None:
        p_ones = H(th.ones_like(u0), views)

    u_ones = u_ones.clamp_min(eps)
    p_ones = p_ones.clamp_min(eps)

    u = u0.clone()
    for j in range(max_iter):
        for i in range(group):
            # Select subset of projections and views
            p_i = p[:, :, i::group].contiguous()
            views_i = views[i::group].contiguous()

            # Compute forward projection and error
            p_proj = H(u, views_i)
            p_error = (p_i - p_proj) / p_ones[:, :, i::group]

            # Back-project error
            u_error = H_t(p_error, views_i) / u_ones

            # Update estimate
            u = u + w * u_error

    return u

def pix2atten(x):
    y = (x + 1) / 2 * 2000 - 1024
    y = y / 1000 * 0.0192 + 0.0192
    return y

def atten2pix(y):
    x = (y - 0.0192) / 0.0192 * 1000 + 1024
    x = x / 2000 * 2 - 1
    return x


def conditioning(x0, z, y, eta, H, H_t, views, method, max_iter, lr, preprocess=True, group=16, u_ones=None, p_ones=None, **kwargs,):
    forward_projection = lambda x: H(x, views)
    projection_transpose = lambda p: H_t(p, views)
    # if "split" in kwargs:
    #     if re.search("LowDose", kwargs['split']):
    #         W = th.exp(-y)
    #     else:
    #         W = None
    # else:
    #     W = None
    if "W" in kwargs:
        W = kwargs["W"]
    else:
        W = None
    if preprocess:
        x0 = pix2atten(x0)
        z = pix2atten(z)
    if method == "gd":
        x = gradient_descent(x0, y, forward_projection, projection_transpose, z, eta, max_iter, lr, W)
    elif method == "cg":
        x = CG(x0, y, forward_projection, projection_transpose, z, eta, max_iter, W)
        # x = CG_with_TV(x0, y, forward_projection, projection_transpose, z, eta, kwargs["lambda_tv"], max_iter, W=None, tol=1e-4)
    elif method == "sart":
        x = OS_SART(x0, y, H, H_t, views, max_iter=max_iter, group=group, u_ones=u_ones, p_ones=p_ones)
    else:
        raise TypeError("Wrong method!")
    if preprocess:
        x = atten2pix(x)
    return x

# EDMSampler class handles sampling using the SDE
class EDMSampler:
    def __init__(self, sde, model):
        """Initialize EDMSampler with an SDE and model."""
        self.sde = sde
        self.model = model

    def update(self, input, sigma, dtype, model_kwargs=None):
        """Predict the output for a single iteration."""
        if model_kwargs is None:
            model_kwargs = {}
        input = input.to(dtype)
        sigma = sigma.to(dtype)
        coef_in = self.sde.scale_input(sigma)
        c_noise = self.sde.scale_noise(sigma).reshape(input.shape[0])
        coef_skip, coef_out = self.sde.scale_output(sigma)

        # Perturb the input and compute the output
        c_in = coef_in * input
        out = self.model(c_in, c_noise, **model_kwargs)
        c_out = coef_skip * input + coef_out * out
        return c_out

    def sample2(self, shape, condition_kwargs, input=None, t_start=1.0, model_kwargs=None, device=None, dtype=None, steps=None, solver="euler",
               alpha=1, stochastic=False, S_churn=0, S_min=0, S_max=float('inf'), S_noise=1):
        """Perform sampling using the SDE to generate data."""
        device = next(self.model.parameters()).device if device is None else device
        dtype = next(self.model.parameters()).dtype if dtype is None else dtype
        steps = self.sde.N if steps is None else steps
        step_indices = th.linspace(1 - t_start, 1.0, steps, device=device)
        t_steps = self.sde.time_step(step_indices)
        sigma_steps = self.sde.sigma(t_steps)
        t_steps = self.sde.sigma_inv(self.sde.round_sigma(sigma_steps))
        t_steps = th.cat([t_steps, th.zeros_like(t_steps[:1])]) # t_N = 0

        if input is None and t_start == 1.0:
            x0 = self.sde.initial_sample(t_steps[0], shape, device, th.float64)
        elif input is not None:
            batch = th.zeros(shape, device=device, dtype=dtype)
            sigma_shape = [batch.shape[0]] + [1] * (batch.ndimension() - 1)
            sigma = th.ones(sigma_shape, device=device, dtype=dtype) * sigma_steps[0]
            noise = self.sde.noise_sample(batch, sigma)
            x0 = input + noise
        else:
            raise TypeError(f'Input and t_start error!')
        
        x_next = x0
        # eta_min = 1.0
        # eta_max = condition_kwargs["eta"]
        # iter_max = condition_kwargs["max_iter"]
        # group = 16
        t_old = 1.0
        with th.no_grad():
            for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
                # condition_kwargs["eta"] = eta_min + (eta_max - eta_min) * (i / (steps - 1))
                # condition_kwargs["max_iter"] = int(1 + (iter_max - 1) * (1 - i / (steps - 1)))
                # logger.log(f"{i}, {condition_kwargs['eta']}, {condition_kwargs['max_iter']}")
                # if (i+1) % (steps // 10) == 0:
                #     group = group // 2
                #     group = max(group, 1)
                x_cur = x_next

                if stochastic:             
                    gamma = min(S_churn / steps, np.sqrt(2) - 1) if S_min <= self.sde.sigma(t_cur) <= S_max else 0
                    sigma_cur = self.sde.sigma(t_cur)
                    s_cur = self.sde.s(t_cur)
                    t_hat = self.sde.sigma_inv(self.sde.round_sigma(sigma_cur + gamma * sigma_cur))
                    sigma_hat = self.sde.sigma(t_hat)
                    s_hat = self.sde.s(t_hat)
                    x_hat = s_hat / s_cur * x_cur + (sigma_hat ** 2 - sigma_cur ** 2).clip(min=0).sqrt() * s_hat * S_noise * th.randn_like(x_cur)
                else:
                    t_hat = t_cur
                    x_hat = x_cur

                h = t_next - t_hat
                t_hat = _expand_dims(th.ones(shape[0], device=device) * t_hat, shape)
                t_next = _expand_dims(th.ones(shape[0], device=device) * t_next, shape)

                sigma_hat = self.sde.sigma(t_hat)
                s_hat = self.sde.s(t_hat)
                sigma_prime_hat = self.sde.sigma_prime(t_hat)
                s_prime_hat = self.sde.s_prime(t_hat)
                denoised = self.update(x_hat / s_hat, sigma_hat, dtype, model_kwargs).to(th.float64)
                if i == 0:
                    u_prev = denoised
                if (1 - step_indices[i]) > condition_kwargs["condition_end"]:
                    denoised = conditioning(u_prev, denoised, **condition_kwargs)
                    if not condition_kwargs['tv_denoise'] == "None":
                        # denoised = cp_tv_aniso_denoise(denoised, lambda_tv=condition_kwargs["lambda_tv"], n_iter=condition_kwargs["tv_iter"])
                        if condition_kwargs['tv_denoise'] == "tv":
                            denoised = Chambolle_Pock_TV(denoised, condition_kwargs["lambda_tv"], condition_kwargs["tv_iter"])
                        elif condition_kwargs['tv_denoise'] == "tgv":
                            denoised = tgv_denoise_pd(denoised, condition_kwargs["lambda_tv"], condition_kwargs["alpha"], 12, condition_kwargs["tv_iter"])
                    t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_old ** 2)) / 2.0
                    denoised = denoised + (t_old - 1.0) / t_new * (denoised - u_prev)
                # else:
                #     denoised = conditioning()
                u_prev = denoised
                t_old = t_new
                d_cur = (sigma_prime_hat / sigma_hat + s_prime_hat / s_hat) * x_hat - sigma_prime_hat * s_hat / sigma_hat * denoised
                x_tmp = x_hat + alpha * h * d_cur
                t_tmp = t_hat + alpha * h

                if solver == 'euler' or i == steps - 1:
                    x_next = x_hat + h * d_cur
                else:
                    assert solver == 'heun'
                    sigma_tmp = self.sde.sigma(t_tmp)
                    s_tmp = self.sde.s(t_tmp)
                    sigma_prime_tmp = self.sde.sigma_prime(t_tmp)
                    s_prime_tmp = self.sde.s_prime(t_tmp)
                    denoised = self.update(x_tmp / s_tmp, sigma_tmp, dtype, model_kwargs).to(th.float64)
                    # if (1 - step_indices[i]) > 0.1:
                    # denoised = conditioning(u_prev, denoised, **condition_kwargs)
                    d_tmp = (sigma_prime_tmp / sigma_tmp + s_prime_tmp / s_tmp) * x_tmp - sigma_prime_tmp * s_tmp / sigma_tmp * denoised
                    x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_tmp)

        return x_next
    
    def sample(self, shape, input=None, model_kwargs=None, device=None, dtype=None, steps=None, solver="heun",
               alpha=1, stochastic=False, S_churn=0, S_min=0, S_max=float('inf'), S_noise=1):
        """Perform sampling using the SDE to generate data."""
        device = next(self.model.parameters()).device if device is None else device
        dtype = next(self.model.parameters()).dtype if dtype is None else dtype
        steps = self.sde.N if steps is None else steps
        step_indices = th.arange(steps, dtype=th.float64, device=device)
        t_steps = self.sde.time_step(step_indices / (steps - 1))
        sigma_steps = self.sde.sigma(t_steps)
        t_steps = self.sde.sigma_inv(self.sde.round_sigma(sigma_steps))
        t_steps = th.cat([t_steps, th.zeros_like(t_steps[:1])]) # t_N = 0

        x0 = self.sde.initial_sample(t_steps[0], shape, device, th.float64) if input is None else input.to(device).to(th.float64)
        x_next = x0
        with th.no_grad():
            for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
                x_cur = x_next

                if stochastic:             
                    gamma = min(S_churn / steps, np.sqrt(2) - 1) if S_min <= self.sde.sigma(t_cur) <= S_max else 0
                    sigma_cur = self.sde.sigma(t_cur)
                    s_cur = self.sde.s(t_cur)
                    t_hat = self.sde.sigma_inv(self.sde.round_sigma(sigma_cur + gamma * sigma_cur))                  
                    sigma_hat = self.sde.sigma(t_hat)
                    s_hat = self.sde.s(t_hat)
                    x_hat = s_hat / s_cur * x_cur + (sigma_hat ** 2 - sigma_cur ** 2).clip(min=0).sqrt() * s_hat * S_noise * th.randn_like(x_cur)
                else:
                    t_hat = t_cur
                    x_hat = x_cur

                h = t_next - t_hat
                t_hat = _expand_dims(th.ones(shape[0], device=device) * t_hat, shape)
                t_next = _expand_dims(th.ones(shape[0], device=device) * t_next, shape)

                sigma_hat = self.sde.sigma(t_hat)
                s_hat = self.sde.s(t_hat)
                sigma_prime_hat = self.sde.sigma_prime(t_hat)
                s_prime_hat = self.sde.s_prime(t_hat)
                denoised = self.update(x_hat / s_hat, sigma_hat, dtype, model_kwargs).to(th.float64)
                d_cur = (sigma_prime_hat / sigma_hat + s_prime_hat / s_hat) * x_hat - sigma_prime_hat * s_hat / sigma_hat * denoised
                x_tmp = x_hat + alpha * h * d_cur
                t_tmp = t_hat + alpha * h

                if solver == 'euler' or i == steps - 1:
                    x_next = x_hat + h * d_cur
                else:
                    assert solver == 'heun'
                    sigma_tmp = self.sde.sigma(t_tmp)
                    s_tmp = self.sde.s(t_tmp)
                    sigma_prime_tmp = self.sde.sigma_prime(t_tmp)
                    s_prime_tmp = self.sde.s_prime(t_tmp)
                    denoised = self.update(x_tmp / s_tmp, sigma_tmp, dtype, model_kwargs).to(th.float64)
                    d_tmp = (sigma_prime_tmp / sigma_tmp + s_prime_tmp / s_tmp) * x_tmp - sigma_prime_tmp * s_tmp / sigma_tmp * denoised
                    x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_tmp)

        return x_next

