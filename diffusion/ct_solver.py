import torch as th
import numpy as np
from .sde_lib import _expand_dims

def conjugate_gradient(x0, y, H, H_t, num_iter=1, tol=1e-4, lambda_reg=0.1, denoised_x=None):
    """
    Conjugate Gradient method to solve min_x 0.5 * ||Hx - y||_2^2

    Args:
        x0 (Tensor): Initial guess.
        y (Tensor): Measurement.
        H (callable): Forward operator.
        H_t (callable): Adjoint (transpose) of H.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for stopping criteria.

    Returns:
        Tensor: Solution x.
    """
    x = x0.clone()
    grad = H_t(H(x) - y)
    if denoised_x is not None:
        grad += lambda_reg * (x - denoised_x)
    d = -grad

    eps = th.finfo(x.dtype).eps  # For numerical stability

    for i in range(num_iter):
        Hd = H(d)

        # Compute step size
        denom = (Hd ** 2).sum(dim=(2, 3), keepdim=True).clamp_min(eps)
        alpha = (grad * d).sum(dim=(2, 3), keepdim=True) / denom

        # Update x
        x_new = x + alpha * d

        # Check convergence
        error = ((x_new - x) ** 2).sum().sqrt() / (x.numel() + eps)
        if error < tol:
            x = x_new
            break

        # Compute new gradient
        grad_new = H_t(H(x_new) - y)
        if denoised_x is not None:
            grad_new += lambda_reg * (x_new - denoised_x)

        # Compute beta using Fletcher-Reeves formula
        num = (grad_new ** 2).sum(dim=(2, 3), keepdim=True)
        denom = (grad ** 2).sum(dim=(2, 3), keepdim=True).clamp_min(eps)
        beta = num / denom

        # Update direction
        d = -grad_new + beta * d

        # Prepare for next iteration
        x = x_new
        grad = grad_new

    return x

def gradient_descent(x0, y, H, H_t, alpha=1e-3, num_iter=1, tol=1e-4, lambda_reg=0.1, denoised_x=None):
    """
    Gradient Descent method to solve min_x 0.5 * ||Hx - y||_2^2

    Args:
        x0 (Tensor): Initial guess.
        y (Tensor): Measurement.
        H (callable): Forward operator.
        H_t (callable): Adjoint (transpose) of H.
        alpha (float): Learning rate.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for stopping criteria.

    Returns:
        Tensor: Solution x.
    """
    x = x0.clone()
    eps = th.finfo(x.dtype).eps  # For numerical stability

    for i in range(num_iter):
        # Compute gradient
        grad = H_t(H(x) - y)

        if denoised_x is not None:
            grad_reg = x - denoised_x
            grad += lambda_reg * grad_reg

        # Update x
        x_new = x - alpha * grad

        # Check convergence
        error = ((x_new - x) ** 2).sum().sqrt() / (x.numel() + eps)
        if error < tol:
            x = x_new
            break

        x = x_new

    return x

def OS_SART(u0, p, H, H_t, views, w=1.0, num_iter=1, group=16, u_ones=None, p_ones=None, lambda_reg=0.1, denoised_x=None):
    """
    Ordered Subsets Simultaneous Algebraic Reconstruction Technique (OS-SART).

    Args:
        u0 (Tensor): Initial estimate.
        p (Tensor): Projection data.
        H (callable): Forward operator.
        H_t (callable): Adjoint (transpose) operator.
        views (Tensor): View angles or projection matrices.
        w (float): Relaxation parameter.
        num_iter (int): Number of iterations.
        group (int): Number of subsets.
        u_ones (Tensor, optional): Precomputed normalization term for back-projection.
        p_ones (Tensor, optional): Precomputed normalization term for projection.

    Returns:
        Tensor: Reconstructed image.
    """

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

    for j in range(num_iter):
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

        if denoised_x is not None:
            u = u - lambda_reg * (u - denoised_x)

    return u

def conditioning(x, y, views, H, H_t, method, **kwargs):
    """
    Apply selected iterative reconstruction method.

    Args:
        x (Tensor): Initial estimate.
        y (Tensor): Projection data.
        views (Tensor): Projection views.
        H (callable): Forward operator.
        H_t (callable): Adjoint operator.
        method (str): Reconstruction method ('cg', 'gd', 'sart').
        **kwargs: Additional parameters for specific methods.

    Returns:
        Tensor: Reconstructed image.
    """
    method = method.lower()
    projection = lambda img: H(img, views)
    backprojection = lambda proj: H_t(proj, views)

    if method == "cg":
        res = conjugate_gradient(x, y, projection, backprojection,
                                 max_iter=kwargs.get("num_iter", 1),
                                 tol=kwargs.get("tol", 1e-4))
    elif method == "gd":
        res = gradient_descent(x, y, projection, backprojection,
                               alpha=kwargs.get("alpha", 1e-3),
                               max_iter=kwargs.get("num_iter", 1),
                               tol=kwargs.get("tol", 1e-4))
    elif method == "sart":
        res = OS_SART(x, y, H, H_t, views,
                      w=kwargs.get("w", 1.0),
                      num_iter=kwargs.get("num_iter", 1),
                      group=kwargs.get("group", 16),
                      u_ones=kwargs.get("u_ones", None),
                      p_ones=kwargs.get("p_ones", None))
    else:
        raise ValueError(f"Unsupported method '{method}'. Choose from 'cg', 'gd', 'sart'.")

    return res

def red_conditioning(x_t, y, views, H, H_t, denoised_x=None, method="gd", **kwargs):
    """
    Apply RED conditioning using different optimization methods.

    Args:
        x_t (Tensor): Current DDPM sample.
        y (Tensor): Observed measurement.
        views (Tensor): Projection views.
        H (callable): Forward operator.
        H_t (callable): Adjoint operator.
        denoised_x (Tensor, optional): Precomputed denoised version of x_t. If None, direct conditioning is used.
        method (str): Optimization method ('gd', 'cg', 'sart').
        **kwargs: Additional parameters for specific methods.

    Returns:
        Tensor: Conditioned x_t.
    """
    method = method.lower()
    projection = lambda img: H(img, views)
    backprojection = lambda proj: H_t(proj, views)

    if method == "gd":
        res = gradient_descent(x_t, y, projection, backprojection,
                               alpha=kwargs.get("alpha", 1e-3),
                               max_iter=kwargs.get("num_iter", 1),
                               tol=kwargs.get("tol", 1e-4),
                               lambda_reg=kwargs.get("lambda_reg", 0.1),
                               denoised_x=denoised_x)

    elif method == "cg":
        res = conjugate_gradient(x_t, y, projection, backprojection,
                                 max_iter=kwargs.get("num_iter", 1),
                                 tol=kwargs.get("tol", 1e-4),
                                 lambda_reg=kwargs.get("lambda_reg", 0.1),
                                 denoised_x=denoised_x)

    elif method == "sart":
        res = OS_SART(x_t, y, H, H_t, views,
                      w=kwargs.get("w", 1.0),
                      num_iter=kwargs.get("num_iter", 1),
                      group=kwargs.get("group", 16),
                      lambda_reg=kwargs.get("lambda_reg", 0.1),
                      denoised_x=denoised_x)
    else:
        raise ValueError(f"Unsupported method '{method}'. Choose from 'gd', 'cg', 'sart'.")

    return res

class CTEDMSampler:
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