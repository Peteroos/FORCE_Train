import abc
import functools

import torch as th
import numpy as np

# from torchdiffeq import odeint
from scipy import integrate
from .sde_lib import RSDE, VPSDE, VESDE, subVPSDE, _expand_dims

_CORRECTORS = {}
_PREDICTORS = {}

def register_component(component_registry, cls=None, *, name=None):
    """Generic decorator for registering classes to a component registry."""
    def _register(cls):
        local_name = name or cls.__name__
        if local_name in component_registry:
            raise ValueError(f'Already registered model with name: {local_name}')
        component_registry[local_name] = cls
        return cls

    return _register if cls is None else _register(cls)

register_predictor = functools.partial(register_component, _PREDICTORS)
register_corrector = functools.partial(register_component, _CORRECTORS)

def get_component(component_registry, name):
    """Generic function to get a component by name."""
    if name not in component_registry:
        raise ValueError(f"Component with name '{name}' not found.")
    return component_registry[name]

get_predictor = functools.partial(get_component, _PREDICTORS)
get_corrector = functools.partial(get_component, _CORRECTORS)

class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, model, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        self.rsde = RSDE(sde, score_fn, probability_flow)
        self.model = model

    @abc.abstractmethod
    def update_fn(self, x, t, model_kwargs):
        """One update of the predictor.

        Args:
        x: A PyTorch tensor representing the current state
        t: A Pytorch tensor representing the current time step.

        Returns:
        x: A PyTorch tensor of the next state.
        x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass

@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
    """Euler-Maruyama predictor."""
    def __init__(self, sde, model, score_fn, probability_flow=False):
        super().__init__(sde, model, score_fn, probability_flow)

    def update_fn(self, x, t, model_kwargs=None):
        dt = -1. / self.rsde.N
        drift, diffusion = self.rsde.sde(self.model, x, t, model_kwargs)
        x_mean = x + drift * dt
        z = th.randn_like(drift)
        x = x_mean + _expand_dims(diffusion, x_mean.shape) * np.sqrt(-dt) * z
        return x

@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
    """Reverse diffusion predictor."""
    def __init__(self, sde, model, score_fn, probability_flow=False):
        super().__init__(sde, model, score_fn, probability_flow)

    def update_fn(self, x, t, model_kwargs=None):
        f, G = self.rsde.discretize(self.model, x, t, model_kwargs)
        z = th.randn_like(x)
        x_mean = x - f
        x = x_mean + _expand_dims(G, x_mean.shape) * z
        return x

@register_predictor(name='ancestral')
class AncestralPredictor(Predictor):
    """Ancestral predictor."""
    def __init__(self, sde, model, score_fn, probability_flow=False):
        super().__init__(sde, model, score_fn, probability_flow)
        if not isinstance(sde, (VPSDE, VESDE)):
            raise ValueError('Ancestral sampling is only supported for VPSDE and VESDE.')
        
        self._update_fn = self._get_update_fn()        
    
    def _get_update_fn(self):
        if isinstance(self.sde, VPSDE):
            return self._update_vpsde
        elif isinstance(self.sde, VESDE):
            return self._update_vesde
        else:
            raise ValueError('Ancestral sampling is only supported for VPSDE and VESDE.')
    
    def _update_vesde(self, x, t, model_kwargs=None):
        timestep = (t * (self.sde.N - 1) / self.sde.T).long()
        sigma = self.sde.discrete_sigmas[timestep]
        adjacent_sigma = th.where(timestep == 0, th.zeros_like(t), self.sde.discrete_sigmas.to(t.device)[timestep - 1])
        score = self.score_fn(self.model, x, t, model_kwargs)
        x_mean = x + score * _expand_dims((sigma ** 2 - adjacent_sigma ** 2), score.shape)
        std = th.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
        noise = th.randn_like(x)
        x = x_mean + _expand_dims(std, x_mean.shape) * noise
        return x
    
    def _update_vpsde(self, x, t, model_kwargs=None):
        timestep = (t * (self.sde.N - 1) / self.sde.T).long()
        beta = _expand_dims(self.sde.discrete_betas.to(t.device)[timestep], x.shape)
        score = self.score_fn(self.model, x, t, model_kwargs)
        x_mean = x + beta * score / th.sqrt(1. - beta)
        noise = th.randn_like(x)
        x = x_mean + th.sqrt(beta) * noise
        return x

    def update_fn(self, x, t, model_kwargs=None):
        return self._update_fn(x, t, model_kwargs)

@register_predictor(name='none') 
class NonePredictor(Predictor):
    """Non predictor."""
    def __init__(self, sde, model, score_fn, probability_flow=False):
        super().__init__(sde, model, score_fn, probability_flow)
    
    def update_fn(self, x, t, model_kwargs=None):
        return x

class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, model, score_fn, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.model = model
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t, model_kwargs):
        """One update of the corrector.

        Args:
        x: A PyTorch tensor representing the current state
        t: A Pytorch tensor representing the current time step.

        Returns:
        x: A PyTorch tensor of the next state.
        x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass

@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
    """Langevin corrector."""
    def __init__(self, sde, model, score_fn, snr, n_steps):
        super().__init__(sde, model, score_fn, snr, n_steps)

        self._get_alpha = self._get_alpha_fn()

    def _get_alpha_fn(self):
        if isinstance(self.sde, (VPSDE, subVPSDE)):
            return self._get_alpha1
        else:
            return self._get_alpha2

    def _get_alpha1(self, t):
        timestep = (t * (self.sde.N - 1) / self.sde.T).long()
        alpha = self.sde.alphas.to(t.device)[timestep]
        return alpha
    
    def _get_alpha2(self, t):
        return th.ones_like(t)

    def get_alpha(self, t):
        return self._get_alpha(t)

    def update_fn(self, x, t, model_kwargs=None):
        alpha = self.get_alpha(t)

        for i in range(self.n_steps):
            grad = self.score_fn(self.model, x, t, model_kwargs)
            noise = th.randn_like(x)
            grad_norm = th.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = th.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = _expand_dims((self.snr * noise_norm / grad_norm) ** 2 * 2 * alpha, x.shape)
            x_mean = x + step_size * grad
            x = x_mean + th.sqrt(step_size * 2) * noise
        return x
    
@register_corrector(name='ald')
class AnnealedLangevinDynamics(LangevinCorrector):
    """Annealed Langevin dynamics corrector."""
    def __init__(self, sde, model, score_fn, snr, n_steps):
        super().__init__(sde, model, score_fn, snr, n_steps)

    def update_fn(self, x, t, model_kwargs=None):
        alpha = self.get_alpha(t)
        std = self.sde.marginal_prob(x, t)[1]

        for i in range(self.n_steps):
            grad = self.score_fn(self.model, x, t, model_kwargs)
            noise = th.randn_like(x)
            step_size = (self.snr * std) ** 2 * 2 * alpha
            x_mean = x + _expand_dims(step_size, x.shape) * grad
            x = x_mean + noise * th.sqrt(_expand_dims(step_size, x.shape) * 2)

        return x
    
@register_corrector(name='none')
class NoneCorrector(Corrector):
    """Non corrector."""
    def __init__(self, sde, model, score_fn, snr, n_steps):
        super().__init__(sde, model, score_fn, snr, n_steps)

    def update_fn(self, x, t, model_kwargs=None):
        return x
    
class PCSampler:
    """Predictor-corrector sampler."""
    def __init__(self, sde, model, snr, score_fn, predictor="euler_maruyama", corrector="langevin", n_steps=1, probability_flow=False, progress=False, eps=1e-5):
        self.sde = sde
        self.model = model
        self.score_fn = score_fn
        self.progress = progress
        self.eps = eps
        self.predictor = get_predictor(predictor)(sde, model, score_fn, probability_flow)
        self.corrector = get_corrector(corrector)(sde, model, score_fn, snr, n_steps)

    def sample(self, shape, input=None, model_kwargs=None, device=None):
        device = next(self.model.parameters()).device if device is None else device
        x = self.sde.prior_sampling(shape).to(device) if input is None else input.to(device)
        timesteps = th.linspace(self.sde.T, self.eps, self.sde.N, device=device)
        indices = list(range(self.sde.N))
        if self.progress:
            from tqdm import tqdm
            indices = tqdm(indices, desc='Sampling')
        with th.no_grad():
            for i in indices:
                t = timesteps[i]
                vec_t = th.ones(shape[0], device=t.device) * t            
                x = self.predictor.update_fn(x, vec_t, model_kwargs)
                x = self.corrector.update_fn(x, vec_t, model_kwargs)

        return x

def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return th.from_numpy(x.reshape(shape))

class ODESampler:
    """ODE sampler."""
    def __init__(self, sde, model, score_fn, probability_flow=False, eps=1e-5):
        self.sde = sde
        self.model = model
        self.score_fn = score_fn
        self.eps = eps
        self.rsde = RSDE(sde, score_fn, probability_flow)
        self.predictor = get_predictor("reverse_diffusion")(sde, model, score_fn, probability_flow)

    def sample(self, shape, input=None, model_kwargs=None, device=None, rtol=1e-7, atol=1e-9, method="RK45", steps=None, denoise=True):
        device = next(self.model.parameters()).device if device is None else device
        x0 = self.sde.prior_sampling(shape).to(device) if input is None else input.to(device)
        # steps = self.sde.N if steps is None else steps
        # timesteps = th.linspace(self.sde.T, self.eps, steps, device=device)
        timesteps = np.linspace(self.sde.T, self.eps, steps) if not steps is None else None

        def ode_func(t, x):
            x = from_flattened_numpy(x, shape).to(device).type(x0.dtype)
            vec_t = th.ones(shape[0], device=x.device) * t
            drift = self.rsde.sde(self.model, x, vec_t, model_kwargs)[0]
            return to_flattened_numpy(drift)
        with th.no_grad():
            # solution = odeint(ode_func, x0, timesteps, rtol=rtol, atol=atol, method=method)
            solution = integrate.solve_ivp(ode_func, (self.sde.T, self.eps), to_flattened_numpy(x0), rtol=rtol, atol=atol, method=method, t_eval=timesteps)
            x = th.tensor(solution.y[:, -1]).reshape(shape).to(device).type(x0.dtype)
            if denoise == True:
                vec_eps = th.ones(x.shape[0], device=x.device) * self.eps
                x = self.predictor.update_fn(x, vec_eps, model_kwargs)

        return x
