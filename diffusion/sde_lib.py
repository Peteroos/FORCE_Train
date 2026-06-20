"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc
import torch as th
import numpy as np

class SDE(abc.ABC):
	"""SDE abstract class. Functions are designed for a mini-batch of inputs."""

	def __init__(self, N):
		"""Construct an SDE.

		Args:
		N: number of discretization time steps.
		"""
		super().__init__()
		self.N = N

	@property
	@abc.abstractmethod
	def T(self):
		"""End time of the SDE."""
		pass

	@abc.abstractmethod
	def sde(self, x, t):
		pass

	@abc.abstractmethod
	def marginal_prob(self, x, t):
		"""Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
		pass

	@abc.abstractmethod
	def prior_sampling(self, shape):
		"""Generate one sample from the prior distribution, $p_T(x)$."""
		pass

	@abc.abstractmethod
	def prior_logp(self, z):
		"""Compute log-density of the prior distribution.

		Useful for computing the log-likelihood via probability flow ODE.

		Args:
		z: latent code
		Returns:
		log probability density
		"""
		pass

	def discretize(self, x, t):
		"""Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

		Useful for reverse diffusion sampling and probabiliy flow sampling.
		Defaults to Euler-Maruyama discretization.

		Args:
		x: a torch tensor
		t: a torch float representing the time step (from 0 to `self.T`)

		Returns:
		f, G
		"""
		dt = 1 / self.N
		drift, diffusion = self.sde(x, t)
		f = drift * dt
		G = diffusion * th.sqrt(th.tensor(dt, device=t.device))
		return f, G


class RSDE(SDE):
	def __init__(self, sde, score_fn, probability_flow=False):
		super().__init__(sde.N)
		self.probability_flow = probability_flow
		self.sde_fn = sde.sde
		self.discretize_fn = sde.discretize
		self.score_fn = score_fn

	@property
	def T(self):
		return 1

	def sde(self, model, x, t, model_kwargs=None):
		"""Create the drift and diffusion functions for the reverse SDE/ODE."""
		drift, diffusion = self.sde_fn(x, t)
		score = self.score_fn(model, x, t, model_kwargs)
		drift = drift - _expand_dims(diffusion, score.shape) ** 2 * score * (0.5 if self.probability_flow else 1.)
		# Set the diffusion function to zero for ODEs.
		diffusion = 0. if self.probability_flow else diffusion
		return drift, diffusion

	def discretize(self, model, x, t, model_kwargs=None):
		"""Create discretized iteration rules for the reverse diffusion sampler."""
		f, G = self.discretize_fn(x, t)
		score = self.score_fn(model, x, t, model_kwargs)
		rev_f = f - _expand_dims(G, score.shape) ** 2 * score * (0.5 if self.probability_flow else 1.)
		rev_G = th.zeros_like(G) if self.probability_flow else G
		return rev_f, rev_G

	def marginal_prob(self, x, t):
		pass

	def prior_sampling(self, shape):
		pass

	def prior_logp(self, z):
		pass


class VPSDE(SDE):
	def __init__(self, beta_min=0.1, beta_max=20, N=1000):
		"""Construct a Variance Preserving SDE.

		Args:
		beta_min: value of beta(0)
		beta_max: value of beta(1)
		N: number of discretization steps
		"""
		super().__init__(N)
		self.beta_0 = beta_min
		self.beta_1 = beta_max
		self.N = N
		self.discrete_betas = th.linspace(beta_min / N, beta_max / N, N)
		self.alphas = 1. - self.discrete_betas
		self.alphas_cumprod = th.cumprod(self.alphas, dim=0)
		self.sqrt_alphas_cumprod = th.sqrt(self.alphas_cumprod)
		self.sqrt_1m_alphas_cumprod = th.sqrt(1. - self.alphas_cumprod)

	@property
	def T(self):
		return 1

	def sde(self, x, t):
		beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
		drift = - 0.5 * _expand_dims(beta_t, x.shape) * x
		diffusion = th.sqrt(beta_t)
		return drift, diffusion

	def marginal_prob(self, x, t):
		log_mean_coeff = - 0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
		mean = th.exp(_expand_dims(log_mean_coeff, x.shape)) * x
		std = th.sqrt(1. - th.exp(2. * log_mean_coeff))
		return mean, std

	def prior_sampling(self, shape):
		return th.randn(*shape)

	def prior_logp(self, z):
		shape = z.shape
		N = np.prod(shape[1:])
		logps = - N / 2. * np.log(2 * np.pi) - th.sum(z ** 2, dim=tuple(i for i in range(1, z.dim()))) / 2.
		return logps

	def discretize(self, x, t):
		"""DDPM discretization."""
		timestep = (t * (self.N - 1) / self.T).long()
		beta = self.discrete_betas.to(x.device)[timestep]
		alpha = self.alphas.to(x.device)[timestep]
		sqrt_beta = th.sqrt(beta)
		f = th.sqrt(_expand_dims(alpha, x.shape)) * x - x
		G = sqrt_beta
		return f, G


class subVPSDE(VPSDE):
	def __init__(self, beta_min=0.1, beta_max=20, N=1000):
		"""Construct the sub-VP SDE that excels at likelihoods.

		Args:
		beta_min: value of beta(0)
		beta_max: value of beta(1)
		N: number of discretization steps
		"""
		super().__init__(beta_min, beta_max, N)
		# self.beta_0 = beta_min
		# self.beta_1 = beta_max
		# self.N = N

	@property
	def T(self):
		return 1

	def sde(self, x, t):
		beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
		drift = -0.5 * _expand_dims(beta_t, x.shape) * x
		discount = 1. - th.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
		diffusion = th.sqrt(beta_t * discount)
		return drift, diffusion

	def marginal_prob(self, x, t):
		log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
		mean = th.exp(_expand_dims(log_mean_coeff, x.shape)) * x
		std = 1 - th.exp(2. * log_mean_coeff)
		return mean, std

	def prior_sampling(self, shape):
		return th.randn(*shape)

	def prior_logp(self, z):
		shape = z.shape
		N = np.prod(shape[1:])
		return - N / 2. * np.log(2 * np.pi) - th.sum(z ** 2, dim=tuple(i for i in range(1, z.dim()))) / 2.


class VESDE(SDE):
	def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
		"""Construct a Variance Exploding SDE.

		Args:
		sigma_min: smallest sigma.
		sigma_max: largest sigma.
		N: number of discretization steps
		"""
		super().__init__(N)
		self.sigma_min = sigma_min
		self.sigma_max = sigma_max
		self.discrete_sigmas = th.exp(th.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
		self.N = N

	@property
	def T(self):
		return 1

	def sde(self, x, t):
		sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
		drift = th.zeros_like(x)
		diffusion = sigma * th.sqrt(th.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
													device=t.device))
		return drift, diffusion

	def marginal_prob(self, x, t):
		std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
		mean = x
		return mean, std

	def prior_sampling(self, shape):
		return th.randn(*shape) * self.sigma_max

	def prior_logp(self, z):
		shape = z.shape
		N = np.prod(shape[1:])
		return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - th.sum(z ** 2, dim=tuple(i for i in range(1, z.dim()))) / (2 * self.sigma_max ** 2)

	def discretize(self, x, t):
		"""SMLD(NCSN) discretization."""
		timestep = (t * (self.N - 1) / self.T).long()
		sigma = self.discrete_sigmas.to(t.device)[timestep]
		adjacent_sigma = th.where(timestep == 0, th.zeros_like(t),
									self.discrete_sigmas[timestep - 1].to(t.device))
		f = th.zeros_like(x)
		G = th.sqrt(sigma ** 2 - adjacent_sigma ** 2)
		return f, G


def _expand_dims(tensor, shape):
    res = tensor
    while len(res.shape) < len(shape):
        res = res[..., None]
    return res