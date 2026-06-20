import torch as th
import diffusion.sde_lib as sde_lib
from .sde_lib import _expand_dims

class SDEDiffusion:
    def __init__(
        self,
        sde,
        continuous=True,
        reduce_mean=True,
        time_eps = 1e-5,
        likelihood_weighting = False,
    ):
        self.sde = sde
        self.continuous = continuous
        self.reduce_mean = reduce_mean
        self.time_eps = time_eps
        self.likelihood_weighting = likelihood_weighting
        self.num_timesteps = sde.N
        self.reduce_op = th.mean if self.reduce_mean else lambda *args, **kwargs: 0.5 * th.sum(*args, **kwargs)

    def score_fn(self, model, x, t, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        if isinstance(self.sde, (sde_lib.VPSDE, sde_lib.subVPSDE)):
            if self.continuous or isinstance(self.sde, sde_lib.subVPSDE):
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for
                # continuously-trained models.
                labels = t * 999
                std = self.sde.marginal_prob(th.zeros_like(x), t)[1]
            else:
                # For VP-trained models, t=0 corresponds to the lowest noise level
                labels = t * (self.sde.N - 1)
                std = self.sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]
            score = model(x, labels, **model_kwargs)
            score = - score / _expand_dims(std, score.shape)
        
        elif isinstance(self.sde, sde_lib.VESDE):
            if self.continuous:
                labels = self.sde.marginal_prob(th.zeros_like(x), t)[1]
            else:
                # For VE-trained models, t=0 corresponds to the highest noise level
                labels = (self.sde.T - t) * (self.sde.N - 1)
                labels = th.round(labels).long()
            score = model(x, labels, **model_kwargs)

        else:
            raise NotImplementedError(f"SDE class {self.sde.__class__.__name__} not yet supported.")
        
        return score
    
    def sde_loss_fn(self, model, batch, model_kwargs=None):
        """Compute the loss function.

        Args:
        batch: A mini-batch of training data.

        Returns:
        loss: A scalar that represents the average loss value across the mini-batch.
        """
        t = th.rand(batch.shape[0], device=batch.device) * (self.sde.T - self.time_eps) + self.time_eps
        z = th.randn_like(batch)
        mean, std = self.sde.marginal_prob(batch, t)
        std = _expand_dims(std, mean.shape)
        perturbed_data = mean + std * z
        score = self.score_fn(model, perturbed_data, t, model_kwargs)

        if not self.likelihood_weighting:
            losses = th.square(score * std + z)
            losses = self.reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = self.sde.sde(th.zeros_like(batch), t)[1] ** 2
            losses = th.square(score + z / std)
            losses = self.reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

        # loss = th.mean(losses)
        return {"loss": losses}
    
    def smld_loss_fn(self, model, batch, model_kwargs=None):
        """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
        assert isinstance(self.sde, sde_lib.VESDE), "SMLD training only works for VESDEs."

        # Previous SMLD models assume descending sigmas        

        labels = th.randint(0, self.sde.N, (batch.shape[0],), device=batch.device)
        sigmas = _extract_into_tensor(self.smld_sigma_array, labels, batch.shape)
        noise = th.randn_like(batch) * sigmas
        perturbed_data = noise + batch
        score = model(perturbed_data, labels, **model_kwargs)
        target = - noise / (sigmas ** 2)
        losses = th.square(score - target)
        losses = self.reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas.squeeze() ** 2
        # loss = th.mean(losses)
        return {"loss": losses}
    
    def ddpm_loss_fn(self, model, batch, model_kwargs=None):
        """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
        assert isinstance(self.sde, sde_lib.VPSDE), "DDPM training only works for VPSDEs."   
          
        labels = th.randint(0, self.sde.N, (batch.shape[0],), device=batch.device)
        sqrt_alphas_cumprod = self.sde.sqrt_alphas_cumprod.to(batch.device)
        sqrt_1m_alphas_cumprod = self.sde.sqrt_1m_alphas_cumprod.to(batch.device)
        noise = th.randn_like(batch)
        perturbed_data = _extract_into_tensor(sqrt_alphas_cumprod, labels, batch.shape) * batch + \
                        _extract_into_tensor(sqrt_1m_alphas_cumprod, labels, batch.shape) * noise
        score = model(perturbed_data, labels, **model_kwargs)
        losses = th.square(score - noise)
        losses = self.reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        # loss = th.mean(losses)
        return {"loss": losses}
    
    def training_losses(self, model, x_start, t, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        if self.continuous:
            return self.sde_loss_fn(model, x_start, model_kwargs)
        else:
            assert not self.likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
            if isinstance(self.sde, sde_lib.VESDE):
                self.smld_sigma_array = th.flip(self.sde.discrete_sigmas, dims=(0,))
                return self.smld_loss_fn(model, x_start, model_kwargs)
            elif isinstance(self.sde, sde_lib.VPSDE):
                return self.ddpm_loss_fn(model, x_start, model_kwargs)
            else:
                raise ValueError(f"Discrete training for {self.sde.__class__.__name__} is not recommended.")

def _extract_into_tensor(tensor, timesteps, broadcast_shape):
    """
    Extract values from a 1-D tensor for a batch of indices.

    :param tensor: the 1-D torch tensor.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = tensor.to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

