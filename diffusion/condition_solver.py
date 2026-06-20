import torch
import torch.nn.functional as F
from .dpm_solver import DPM_Solver, expand_dims
from .respace import SpacedDiffusion
from .gaussian import _extract_into_tensor
from .sde import SDEDiffusion
import diffusion.sde_lib as sde_lib
import numpy as np


class DPMConditionSolver(DPM_Solver):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def conditioning(
        self,
        x,
        y,
        Afun,
        Ainv,
        mask,
        coeff1,
        coeff2,
        t,
    ):
        y_t = self.get_condition_t(x, y, Afun, t)
        y_x = Afun(x)
        conditioned_yt = (y_t * coeff1 + y_x * (1 - coeff1)) * mask + (y_t * coeff2 + y_x * (1 - coeff2)) * (1 - mask)
        conditioned_x = Ainv(conditioned_yt)
        return conditioned_x

    def get_condition_t(
        self, 
        x, 
        y, 
        Afun, 
        t
    ):
        ns = self.noise_schedule
        log_mean_coeff = ns.marginal_log_mean_coeff(t)
        mean = torch.exp(expand_dims(log_mean_coeff, y.dim())) * y
        std = ns.marginal_std(t)
        z = torch.randn_like(x)
        Az = Afun(z)
        condition_t = mean + expand_dims(std, y.dim()) * Az
        return condition_t
        
    def conditioned_dpm_solver_adaptive(
        self, 
        x,
        condition_kwargs,
        order, 
        t_T, 
        t_0, 
        h_init=0.05, 
        atol=0.0078, 
        rtol=0.05, 
        theta=0.9, 
        t_err=1e-5, 
        solver_type='dpmsolver',
        freq=1,
    ):
        ns = self.noise_schedule
        s = t_T * torch.ones((1,)).to(x)
        lambda_s = ns.marginal_lambda(s)
        lambda_0 = ns.marginal_lambda(t_0 * torch.ones_like(s).to(x))
        h = h_init * torch.ones_like(s).to(x)
        x_prev = x
        nfe = 0
        step = 0
        if order == 2:
            r1 = 0.5
            lower_update = lambda x, s, t: self.dpm_solver_first_update(x, s, t, return_intermediate=True)
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_second_update(x, s, t, r1=r1, solver_type=solver_type, **kwargs)
        elif order == 3:
            r1, r2 = 1. / 3., 2. / 3.
            lower_update = lambda x, s, t: self.singlestep_dpm_solver_second_update(x, s, t, r1=r1, return_intermediate=True, solver_type=solver_type)
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_third_update(x, s, t, r1=r1, r2=r2, solver_type=solver_type, **kwargs)
        else:
            raise ValueError("For adaptive step size solver, order must be 2 or 3, got {}".format(order))       
        while torch.abs((s - t_0)).mean() > t_err:
            t = ns.inverse_lambda(lambda_s + h)
            if (step % freq) == 0:
                x = self.conditioning(x, t=s, **condition_kwargs)
            x_lower, lower_noise_kwargs = lower_update(x, s, t)
            x_higher = higher_update(x, s, t, **lower_noise_kwargs)
            delta = torch.max(torch.ones_like(x).to(x) * atol, rtol * torch.max(torch.abs(x_lower), torch.abs(x_prev)))
            norm_fn = lambda v: torch.sqrt(torch.square(v.reshape((v.shape[0], -1))).mean(dim=-1, keepdim=True))
            E = norm_fn((x_higher - x_lower) / delta).max()
            if torch.all(E <= 1.):
                x = x_higher
                s = t
                x_prev = x_lower
                lambda_s = ns.marginal_lambda(s)
            h = torch.min(theta * h * torch.float_power(E, -1. / order).float(), lambda_0 - lambda_s)
            nfe += order
            step +=1
        print('adaptive solver nfe', nfe)
        return x
    
    
    def conditioned_sample(
        self, 
        x,
        condition_kwargs,
        freq=1,
        steps=20, 
        t_start=None, 
        t_end=None, 
        order=2, 
        skip_type='time_uniform',
        method='multistep', 
        lower_order_final=True, 
        denoise_to_zero=False, 
        solver_type='dpmsolver',
        atol=0.0078, 
        rtol=0.05, 
        return_intermediate=False,
    ):
        t_0 = 1. / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert t_0 > 0 and t_T > 0, "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        if return_intermediate:
            assert method in ['multistep', 'singlestep', 'singlestep_fixed'], "Cannot use adaptive solver when saving intermediate values"
        if self.correcting_xt_fn is not None:
            assert method in ['multistep', 'singlestep', 'singlestep_fixed'], "Cannot use adaptive solver when correcting_xt_fn is not None"
        device = x.device
        intermediates = []
        with torch.no_grad():
            if method == 'adaptive':
                x = self.conditioned_dpm_solver_adaptive(x, condition_kwargs, order=order, t_T=t_T, t_0=t_0, atol=atol, rtol=rtol, solver_type=solver_type, freq=freq)
            elif method == 'multistep':
                assert steps >= order
                timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
                assert timesteps.shape[0] - 1 == steps
                # Init the initial values.
                step = 0
                t = timesteps[step]
                if (step % freq) == 0:
                    x = self.conditioning(x, t=t, **condition_kwargs)
                t_prev_list = [t]
                model_prev_list = [self.model_fn(x, t)]
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step)
                if return_intermediate:
                    intermediates.append(x)
                # Init the first `order` values by lower order multistep DPM-Solver.
                for step in range(1, order):
                    t = timesteps[step]
                    x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, step, solver_type=solver_type)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                    if (step % freq) == 0:
                        x = self.conditioning(x, t=t, **condition_kwargs)
                    t_prev_list.append(t)
                    model_prev_list.append(self.model_fn(x, t))
                # Compute the remaining values by `order`-th order multistep DPM-Solver.
                for step in range(order, steps + 1):
                    t = timesteps[step]
                    # We only use lower order for steps < 10
                    if lower_order_final and steps < 10:
                        step_order = min(order, steps + 1 - step)
                    else:
                        step_order = order
                    x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, step_order, solver_type=solver_type)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                    if (step % freq) == 0:
                        x = self.conditioning(x, t=t, **condition_kwargs)
                    for i in range(order - 1):
                        t_prev_list[i] = t_prev_list[i + 1]
                        model_prev_list[i] = model_prev_list[i + 1]
                    t_prev_list[-1] = t
                    # We do not need to evaluate the final model value.
                    if step < steps:
                        model_prev_list[-1] = self.model_fn(x, t)
            elif method in ['singlestep', 'singlestep_fixed']:
                if method == 'singlestep':
                    timesteps_outer, orders = self.get_orders_and_timesteps_for_singlestep_solver(steps=steps, order=order, skip_type=skip_type, t_T=t_T, t_0=t_0, device=device)
                elif method == 'singlestep_fixed':
                    K = steps // order
                    orders = [order,] * K
                    timesteps_outer = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=K, device=device)
                for step, order in enumerate(orders):
                    s, t = timesteps_outer[step], timesteps_outer[step + 1]
                    if (step % freq) == 0:
                        x = self.conditioning(x, t=s, **condition_kwargs)
                    timesteps_inner = self.get_time_steps(skip_type=skip_type, t_T=s.item(), t_0=t.item(), N=order, device=device)
                    lambda_inner = self.noise_schedule.marginal_lambda(timesteps_inner)
                    h = lambda_inner[-1] - lambda_inner[0]
                    r1 = None if order <= 1 else (lambda_inner[1] - lambda_inner[0]) / h
                    r2 = None if order <= 2 else (lambda_inner[2] - lambda_inner[0]) / h
                    x = self.singlestep_dpm_solver_update(x, s, t, order, solver_type=solver_type, r1=r1, r2=r2)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
            else:
                raise ValueError("Got wrong method {}".format(method))
            if denoise_to_zero:
                t = torch.ones((1,)).to(device) * t_0
                x = self.denoise_to_zero_fn(x, t)
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step + 1)
                if return_intermediate:
                    intermediates.append(x)
        if return_intermediate:
            return x, intermediates
        else:
            return x


class GaussianConditionSolver(SpacedDiffusion):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def conditioning(
        self,
        x,
        y,
        Afun,
        Ainv,
        mask,
        coeff1,
        coeff2,
        t,
    ):
        y_t = self.get_condition_t(x, y, Afun, t)
        y_x = Afun(x)
        conditioned_yt = (y_t * coeff1 + y_x * (1 - coeff1)) * mask + (y_t * coeff2 + y_x * (1 - coeff2)) * (1 - mask)
        conditioned_x = Ainv(conditioned_yt)
        return conditioned_x

    def get_condition_t(
        self, 
        x, 
        y, 
        Afun, 
        t
    ):
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, y.shape) * y
        std = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, y.shape)
        z = torch.randn_like(x)
        Az = Afun(z)
        condition_t = mean + std * Az
        return condition_t
                
    def conditioned_p_sample_loop(
        self,
        model,
        shape,
        condition_kwargs,
        freq=1,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,        
    ):
        final = None
        for sample in self.conditioned_p_sample_loop_progressive(
            model,
            shape,
            condition_kwargs,
            freq=freq,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,            
        ):
            final = sample
        return final["sample"]
    
    def conditioned_p_sample_loop_progressive(
        self,
        model,
        shape,
        condition_kwargs,
        freq=1,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,        
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for step, i in enumerate(indices):
            t = torch.tensor([i] * shape[0], device=device)
            if (step % freq) == 0:
                img = self.conditioning(img, t=t, **condition_kwargs)
            with torch.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def conditioned_ddim_sample_loop(
        self,
        model,
        shape,
        condition_kwargs,
        freq=1,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.conditioned_ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            condition_kwargs=condition_kwargs,
            freq=freq,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def conditioned_ddim_sample_loop_progressive(
        self,
        model,
        shape,
        condition_kwargs,
        freq=1,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for step, i in enumerate(indices):
            t = torch.tensor([i] * shape[0], device=device)
            if (step % freq) == 0:
                img = self.conditioning(img, t=t, **condition_kwargs)
            with torch.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]


class SDEConditionSolver(SDEDiffusion):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def conditioning(
        self,
        x,
        y,
        Afun,
        Ainv,
        mask,
        coeff1,
        coeff2,
        t,
    ):
        y_t = self.get_condition_t(x, y, Afun, t)
        y_x = Afun(x)
        conditioned_yt = (y_t * coeff1 + y_x * (1 - coeff1)) * mask + (y_t * coeff2 + y_x * (1 - coeff2)) * (1 - mask)
        conditioned_x = Ainv(conditioned_yt)
        return conditioned_x

    def get_condition_t(
        self, 
        x, 
        y, 
        Afun, 
        t
    ):
        mean, std = self.sde.marginal_prob(y, t)
        z = torch.randn_like(x)
        Az = Afun(z)
        condition_t = mean + expand_dims(std, y.dim()) * Az
        return condition_t
        
    def conditioned_sample(
        self,
        model,
        shape,
        condition_kwargs,
        freq=1,
        noise=None,
        model_kwargs=None, 
        device=None,
        method="sde",
        sample_kwargs=None,        
    ):
        if sample_kwargs is None:
            sample_kwargs = {}

        if method.lower() =="sde":
            return self.conditioned_pc_sample(model, shape, condition_kwargs, freq, noise, model_kwargs, device, **sample_kwargs)

        # elif method.lower() =="ode":
        #     return self.ode_sample(model, shape, noise, model_kwargs, device, **sample_kwargs)

        else:
            raise NotImplementedError(f"Smpling method {method} not yet supported.")
    
    def conditioned_pc_sample(
        self, 
        model, 
        shape,
        condition_kwargs, 
        freq=1,
        noise=None,
        model_kwargs=None, 
        device=None,
        predictor = "euler",
        corrector = "langevin",
        corrector_steps = 1,
        progress=False,
        denoise=True,
    ):
        self.rsde = sde_lib.RSDE(self.sde, self.score_fn, probability_flow=False)
        if device is None:
            device = next(model.parameters()).device
        if noise is not None:
            x = noise
        else:
            x = self.sde.prior_sampling(shape).to(device)
        
        timesteps = torch.linspace(self.sde.T, self.time_eps, self.sde.N, device=device)
        indices = list(range(self.sde.N))
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        with torch.no_grad():
            for step, i in enumerate(indices):
                t = timesteps[i]
                vec_t = torch.ones(shape[0], device=t.device) * t
                if (step % freq) == 0:
                    x = self.conditioning(x, t=vec_t, **condition_kwargs)
                x = self.correct(model, x, vec_t, model_kwargs, corrector, corrector_steps)
                x = self.predict(model, x, vec_t, model_kwargs, predictor)

        return x
    

class DPMMCGSolver(DPM_Solver):
    def __init__(
        self,
        lamb_schedule,
        norm_weight,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lamb_schedule = lamb_schedule
        self.norm_weight = norm_weight

    def predict_x0_from_xt(
        self,
        xt,
        t,
        model_t,
    ):
        if self.algorithm_type == "dpmsolver++":
            return model_t
        else:
            ns = self.noise_schedule
            noise = model_t
            alpha_t, sigma_t = ns.marginal_alpha(t), ns.marginal_std(t)
            x0 = (xt - sigma_t * noise) / alpha_t
            if self.correcting_x0_fn is not None:
                x0 = self.correcting_x0_fn(x0, t)
            return x0
        
    def conditioning(
        self,
        x_t,
        x_s,
        y,
        Afun,
        Afun_t,
        Projection,
        norm_const,
        mask,
        coeff1,
        coeff2,
        t,
        model_s,
        noise_projection=False,
        **kwargs,
    ):
        lamb = self.lamb_schedule.get_current_lambda(t)
        norm_weight = self.norm_weight
        x0 = self.predict_x0_from_xt(x_t, t, model_s)
        y_error = (y - Afun(x0)) * mask
        # y_error = Projection(y_error)
        norm = torch.norm(y_error)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_s)[0]
        norm_grad = Projection(Afun(norm_grad) * (1. - mask))
        y_t = self.get_condition_t(x_t, y, Afun, t, noise_projection)
        y_t_error = y_t - Afun(x_t)
        correction_yt = y_t_error * (coeff1 * mask + coeff2 * (1. - mask))
        correction_x = Afun_t(correction_yt) / norm_const
        conditioned_x = x_t + lamb * correction_x - norm_weight * norm_grad
        conditioned_x = conditioned_x.detach()
        return conditioned_x

    def get_condition_t(
        self, 
        x, 
        y, 
        Afun,
        t,
        noise_projection,
    ):
        ns = self.noise_schedule
        log_mean_coeff = ns.marginal_log_mean_coeff(t)
        mean = torch.exp(expand_dims(log_mean_coeff, y.dim())) * y
        std = ns.marginal_std(t)
        if noise_projection:
            z = torch.randn_like(x)
            z = Afun(z)
        else:
            z = torch.randn_like(y)
        condition_t = mean + expand_dims(std, y.dim()) * z
        return condition_t
    
    def conditioned_dpm_solver_adaptive(
        self, 
        x,
        condition_kwargs,
        order, 
        t_T, 
        t_0, 
        h_init=0.05, 
        atol=0.0078, 
        rtol=0.05, 
        theta=0.9, 
        t_err=1e-5, 
        solver_type='dpmsolver',
        freq=1,
    ):
        ns = self.noise_schedule
        s = t_T * torch.ones((1,)).to(x)
        lambda_s = ns.marginal_lambda(s)
        lambda_0 = ns.marginal_lambda(t_0 * torch.ones_like(s).to(x))
        h = h_init * torch.ones_like(s).to(x)
        x_prev = x
        nfe = 0
        step = 0
        if order == 2:
            r1 = 0.5
            lower_update = lambda x, s, t: self.dpm_solver_first_update(x, s, t, return_intermediate=True)
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_second_update(x, s, t, r1=r1, solver_type=solver_type, **kwargs)
        elif order == 3:
            r1, r2 = 1. / 3., 2. / 3.
            lower_update = lambda x, s, t: self.singlestep_dpm_solver_second_update(x, s, t, r1=r1, return_intermediate=True, solver_type=solver_type)
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_third_update(x, s, t, r1=r1, r2=r2, solver_type=solver_type, **kwargs)
        else:
            raise ValueError("For adaptive step size solver, order must be 2 or 3, got {}".format(order))       
        while torch.abs((s - t_0)).mean() > t_err:
            t = ns.inverse_lambda(lambda_s + h)
            x.requires_grad_()
            x_lower, lower_noise_kwargs = lower_update(x, s, t)
            x_higher = higher_update(x, s, t, **lower_noise_kwargs)            
            delta = torch.max(torch.ones_like(x).to(x) * atol, rtol * torch.max(torch.abs(x_lower), torch.abs(x_prev)))
            norm_fn = lambda v: torch.sqrt(torch.square(v.reshape((v.shape[0], -1))).mean(dim=-1, keepdim=True))
            E = norm_fn((x_higher - x_lower) / delta).max()
            if (step % freq) == 0:
                x_higher = self.conditioning(x_t=x_higher, x_s=x, t=t, **condition_kwargs, **lower_noise_kwargs)
            else:
                x_higher = x_higher.detach()
            if torch.all(E <= 1.):
                x = x_higher
                s = t
                x_prev = x_lower
                lambda_s = ns.marginal_lambda(s)            
            h = torch.min(theta * h * torch.float_power(E, -1. / order).float(), lambda_0 - lambda_s)
            nfe += order
            step +=1
        print('adaptive solver nfe', nfe)
        return x
    
    def conditioned_sample(
        self, 
        x,
        condition_kwargs,
        freq=1,
        steps=20, 
        t_start=None, 
        t_end=None, 
        order=2, 
        skip_type='time_uniform',
        method='multistep', 
        lower_order_final=True, 
        denoise_to_zero=False, 
        solver_type='dpmsolver',
        atol=0.0078, 
        rtol=0.05, 
        return_intermediate=False,
    ):
        t_0 = 1. / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert t_0 > 0 and t_T > 0, "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        if return_intermediate:
            assert method in ['multistep', 'singlestep', 'singlestep_fixed'], "Cannot use adaptive solver when saving intermediate values"
        if self.correcting_xt_fn is not None:
            assert method in ['multistep', 'singlestep', 'singlestep_fixed'], "Cannot use adaptive solver when correcting_xt_fn is not None"
        device = x.device
        intermediates = []
        # with torch.no_grad():
        if method == 'adaptive':
            x = self.conditioned_dpm_solver_adaptive(x, condition_kwargs, order=order, t_T=t_T, t_0=t_0, atol=atol, rtol=rtol, solver_type=solver_type, freq=freq)
        elif method == 'multistep':
            assert steps >= order
            timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
            assert timesteps.shape[0] - 1 == steps
            # Init the initial values.
            step = 0
            t = timesteps[step]                
            t_prev_list = [t]
            model_prev_list = [self.model_fn(x, t)]
            if self.correcting_xt_fn is not None:
                x = self.correcting_xt_fn(x, t, step)
            if return_intermediate:
                intermediates.append(x)
            # Init the first `order` values by lower order multistep DPM-Solver.
            for step in range(1, order):
                t = timesteps[step]
                x.requires_grad_()
                out = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, step, solver_type=solver_type)
                if self.correcting_xt_fn is not None:
                    out = self.correcting_xt_fn(out, t, step)
                if (step % freq) == 0:
                    x = self.conditioning(x_t=out, x_s=x, t=t, **condition_kwargs, model_s=model_prev_list[-1])
                else:
                    x = out.detach()
                if return_intermediate:
                    intermediates.append(x)                    
                t_prev_list.append(t)
                model_prev_list.append(self.model_fn(x, t))
                
            # Compute the remaining values by `order`-th order multistep DPM-Solver.
            for step in range(order, steps + 1):
                t = timesteps[step]
                x.requires_grad_()
                # We only use lower order for steps < 10
                if lower_order_final and steps < 10:
                    step_order = min(order, steps + 1 - step)
                else:
                    step_order = order
                out = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, step_order, solver_type=solver_type)
                if self.correcting_xt_fn is not None:
                    out = self.correcting_xt_fn(out, t, step)
                if (step % freq) == 0:
                    x = self.conditioning(x_t=out, x_s=x, t=t, **condition_kwargs, model_s=model_prev_list[-1])
                else:
                    x = out.detach()
                if return_intermediate:
                    intermediates.append(x)
                for i in range(order - 1):
                    t_prev_list[i] = t_prev_list[i + 1]
                    model_prev_list[i] = model_prev_list[i + 1]
                t_prev_list[-1] = t
                # We do not need to evaluate the final model value.
                if step < steps:
                    model_prev_list[-1] = self.model_fn(x, t)
        elif method in ['singlestep', 'singlestep_fixed']:
            if method == 'singlestep':
                timesteps_outer, orders = self.get_orders_and_timesteps_for_singlestep_solver(steps=steps, order=order, skip_type=skip_type, t_T=t_T, t_0=t_0, device=device)
            elif method == 'singlestep_fixed':
                K = steps // order
                orders = [order,] * K
                timesteps_outer = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=K, device=device)
            for step, order in enumerate(orders):
                s, t = timesteps_outer[step], timesteps_outer[step + 1]
                x.requires_grad_()
                timesteps_inner = self.get_time_steps(skip_type=skip_type, t_T=s.item(), t_0=t.item(), N=order, device=device)
                lambda_inner = self.noise_schedule.marginal_lambda(timesteps_inner)
                h = lambda_inner[-1] - lambda_inner[0]
                r1 = None if order <= 1 else (lambda_inner[1] - lambda_inner[0]) / h
                r2 = None if order <= 2 else (lambda_inner[2] - lambda_inner[0]) / h
                out, noise_kwargs = self.singlestep_dpm_solver_update(x, s, t, order, return_intermediate=True, solver_type=solver_type, r1=r1, r2=r2)                    
                if self.correcting_xt_fn is not None:
                    out = self.correcting_xt_fn(out, t, step)
                if (step % freq) == 0:
                    x = self.conditioning(x_t=out, x_s=x, t=t, **condition_kwargs, **noise_kwargs)
                else:
                    x = out.detach()
                if return_intermediate:
                    intermediates.append(x)
        else:
            raise ValueError("Got wrong method {}".format(method))
        if denoise_to_zero:
            t = torch.ones((1,)).to(device) * t_0
            x = self.denoise_to_zero_fn(x, t)
            if self.correcting_xt_fn is not None:
                x = self.correcting_xt_fn(x, t, step + 1)
            if return_intermediate:
                intermediates.append(x)
        if return_intermediate:
            return x, intermediates
        else:
            return x

def CG(x0, y, H, H_t, max_iter, tol=1e-4):
    # min _x 1/2 * ||Hx-y||_2^2
    x = x0.clone()
    grad_old = None
    for i in range(max_iter):
        u = x.clone()
        tmp = H(x) - y
        grad = H_t(tmp)
        if i == 0:
            d = - grad
        else:
            beta = (grad ** 2).sum((2,3), keepdim=True) / (grad_old ** 2).sum((2,3), keepdim=True).clamp_(min=1e-8)
            d = - grad + beta * d
        grad_old = grad
        Hd = H(d)
        step = - (grad * d).sum((2,3), keepdim=True) / (Hd ** 2).sum((2,3), keepdim=True).clamp_(min=1e-8)
        x = x + step * d
        error = ((x - u) ** 2).sum() / x0.size(0)
        if error < tol:
            break
    return x        

class GaussianMCGSolver(SpacedDiffusion):
    def __init__(
        self,
        lamb_schedule,
        norm_weight,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lamb_schedule = lamb_schedule
        self.norm_weight = norm_weight
        
    def conditioning(
        self,
        x_t,
        x_prev,
        x0,
        y,
        Afun_low,
        Afun_t_low,
        Afun_inv_low,
        Afun_high,
        Afun_inv_high,
        norm_const,
        mask,
        t,
        noise_projection=False,
    ):
        lamb = self.lamb_schedule.get_current_lambda(t)
        lamb = expand_dims(lamb, x_t.dim())
        # norm_weight = self.norm_weight
        # y_error = (y - Afun_low(x0))
        # x_error = Afun_inv_low(y_error)
        # norm = torch.norm(x_error)
        # norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0] * norm_weight
        # norm_grad = Afun_inv_high(Afun_high(norm_grad) * (1. - mask))
        # y_t = self.get_condition_t(x_t, y, Afun_low, t, noise_projection)
        # conditioned_x = CG(x_t, y, Afun_low, Afun_t_low, 5) - norm_grad
        y_t = y
        y_t_error = y_t - Afun_low(x_t)
        correction_x = Afun_t_low(y_t_error) / norm_const
        conditioned_x = x_t + lamb * correction_x
        # conditioned_x = x_t + lamb * correction_x - norm_grad
        conditioned_x = conditioned_x.detach()
        return conditioned_x

    def get_condition_t(
        self, 
        x, 
        y, 
        Afun,
        t,
        noise_projection,
    ):
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, y.shape) * y
        std = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, y.shape)
        if noise_projection:
            z = torch.randn_like(x)
            z = Afun(z)
        else:
            z = torch.randn_like(y)
        condition_t = mean + std * z
        return condition_t
                
    def conditioned_p_sample_loop(
        self,
        model,
        shape,
        condition_kwargs,
        freq=1,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,        
    ):
        final = None
        for sample in self.conditioned_p_sample_loop_progressive(
            model,
            shape,
            condition_kwargs,
            freq=freq,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,            
        ):
            final = sample
        return final["sample"]
    
    def conditioned_p_sample_loop_progressive(
        self,
        model,
        shape,
        condition_kwargs,
        freq=1,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        t_old = 1
        x_old = img.clone()
        y = img.clone()
        for step, i in enumerate(indices):
            t = torch.tensor([i] * shape[0], device=device)
            # img.requires_grad_()
            t_new = (1 + np.sqrt(1 + t_old ** 2)) / 2
            if (step % freq) == 0:
                img = self.conditioning(x_t=y, x_prev=None, x0=None, t=t, **condition_kwargs)
            with torch.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
            # if (step % freq) == 0:
            #     out["sample"] = self.conditioning(x_t=out["sample"], x_prev=img, x0=out["pred_xstart"], t=t, **condition_kwargs)

            yield out
            x_new = out["sample"].detach()
            y = x_new + (t_old - 1) / t_new * (x_new - x_old)
            x_old = x_new
            # img = out["sample"].detach()

    def conditioned_ddim_sample_loop(
        self,
        model,
        shape,
        condition_kwargs,
        freq=1,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.conditioned_ddim_sample_loop_progressive(
            model,
            shape,
            condition_kwargs=condition_kwargs,
            freq=freq,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def conditioned_ddim_sample_loop_progressive(
        self,
        model,
        shape,
        condition_kwargs,
        freq=1,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for step, i in enumerate(indices):
            t = torch.tensor([i] * shape[0], device=device)
            img.requires_grad_()
            out = self.ddim_sample(
                model,
                img,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                eta=eta,
            )
            if (step % freq) == 0:
                out["sample"] = self.conditioning(x_t=out["sample"], x_prev=img, x0=out["pred_xstart"], t=t, **condition_kwargs)
            yield out
            img = out["sample"]
            

class SDEMCGSolver(SDEDiffusion):
    def __init__(
        self,
        lamb_schedule,
        norm_weight,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lamb_schedule = lamb_schedule
        self.norm_weight = norm_weight
        
    def predict_x0_from_xt(
        self,
        xt,
        t,
        score,
    ):
        mean_coef, std = self.sde.marginal_prob(torch.ones_like(xt), t)
        x0 = (xt + (std ** 2) * score) / mean_coef
        return x0

        
    def conditioning(
        self,
        x_t,
        x_prev,
        y,
        Afun,
        Afun_t,
        Projection,
        norm_const,
        mask,
        coeff1,
        coeff2,
        t,
        score,
        noise_projection=False,
    ):
        lamb = self.lamb_schedule.get_current_lambda(t)
        norm_weight = self.norm_weight
        x0 = self.predict_x0_from_xt(x_t, t, score)
        y_error = (y - Afun(x0)) * mask
        # y_error = Projection(y_error)
        norm = torch.norm(y_error)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        norm_grad = Projection(Afun(norm_grad) * (1. - mask))
        y_t = self.get_condition_t(x_t, y, Afun, t, noise_projection)
        y_t_error = y_t - Afun(x_t)
        correction_yt = y_t_error * (coeff1 * mask + coeff2 * (1. - mask))
        correction_x = Afun_t(correction_yt) / norm_const
        conditioned_x = x_t + lamb * correction_x - norm_weight * norm_grad
        conditioned_x = conditioned_x.detach()
        return conditioned_x

    def get_condition_t(
        self, 
        x, 
        y, 
        Afun,
        t,
        noise_projection,
    ):
        mean, std = self.sde.marginal_prob(y, t)
        if noise_projection:
            z = torch.randn_like(x)
            z = Afun(z)
        else:
            z = torch.randn_like(y)
        condition_t = mean + expand_dims(std, y.dim()) * z
        return condition_t
        
    def conditioned_sample(
        self,
        model,
        shape,
        condition_kwargs,
        freq=1,
        noise=None,
        model_kwargs=None, 
        device=None,
        method="sde",
        sample_kwargs=None,        
    ):
        if sample_kwargs is None:
            sample_kwargs = {}

        if method.lower() =="sde":
            return self.conditioned_pc_sample(model, shape, condition_kwargs, freq, noise, model_kwargs, device, **sample_kwargs)

        else:
            raise NotImplementedError(f"Smpling method {method} not yet supported.")
    
    def conditioned_pc_sample(
        self, 
        model, 
        shape,
        condition_kwargs, 
        freq=1,
        noise=None,
        model_kwargs=None, 
        device=None,
        predictor = "euler",
        corrector = "langevin",
        corrector_steps = 1,
        progress=False,
        denoise=True,
    ):
        self.rsde = sde_lib.RSDE(self.sde, self.score_fn, probability_flow=False)
        if device is None:
            device = next(model.parameters()).device
        if noise is not None:
            x = noise
        else:
            x = self.sde.prior_sampling(shape).to(device)
        
        timesteps = torch.linspace(self.sde.T, self.time_eps, self.sde.N, device=device)
        indices = list(range(self.sde.N))
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        for step, i in enumerate(indices):
            t = timesteps[i]
            vec_t = torch.ones(shape[0], device=t.device) * t                
            x = self.correct(model, x, vec_t, model_kwargs, corrector, corrector_steps)
            x.requires_grad_()
            score = self.score_fn(model, x, vec_t, model_kwargs)
            out = self.predict(model, x, vec_t, model_kwargs, predictor)
            if (step % freq) == 0:
                x = self.conditioning(x_t=out, x_prev=x, score=score, t=vec_t, **condition_kwargs)
            else:
                x = out.detach()

        return x
    

class lambda_schedule:
    def __init__(self, T=1000):
        self.T = T

    def get_current_lambda(self, t):
        pass


class lambda_schedule_linear(lambda_schedule):
    def __init__(self, start_lamb=1.0, end_lamb=0.0):
        super().__init__()
        self.start_lamb = start_lamb
        self.end_lamb = end_lamb

    def get_current_lambda(self, t):
        return self.start_lamb + (self.end_lamb - self.start_lamb) * (1 - t / self.T)


class lambda_schedule_const(lambda_schedule):
    def __init__(self, lamb=1.0):
        super().__init__()
        self.lamb = lamb

    def get_current_lambda(self, t):
        return self.lamb * 1.0 + t * 0.