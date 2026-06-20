from diffusion import sde_lib
from diffusion.sde import SDEDiffusion
from .script_util import create_model

def sde_defaults():
    return dict(
        # snr=0.16,
        beta_min=0.1,
        beta_max=20,
        sde_type="vpsde",
        diffusion_steps=1000,
        continuous=True,
        reduce_mean=True,
        likelihood_weighting=False,
    )

def model_and_sde_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        image_size=64,
        in_channels=1,
        num_channels=64,
        out_channels=1,
        num_res_blocks=2,
        num_heads=1,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="",
        channel_mult="",
        dropout=0.0,
        dims=2,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=False,
        resblock_updown=False,
        # use_fp16=False,
        use_new_attention_order=False,
    )
    res.update(sde_defaults())
    return res

def create_sde(
    beta_min,
    beta_max,
    sde_type="vpsde",
    diffusion_steps=1000,
    continuous=True,
    reduce_mean=True,
    likelihood_weighting=False,
):
    if sde_type.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=beta_min, beta_max=beta_max, N=diffusion_steps)
        sampling_eps = 1e-3
    elif sde_type.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=beta_min, beta_max=beta_max, N=diffusion_steps)
        sampling_eps = 1e-3
    elif sde_type.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=beta_min, sigma_max=beta_max, N=diffusion_steps)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {sde_type} unknown.")

    diffusion = SDEDiffusion(
        sde=sde,
        continuous=continuous,
        reduce_mean=reduce_mean,
        time_eps = sampling_eps,
        likelihood_weighting = likelihood_weighting,
    )
    return diffusion

def create_model_and_sde(
    image_size,
    class_cond,
    in_channels,
    num_channels,
    out_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    dims,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_new_attention_order,
    beta_min,
    beta_max,
    sde_type,
    diffusion_steps,
    continuous,
    reduce_mean,
    likelihood_weighting,
):
    model = create_model(
        image_size,
        in_channels,
        num_channels,
        out_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
        dims=dims,
    )
    sde = create_sde(
        beta_min,
        beta_max,
        sde_type=sde_type,
        diffusion_steps=diffusion_steps,
        continuous=continuous,
        reduce_mean=reduce_mean,
        likelihood_weighting=likelihood_weighting,
    )
    return model, sde