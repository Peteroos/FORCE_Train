import inspect
from diffusion.edm import registered_sdes, EDMDiffusion
from .script_util import create_model

def sde_defaults():
    return dict(
        beta_min=0.1,
        beta_max=20,
        eps=1e-5,
        sigma_min=0.001,
        sigma_max=80,
        P_mean=-1.2, 
        P_std=1.2, 
        sigma_data=0.5,
        rho=7,
        sde_type="vpsde",
        diffusion_steps=1000,
        reduce_mean=True,
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

def create_sde(sde_type, reduce_mean, **kwargs):
    sde_type = sde_type.lower()
    if sde_type not in registered_sdes:
        raise NotImplementedError(f"SDE {sde_type} is unknown. Available options: {list(registered_sdes.keys())}")
    sde_class = registered_sdes[sde_type]
    constructor_signature = inspect.signature(sde_class.__init__)
    filtered_kwargs = {
        key: value for key, value in kwargs.items() if key in constructor_signature.parameters
    }
    sde = sde_class(**filtered_kwargs)
    diffusion = EDMDiffusion(
        sde=sde,
        reduce_mean=reduce_mean,
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
    eps,
    sigma_min,
    sigma_max,
    P_mean,
    P_std,
    sigma_data,
    rho,
    sde_type,
    diffusion_steps,
    reduce_mean,
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
        beta_min=beta_min,
        beta_max=beta_max,
        eps=eps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        P_mean=P_mean,
        P_std=P_std,
        sigma_data=sigma_data,
        rho=rho,
        sde_type=sde_type,
        diffusion_steps=diffusion_steps,
        reduce_mean=reduce_mean,
    )
    return model, sde

