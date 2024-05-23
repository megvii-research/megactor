
import torch.nn.functional as F
vlb_loss_func = None

def vlb_mse(model_output, x_t, x_start, target, timestep, config):
    global vlb_loss_func
    if vlb_loss_func is None:
        from animatediff.utils.variational_lower_bound_loss import VariationalLowerBoundLoss
        num_train_timesteps = config["noise_scheduler_kwargs"]["num_train_timesteps"] if "num_train_timesteps" in config["noise_scheduler_kwargs"] else 1000
        vlb_loss_func = VariationalLowerBoundLoss(
            num_train_timesteps = num_train_timesteps,
            beta_start = config["noise_scheduler_kwargs"]["beta_start"],
            beta_end = config["noise_scheduler_kwargs"]["beta_end"],
            beta_schedule = config["noise_scheduler_kwargs"]["beta_schedule"],
        )
    mse_loss_val = F.mse_loss(model_output.float(), target.float(), reduction="mean") 
    vlb_loss_val = vlb_loss_func(model_output, x_t, x_start, timestep).mean()
    return mse_loss_val * 0.5 + vlb_loss_val * 0.5
 
def mse(model_output, x_t, x_start, target, timestep, config):
    mse_loss_val = F.mse_loss(model_output.float(), target.float(), reduction="mean") 
    return mse_loss_val