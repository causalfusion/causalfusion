# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

import enum

import numpy as np
import torch


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.
    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()


def get_named_beta_schedule(num_diffusion_timesteps):
    # Linear schedule from Ho et al, extended to work for any number of
    # diffusion steps.
    scale = 1000 / num_diffusion_timesteps
    return np.linspace(
        start=scale * 0.0001,
        stop=scale * 0.02,
        num=num_diffusion_timesteps,
        dtype=np.float64
    )


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.
    """

    def __init__(self, *, betas, model_var_type):
        self.model_var_type = model_var_type

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 
        # at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        ) if len(self.posterior_variance) > 1 else np.array([])

        self.posterior_mean_coef1 = (
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, xn, t, model_kwargs):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        Args:
            xn: the tensor at time t, has a shape of [N, Lq, D].
            t: a 1-D Tensor of timesteps.
            model_kwargs: a dict contains pos_emb, kv_cache, cfg_scale.
        Return: a dict with the following keys:
            - 'mean': the model mean output.
            - 'variance': the model variance output.
            - 'log_variance': the log of 'variance'.
            - 'pred_xstart': the prediction for x_0.
        """
        pred_noise = model(xn, t, **model_kwargs)

        model_variance, model_log_variance = {
            # for fixedlarge, we set the initial (log-)variance like so
            # to get a better decoder log likelihood.
            ModelVarType.FIXED_LARGE: (
                np.append(self.posterior_variance[1], self.betas[1:]),
                np.log(np.append(self.posterior_variance[1], self.betas[1:])),
            ),
            ModelVarType.FIXED_SMALL: (
                self.posterior_variance,
                self.posterior_log_variance_clipped,
            ),
        }[self.model_var_type]
        model_variance = _extract_into_tensor(model_variance, t, xn.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, xn.shape)

        pred_xstart = self._predict_xstart_from_eps(x_t=xn, t=t, eps=pred_noise)
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=xn, t=t)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == xn.shape
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def p_sample(self, model, xn, t, kv_cache, pos_embed, cfg_scale):
        """
        Sample x_{t-1} from the model at the given timestep.
        Args:
            model: the model to sample from.
            xn: the current tensor at x_{t-1}.
            pos_embed: pos_embed of xn
            t: the value of t, starting at 0 for the first diffusion step.
            kv_cache: a dict.
        Return: 
            a dict containing the following keys:
                - 'sample': a random sample from the model.
                - 'pred_xstart': a prediction of x_0.
        """
        model_kwargs = dict(pos_embed=pos_embed, kv_cache=kv_cache, cfg_scale=cfg_scale)
        out = self.p_mean_variance(model, xn, t, model_kwargs)
        noise = torch.randn_like(xn)
        # no noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(xn.shape) - 1)))
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(self, model, xn, kv_cache, pos_embed, cfg_scale=1.0):
        """
        Generate samples from the model.
        Args:
            model: the model module.
            xn: Lq noise latent.
            pos_embed: pos_embed of xn
            kv_cache: a dict.
        Return: 
            a non-differentiable batch of samples.
        """
        final = None
        for sample in self._p_sample_loop(model, xn, kv_cache, pos_embed, cfg_scale):
            final = sample
        return final["sample"]

    def _p_sample_loop(self, model, xn, kv_cache, pos_embed, cfg_scale):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        indices = list(range(self.num_timesteps))[::-1]
        for i in indices:
            t = torch.tensor([i] * xn.shape[0], device=xn.device)
            with torch.no_grad():
                out = self.p_sample(model, xn, t, kv_cache, pos_embed, cfg_scale)
                yield out
                xn = out["sample"]

    def training_losses(
        self, model, x_start, t, y, attn_mask, last_split_size, noise=None
    ):
        """
        Compute training losses for a single timestep.
        Args:
            model: the model to evaluate loss on.
            x_start: the [N, C, ...] tensor of inputs.
            t: a batch of timestep indices.
            y: a batch of label indices.
            attn_mask: generalized causal attn mask.
            noise: if specified, the specific Gaussian noise to try to remove.
        Return: 
            a dict with the key "loss" containing a tensor of shape [N].
            Some mean or variance settings may also have other keys.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}
        mse = model(
            x_t, t, x_start, y, 
            attn_mask=attn_mask, 
            last_split_size=last_split_size, 
            noise=noise
        )
        terms["mse"] = mse
        terms["loss"] = terms["mse"]

        return terms

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)
