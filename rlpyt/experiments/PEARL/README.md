# PEARL as well as the first meta RL algorithm

## Additional interface required

### Agent

* Stores `.encoder_model_kwargs["use_information_bottleneck"]` attribute after initialization.

* Function `.infer_posterior(context)` to learn the task, and return the latent value in batch.

    - `Context` is defined in `rlpyt.samplers.collections`

* Contains `.latent_z` attribute after each `.infer_posterior(context)` call.

* Contains `.compute_latent_KL()` function in torch version return the KL divergence of the prediction (in batch) if its `.encoder_model_kwargs["use_information_bottleneck"]` is `True`.