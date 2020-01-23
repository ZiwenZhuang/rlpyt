""" The entry point to start repeated pearl experiment.
"""

from exptools.launching.variant import VariantLevel, make_variants, update_config
from exptools.launching.affinity import encode_affinity
from exptools.launching.exp_launcher import run_experiments

def get_default_config():
    return dict(
        env= dict(
            name= "hopper", # choose between: "hopper", "pr2", "walker"
        ),
        tasks= dict(
            n_train_tasks= 20,
            n_eval_tasks= 4,
        ),
        sampler= dict(
            batch_T= int(1e3),
            batch_B= 1,
            infer_context_period= 100,
            eval_max_steps=int(1e3),
            eval_n_envs_per_task= 1,
        ),
        algo= dict(
            discount= 0.99,
            batch_size= 256,
            target_update_tau= 0.005,
            learning_rate= 3e-4,
            min_steps_learn= int(1e4),
            n_step_return= 1,
            n_tasks_per_update= 5,
            optim_kwargs= dict(),
            reward_scale= 5.,
            bootstrap_timelimit= False, # currently, I didn't figure out what it means.
        ),
        agent= dict(
            latent_size= 5,
            encoder_model_kwargs= dict(
                hidden_sizes= [300, 300, 300],
                use_information_bottleneck= True,
            ),
            model_kwargs= dict(hidden_sizes= [300, 300, 300]),
            q_model_kwargs= dict(hidden_sizes= [300, 300, 300]),
            v_model_kwargs= dict(hidden_sizes= [300, 300, 300]),
        )
    )

def main(args):
    experiment_title = "pearl_reproduction"
    affinity_code = encode_affinity(
        n_cpu_core= 32,
        n_gpu= 8,
        gpu_per_run= 1,
        contexts_per_gpu= 1,
    )
    default_config = get_default_config()

    # set up variants
    variant_levels = list()

    values = [
        ["hopper",],
        ["pr2",],
        # ["walker",],
    ]
    dir_names = ["{}".format(*v) for v in values]
    keys = [("env", "name")]
    variant_levels.append(VariantLevel(keys, values, dir_names))

    values = [
        [int(2e3),],
        [int(0),],
    ]
    dir_names = ["{}steps_pre_learning".format(*v) for v in values]
    keys = [("algo", "min_steps_learn")]
    variant_levels.append(VariantLevel(keys, values, dir_names))
    
    values = [
        [3e-4,],
        # [3e-5,],
        [3e-10,],
        [3e-16,],
    ]
    dir_names = ["lr{}".format(*v) for v in values]
    keys = [("algo", "learning_rate")]
    variant_levels.append(VariantLevel(keys, values, dir_names))
    
    values = [
        # [5,],
        [10,],
    ]
    dir_names = ["meta_batch{}".format(*v) for v in values]
    keys = [("algo", "n_tasks_per_update")]
    variant_levels.append(VariantLevel(keys, values, dir_names))

    values = [
        # [5,],
        [10,],
        [20,],
    ]
    dir_names = ["batch_B{}".format(*v) for v in values]
    keys = [("sampler", "batch_B")]
    variant_levels.append(VariantLevel(keys, values, dir_names))

    variants, log_dirs = make_variants(*variant_levels)
    for i, variant in enumerate(variants):
        variants[i] = update_config(default_config, variant)

    # setup for some debug option
    if args.debug:
        for variant in variants:
            variant["algo"]["min_steps_learn"]=int(2e2)
            variant["algo"]["replay_ratio"]=128

    run_experiments(
        script="rlpyt/experiments/PEARL/pearl_experiment.py",
        affinity_code=affinity_code,
        experiment_title=experiment_title+("--debug" if args.debug else ""),
        runs_per_setting=1,
        variants=variants,
        log_dirs=log_dirs,
        debug_mode=args.debug,
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--debug', help= 'A common setting of whether to entering debug mode for remote attach',
        type= int, default= 0,
    )

    args = parser.parse_args()
    if args.debug > 0:
        # configuration for remote attach and debug
        import ptvsd
        import sys
        ip_address = ('0.0.0.0', 5050)
        print("Process: " + " ".join(sys.argv[:]))
        print("Is waiting for attach at address: %s:%d" % ip_address, flush= True)
        # Allow other computers to attach to ptvsd at this IP address and port.
        ptvsd.enable_attach(address=ip_address, redirect_output= True)
        # Pause the program until a remote debugger is attached
        ptvsd.wait_for_attach()
        print("Process attached, start running into experiment...", flush= True)
        ptvsd.break_into_debugger()

    main(args)