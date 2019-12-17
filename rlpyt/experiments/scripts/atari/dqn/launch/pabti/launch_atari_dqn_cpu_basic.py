
from exptools.launching.affinity import encode_affinity
from exptools.launching.exp_launcher import run_experiments
from exptools.launching.variant import make_variants, VariantLevel

script = "rlpyt/experiments/scripts/atari/dqn/train/atari_dqn_cpu.py"
affinity_code = encode_affinity(
    n_cpu_core=24,
    n_gpu=6,
    # hyperthread_offset=24,
    n_socket=2,
    # cpu_per_run=2,
)
runs_per_setting = 2
experiment_title = "atari_dqn_basic_cpu"
variant_levels = list()

games = ["pong", "qbert", "chopper_command"]
values = list(zip(games))
dir_names = ["{}".format(*v) for v in values]
keys = [("env", "game")]
variant_levels.append(VariantLevel(keys, values, dir_names))

variants, log_dirs = make_variants(*variant_levels)

default_config_key = "dqn"

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_config_key,),
)
