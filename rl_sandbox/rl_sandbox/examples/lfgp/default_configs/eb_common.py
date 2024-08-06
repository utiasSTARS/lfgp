import rl_sandbox.constants as c
import rl_sandbox.examples.lfgp.default_configs.dac as default_dac
import rl_sandbox.examples.lfgp.default_configs.lfgp as default_lfgp
import rl_sandbox.examples.lfgp.experiment_utils as exp_utils


def get_parser():
    parser = default_lfgp.get_parser()  # same for single task and multitask

    # Example based
    parser.add_argument('--expert_data_mode', type=str, default="obs_only_no_next", help="options are [obs_act, obs_only, obs_only_no_next].")

    # Expert sampling for training critic and policy, in addition to discriminator (if we're using one)
    parser.add_argument('--expbuf_model_sample_rate', type=float, default=0.0,
        help="Proportion of mini-batch samples that should be expert samples for q/policy training.")
    parser.add_argument('--expbuf_model_sample_decay', type=float, default=1.0,
        help="Decay rate for expbuf_model_sample_rate. .99999 brings close to 0 at 1M.")
    parser.add_argument('--expbuf_model_train_mode', type=str, default='both',
        help="Whether expert data trains the critic, or both the actor and critic. Options: [both, critic_only]")
    parser.add_argument('--full_traj_expert_filenames', type=str, required=False,
                    help="Expert filenames for full trajectories, to use in addition to final timesteps.")
    parser.add_argument('--ft_expert_dir_rest', type=str, default='expert_data/full_trajectories/200_per_task')
    parser.add_argument('--add_default_full_traj', action='store_true',
                    help="If set, add the default expert trajectories as defined in env_default_configs.py")

    # n step
    parser.add_argument('--expbuf_last_sample_prop', type=float, default=0.0,
        help="Proportion of mini-batch samples that should be final transitions for discriminator training. 0.0 \
            means regular sampling.")

    # RCE/SQIL
    parser.add_argument('--nth_q_targ_multiplier', type=float, default=.5, help="applies to nth_q_targ n_step, .5 is value used in RCE.")
    parser.add_argument('--sqil_rce_bootstrap_expert_mode', type=str, default="no_boot",
                        help="If boot, sqil and rce bootstrap on expert dones (unlike RCE implementation). no_boot"\
                            " means no bootstrapping on expert dones (but bootstrapping on non-expert handled by no_bootstrap_on_done)")
    parser.add_argument('--q_type', type=str, default="raw", help="Options: [raw, classifier]")
    parser.add_argument('--q_over_max_penalty', type=float, default=0.0,
                        help="If set, a multiplier on the q magnitude over the max possible q based on max reward of 1, "\
                             "using reward_scaling and discount_factor")
    parser.add_argument('--qomp_num_med_filt', type=int, default=50,
                        help="For q over max penalty + discriminator reward, how many max discrim values to use for "
                             "median filter estimate of true discrim max.")
    parser.add_argument('--qomp_policy_max_type', type=str, choices=['max_exp', 'avg_exp'], default='max_exp',
                        help="Whether to use average expert or max expert for q max penalty.")
    parser.add_argument('--q_expert_target_mode', type=str, choices=['max', 'bootstrap'],
                        help="if we're training with sqil_bootstrap_expert_mode set to boot, and a non classifier q, "
                             "chooses whether q targets are bootstrapped gradually, or directly set to max.")
    parser.add_argument('--noise_zero_target_mode', type=str, choices=['none', 'range', 'per_obs'], default='none',
                        help="Generate fake obs/act pairs to label as zero q if not none. 'range' does uniform sampling "
                             "of whole obs range, 'per_obs' takes current buffer obs and adds uniform noise scaled "
                             "by obs range + param below.")
    parser.add_argument('--nzt_per_obs_scale', type=float, default=0.1, help="Scale param for per_obs described above.")
    parser.add_argument('--sqil_policy_reward_label', type=float, choices=[0.0, -1.0], default=0.0,
                        help="Reward label for policy data in SQIL, if not using classifier.")

    return parser

def get_settings(args):
    # some settings to keep things compatible for SQIL and RCE
    if args.reward_model in ['sqil', 'rce']:
        if args.expbuf_model_sample_rate == 0.0:
            print('------------------')
            print(f"WARNING: Reward model {args.reward_model} but requested exp buf model rate of "\
                f"{args.expbuf_model_sample_rate}, must be above 0, setting to 0.5.")
            args.expbuf_model_sample_rate = 0.5
            print('------------------')
        if args.expbuf_model_sample_decay != 1.0:
            print('------------------')
            print(f"WARNING: Reward model {args.reward_model} but requested exp buf model decay of "\
                f"{args.expbuf_model_sample_decay}, setting to 1.0.")
            args.expbuf_model_sample_decay = 1.0
            print('------------------')

    if args.reward_model == 'rce':
        if args.q_type != "classifier":
            print('------------------')
            print(f"WARNING: Q type not set as classifier, setting to classifier to match RCE.")
            args.q_type = "classifier"
            print('------------------')
        if args.sqil_rce_bootstrap_expert_mode == 'boot':
            print('------------------')
            print(f"WARNING: sqil_rce_bootstrap_expert_mode set to 'boot', setting to 'no_boot' to match RCE settings.")
            args.sqil_rce_bootstrap_expert_mode = 'no_boot'
            print('------------------')

    if args.q_type == 'classifier' and not args.no_entropy_in_qloss:
        print('------------------')
        print(f"WARNING: classifier q_type set, turning off entropy in qloss.")
        args.no_entropy_in_qloss = True
        print('------------------')

    if args.single_task:
        experiment_setting = default_dac.get_settings(args)
    else:
        experiment_setting = default_lfgp.get_settings(args)

    experiment_setting[c.RCE_EPS] = 1e-7
    experiment_setting[c.SQIL_RCE_BOOTSTRAP_EXPERT_MODE] = args.sqil_rce_bootstrap_expert_mode
    experiment_setting[c.Q_OVER_MAX_PENALTY] = args.q_over_max_penalty
    experiment_setting["threshold_discriminator"] = args.threshold_discriminator
    experiment_setting[c.Q_REGULARIZER] = args.q_regularizer
    experiment_setting[c.QOMP_NUM_MED_FILT] = args.qomp_num_med_filt
    experiment_setting[c.QOMP_POLICY_MAX_TYPE] = args.qomp_policy_max_type
    experiment_setting[c.Q_EXPERT_TARGET_MODE] = args.q_expert_target_mode
    experiment_setting[c.NOISE_ZERO_TARGET_MODE] = args.noise_zero_target_mode
    experiment_setting[c.NZT_PER_OBS_SCALE] = args.nzt_per_obs_scale
    experiment_setting[c.SQIL_POLICY_REWARD_LABEL] = args.sqil_policy_reward_label

    if args.single_task:
        experiment_setting[c.MODEL_SETTING][c.KWARGS][c.CLASSIFIER_OUTPUT] = args.q_type == 'classifier'
    else:
        experiment_setting[c.INTENTIONS_SETTING][c.KWARGS][c.CLASSIFIER_OUTPUT] = args.q_type == 'classifier'

    prefix = "multi-" if not args.single_task else ""
    if args.env_type == c.MANIPULATOR_LEARNING:
        task_save_name = args.main_task
    else:
        task_save_name = args.env_name

    experiment_setting[c.SAVE_PATH] = exp_utils.get_save_path(
        f"{prefix}{args.reward_model[:4]}", task_save_name, args.seed, args.exp_name, args.top_save_path)

    return experiment_setting
