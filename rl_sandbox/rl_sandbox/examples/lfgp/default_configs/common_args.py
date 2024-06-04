# import argparse
import configargparse


def get_parser():
    # parser = argparse.ArgumentParser(conflict_handler='resolve')  # arguments will be overwritten by other files
    parser = configargparse.ArgumentParser(conflict_handler='resolve')  # arguments will be overwritten by other files

    # RL
    parser.add_argument('--seed', type=int, default=100, help="Random seed")
    parser.add_argument('--device', type=str, default="cpu", help="device to use")
    parser.add_argument('--render', action='store_true', default=False, help="Render training")
    parser.add_argument('--eval_render', action='store_true', default=False, help="Render evaluation")
    parser.add_argument('--max_steps', type=int, default=2000000, help="Number of steps to interact with")
    parser.add_argument('--memory_size', type=int, default=2000000, help="Memory size of buffer")
    parser.add_argument('--eval_freq', type=int, default=100000, help="The frequency of evaluating the performance of the current policy")
    parser.add_argument('--num_evals_per_task', type=int, default=50, help="Number of evaluation episodes per task")
    parser.add_argument('--print_interval', type=int, default=5000, help="Print interval for terminal stats.")
    parser.add_argument('--log_interval', type=int, default=5000, help="Log interval for tensorboard.")
    parser.add_argument('--save_interval', type=int, default=200000, help="Model save interval.")
    parser.add_argument('--buffer_warmup', type=int, default=25000, help="Buffer warmup before starting training.")
    parser.add_argument('--exploration_steps', type=int, default=50000, help="Steps to use random instead of learned policy.")
    parser.add_argument('--no_bootstrap_on_done', action="store_true", help="If set, use dones to prevent bootstrapping on timeouts.")
    parser.add_argument('--no_entropy_in_qloss', action="store_true", help="If set, remove entropy from q loss.")
    parser.add_argument('--debug_run', action="store_true", help="Drop log interval, buffer warmup, and exploration steps for debugging")
    parser.add_argument('--p_weight_decay', type=float, default=0.01, help="Weight decay for policy, critic, discriminator.")
    parser.add_argument('--c_weight_decay', type=float, default=0.01, help="Weight decay for policy, critic, discriminator.")
    parser.add_argument('--d_weight_decay', type=float, default=0.01, help="Weight decay for policy, critic, discriminator.")
    parser.add_argument('--target_polyak_averaging', type=float, default=0.0001, help="Polyak averaging for updates from target.")
    parser.add_argument('--actor_lr', type=float, default=1e-5, help="Actor learning rate.")
    parser.add_argument('--alpha_lr', type=float, default=3e-4, help="Alpha learning rate.")
    parser.add_argument('--critic_lr', type=float, default=3e-4, help="Critic learning rate.")
    parser.add_argument('--discriminator_lr', type=float, default=3e-4, help="Discriminator learning rate.")
    # parser.add_argument('--non_actor_lr', type=float, default=3e-4, help="Critic, discriminator, alpha learning rate.")
    parser.add_argument('--max_episode_length', type=int, default=360, help="Maximum length of episode.")
    parser.add_argument('--sac_alpha_mode', type=str, choices=['learned', 'fixed'], default='learned')
    parser.add_argument('--sac_initial_alpha', type=float, default=0.01, help="Initial SAC alpha value.")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for training disc + model.")
    parser.add_argument('--reward_scaling', type=float, default=1.0, help="Reward scaling.")
    parser.add_argument('--obs_rms', action="store_true",
                        help="Normalize observations as the model takes them in by calculating RMS on the fly.")
    parser.add_argument('--eval_mode', choices=['det', 'sto'], default='det',
                        help="Eval with [det]erministic or [sto]chastic policy.")
    parser.add_argument('--discount_factor', type=float, default=0.99, help="Discount factor.")
    parser.add_argument('--action_repeat', type=int, default=1, help="Action repeat.")
    parser.add_argument('--frame_stack', type=int, default=1, help="Num frames to stack.")
    parser.add_argument('--buffer_randomize_factor', type=float, default=0.0,
                        help="Factor to randomize each dimension of buffer data by, after normalizing")
    parser.add_argument('--exponential_sampling_method', type=str, choices=['none', 'med_prop', 'med_fixed'], default='none',
                        help="If not none, either median is proportional to current buffer size, or it's fixed.")
    parser.add_argument('--exponential_sampling_param', type=float, default=0.1,
                        help="Exponential sampling of data, if 'med_prop', this parameter multiplied by the current "
                             "buffer size is the median. If 'med_fixed, this is the median")
    parser.add_argument('--exponential_uniform_prop', type=float, default=0.5,
                        help="If exponential sampling is on, what proportion should be exponential vs. uniform.")
    parser.add_argument('--num_gradient_updates', type=int, default=1, help="Num training steps per env step.")

    # env
    parser.add_argument('--env_type', type=str, choices=['manipulator_learning', 'sawyer', 'hand_dapg'],
                        default="manipulator_learning")
    parser.add_argument('--env_name', type=str, default="PandaPlayInsertTrayXYZState", help="Env name.")
    parser.add_argument('--main_task', type=str, default="stack_01", help="Main task (for play environment)")
    parser.add_argument('--env_control_hz', type=int, default=20, help="Control rate (for play environment)")
    parser.add_argument('--env_max_real_time', type=float, default=18.0, help="Max real time in s (for play env)")
    parser.add_argument('--env_state_data', type=str,
                        # default=('pos,vel,grip_pos,prev_grip_pos,obj_pos,obj_rot,obj_vel,obj_rot_vel,force_torque'),
                        default=('pos,vel,grip_pos,prev_grip_pos,obj_pos,obj_rot,obj_vel,obj_rot_vel'),
                                     help="Things included in state (for play env)")
    parser.add_argument('--train_during_env_step', action='store_true',
                    help="(for non-sim envs) Perform train/backwards pass during env step delay between act exec and obs gen")

    # expert data
    parser.add_argument('--expert_top_dir', type=str, default='../../lfgp_data/expert_data/')
    parser.add_argument('--expert_dir_rest', type=str, default='stack/')
    parser.add_argument('--expert_filenames', type=str, default='int_0.gz,int_1.gz,int_2.gz,int_3.gz,int_4.gz,int_5.gz',
                        help="comma-separated list (no spaces) of filenames after expert_top_dir and expert_dir_rest.")
    parser.add_argument('--expert_amounts', type=str, default="",
                        help="Amounts of expert data to use. Can be empty string (whole files), single number (same "\
                             "num for each task), or comma-separated list (per-task numbers).")
    parser.add_argument('--expert_randomize_factor', type=float, default=0.0,
                        help="Factor to randomize each dimension of expert data by, after normalizing")
    parser.add_argument('--expbuf_size_type', type=str, choices=['match_bs', 'fraction'], default='fraction',
        help="Fraction means each expert buffer samples batch_size / 2 / num_tasks, match_bs means each samples batch_size."\
             " Significantly increases memory usage and processing time, so batch_size should probably be lowered.")
    parser.add_argument('--rew_min_zero', action='store_true',
                        help="If set, use a median filter to estimate min disc output, and set that as minimum rew. "
                             "Only applies to discriminator based reward.")
    parser.add_argument('--rmz_num_med_filt', type=int, default=50, help="Num points to use for median filter for rews_min_zero")

    # data
    parser.add_argument('--top_save_path', type=str, default='results', help="Top directory for saving results")
    parser.add_argument('--exp_name', type=str, default="", help="String corresponding to the experiment name")
    parser.add_argument('--gpu_buffer', action='store_true', default=False, help="Store buffers on gpu.")
    parser.add_argument('--load_latest_checkpoint', action='store_true', help="Continue training latest exp_name checkpoint")
    parser.add_argument('--checkpoint_name', type=str, default='checkpoint', help="Checkpoint name for load_latest_checkpoint")
    parser.add_argument('--save_checkpoint_name', type=str, default='checkpoint', help="Checkpoint name for saving checkpoints")
    parser.add_argument('--checkpoint_every_ep', action='store_true', help="Save checkpoint after every ep to restart from")
    parser.add_argument('--load_buffer_name', type=str, default='checkpoint',
                        help="Buffer name for loading training jumpoff point")
    parser.add_argument('--load_model_name', type=str, default="", help="Model plus tracking dict name for jumpoff point")
    parser.add_argument('--load_buffer_start_index', type=int, default=-1, help="Starting buffer index for jumpoff point")

    # n step
    parser.add_argument('--n_step', type=int, default=1, help="If greater than 1, add an n-step loss to the q updates.")
    parser.add_argument('--n_step_mode', type=str, default="nth_q_targ",
                        help="N-step modes: options are: [n_rew_only, sum_pad, nth_q_targ].")
    parser.add_argument('--nth_q_targ_multiplier', type=float, default=.5, help="applies to nth_q_targ n_step, .5 is value used in RCE.")

    # lfgp/discriminator
    parser.add_argument('--expbuf_last_sample_prop', type=float, default=0.95,
        help="Proportion of mini-batch samples that should be final transitions for discriminator training. 0.0 \
            means regular sampling.")
    parser.add_argument('--expbuf_model_sample_rate', type=float, default=0.1,
        help="Proportion of mini-batch samples that should be expert samples for q/policy training.")
    parser.add_argument('--expbuf_critic_share_type', type=str, default='share',
        help="Whether all critics learn from all expert buffers or from only their own. Options: [share, no_share]")
    parser.add_argument('--expbuf_policy_share_type', type=str, default='share',
        help="Whether all policies learn from all expert buffers or from only their own. Options: [share, no_share]")
    parser.add_argument('--expbuf_model_train_mode', type=str, default='both',
        help="Whether expert data trains the critic, or both the actor and critic. Options: [both, critic_only]")
    parser.add_argument('--expbuf_model_sample_decay', type=float, default=0.99999,
        help="Decay rate for expbuf_model_sample_rate. .99999 brings close to 0 at 1M.")
    parser.add_argument('--actor_raw_magnitude_penalty', type=float, default=0.0, help="L2 penalty on raw action (before tanh).")
    parser.add_argument('--expert_data_mode', type=str, default="obs_act", help="options are [obs_act, obs_only, obs_only_no_next].")
    parser.add_argument('--reward_model', type=str, default="discriminator", help="Options: [discriminator, sqil, rce]")
    parser.add_argument('--no_shared_layers', action="store_true", help="Remove the set of shared layers for q/policy.")
    parser.add_argument('--expert_critic_weight', type=float, default=1.0,
                        help="If using expert data to train critic, use this as weight in loss.")
    parser.add_argument('--obs_dim_disc_ignore', type=str, required=False,
                        help="comma-separated list of observation dimensions to not be used for discriminator learning.")
    parser.add_argument('--disc_activation_func', type=str, choices=['Tanh', 'ReLU'], default='Tanh',
                        help="Activation function for discriminator.")

    return parser