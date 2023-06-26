# Since our default folder structure is task/seed/algo/experiment_name/datetime,
# we have algo/experiment_name for each task/algo combo

algo_dir_names = ['dac-x', 'multitask_dac', 'multitask_bc', 'dac', 'bc', 'bc_low_data']

main_performance = {
    'unstack_stack_env_only_0': {
        'lfgp_wrs':        "lfgp_wrs/800_steps-90_sp_point5_play_open_200_extra_lasts_ralv2-best",
        'multitask_bc':    "multitask_bc/800_steps-90_sp_point5_play_open_200_extra_lasts_ralv2",
        'dac':             "dac/4800_steps-90_sp_point5_play_open_1200_extra_lastsralv2",
        'bc':              "bc/4800_steps-90_sp_point5_play_open_1200_extra_lasts_ralv2-2M",
    },
    'stack_0': {
        'lfgp_wrs':        "lfgp_wrs/800_steps-90_sp_point5_play_open_200_extra_lasts_best",
        'multitask_bc':    "multitask_bc/800_steps-90_sp_point5_play_open_200_extra_lasts_ralv2",
        'dac':             "dac/4800_steps-90_sp_point5_play_open_1200_extra_lastsralv2",
        'bc':              "bc/4800_steps-90_sp_point5_play_open_1200_extra_lasts_ralv2-2M",
    },
    'bring_0': {
        'lfgp_wrs':        "lfgp_wrs/800_steps-90_sp_point5_play_open_200_extra_lasts_ralv2-best",
        'multitask_bc':    "multitask_bc/800_steps-90_sp_point5_play_open_200_extra_lasts_ralv2",
        'dac':             "dac/4800_steps-90_sp_point5_play_open_1200_extra_lastsralv2",
        'bc':              "bc/4800_steps-90_sp_point5_play_open_1200_extra_lasts_ralv2-2M",
    },
    'insert_0': {
        'lfgp_wrs':        "lfgp_wrs/800_steps-90_sp_point5_play_open_200_extra_lasts_ralv2-4M",
        'multitask_bc':    "multitask_bc/800_steps-90_sp_point5_play_open_200_extra_lasts_ralv2-4M",
        'dac':             "dac/5600_steps-90_sp_point5_play_open_1400_extra_lasts_ralv2-4M",
        'bc':              "bc/5600_steps-90_sp_point5_play_open_1400_extra_lasts_ralv2-4M",
    },
}

play_vs_reset = {
    'stack_0': {
        'top':            "/media/starslab/users/trevor-ablett/dac-x/play_xyz/",
        'play':       "dac-x/play-9000_steps-90_sp_play_open-q-0.6_polyak",
    },
    'unstack_stack_env_only_0': {
        'top':            "/media/starslab/users/trevor-ablett/dac-x/play_xyz/",
        'play':       "dac-x/play-35M_all_9000_steps",
    },
    'bring_0': {
        'top':            "/media/starslab/users/trevor-ablett/dac-x/play_xyz/",
        'play':       "dac-x/play-9000_steps-90_sp_play_open-q-0.6_polyak",
    },
    'insert_0': {
        'top':            "/media/raid5-array/experiments/dac-x/results/dacx_paper_experiments/panda_play",
        'play':       "dac-x/play-9000_steps-90_sp_suc_only-q-0.6_polyak",
    },
}

dataset_ablations = {
    'stack_0': {
        '0.5': {
            'lfgp_wrs': "lfgp_wrs/400_steps-90_sp_point5_play_open_100_extra_lasts_ralv2-half",
            'multitask_bc': "multitask_bc/400_steps-90_sp_point5_play_open_100_extra_lasts_ralv2-half",
            'dac': "dac/2400_steps-90_sp_point5_play_open_600_extra_lastsralv2-half",
            'bc': "bc/2400_steps-90_sp_point5_play_open_600_extra_lasts_ralv2-half",
        },
        '1.0': {
            'lfgp_wrs': "lfgp_wrs/800_steps-90_sp_point5_play_open_200_extra_lasts_best",
            'multitask_bc': "multitask_bc/800_steps-90_sp_point5_play_open_200_extra_lasts_ralv2",
            'dac': "dac/4800_steps-90_sp_point5_play_open_1200_extra_lastsralv2",
            'bc': "bc/4800_steps-90_sp_point5_play_open_1200_extra_lasts_ralv2-2M",
        },
        '1.5': {
            'lfgp_wrs': "lfgp_wrs/1200_steps-90_sp_point5_play_open_300_extra_lasts_ralv2-1point5",
            'multitask_bc': "multitask_bc/1200_steps-90_sp_point5_play_open_300_extra_lasts_ralv2-1point5",
            'dac': "dac/7200_steps-90_sp_point5_play_open_1800_extra_lasts_ralv2-1point5",
            'bc': "bc/7200_steps-90_sp_point5_play_open_1800_extra_lasts_ralv2-1point5",
        },
        'subsample': {
            'lfgp_wrs': "lfgp_wrs/800_steps-90_sp_point5_play_open_20_ss_200_el_ralv2-subsampled",
            'multitask_bc': "multitask_bc/800_steps-90_sp_point5_play_open_20_ss_200_el_ralv2-subsampled",
            'dac': "dac/4800_steps-90_sp_point5_play_open_20_ss_1200_el_ralv2-subsampled",
            'bc': "bc/4800_steps-90_sp_point5_play_open_20_ss_1200_el_ralv2-subsampled",
        },
        'no_final_trans': {
            'lfgp_wrs': "lfgp_wrs/1000_steps-90_sp_point5_play_open_ralv2-1000-2M",
            'multitask_bc': "multitask_bc/1000_steps-90_sp_point5_play_open_ralv2-1000",
            'dac': "dac/6000_steps-90_sp_point5_play_open_ralv2-subsampled",
            'bc': "bc/6000_steps-90_sp_point5_play_open_ralv2-1000",
        },
    }
}

training_ablations = {
    'stack_0': {
        'scheduler': {
            'lfgp_wrs': "lfgp_wrs/800_steps-90_sp_point5_play_open_200_extra_lasts_best",
            'lfgp_wrs_no_hc': "lfgp_wrs/800_steps-90_sp_point5_play_open_200_extra_lasts_ralv2-no-handcraft",
            'lfgp_learned': "lfgp_learned/800_steps-90_sp_point5_play_open_200_extra_lasts_ralv2-learned",
            'lfgp_no_sched': "lfgp_no_sched/800_steps-90_sp_point5_play_open_200_extra_lasts_ralv2-no_sched",
        },
        'expert_train_for_qp_final_ex_prop': {
            'lfgp_wrs': "lfgp_wrs/800_steps-90_sp_point5_play_open_200_extra_lasts_best",
            'lfgp_wrs_wo_qp': "lfgp_wrs/800_steps-90_sp_point5_play_open_200_extra_lasts_ralv2-noqpexpbuf-2M",
            'lfgp_wrs_eq_prop': "lfgp_wrs/800_steps-90_sp_point5_play_open_200_extra_lasts_ralv2-eqdiscsamp-2M",
            'dac': "dac/4800_steps-90_sp_point5_play_open_1200_extra_lastsralv2",
            'dac_wo_qp': "dac/800_steps-90_sp_point5_play_open_200_extra_lasts_ralv2-noqpexpbuf",  # wrong dataset but probably doesn't matter
            'dac_eq_prop': "dac/800_steps-90_sp_point5_play_open_200_extra_lasts_ralv2-eqdiscsamp",  # same
        },
    }
}

baseline_ablations = {
    'stack_0': {
        'bc_dac_alternatives': {
            'multitask_bc': "multitask_bc/800_steps-90_sp_point5_play_open_200_extra_lasts_ralv2",
            'multitask_bc_valid': "multitask_bc/800_steps-90_sp_point5_play_open_200_extra_lasts_ralv2-valid",
            'dac': "dac/4800_steps-90_sp_point5_play_open_1200_extra_lastsralv2",
            'gail': "gail/test0",
            'bc': "bc/4800_steps-90_sp_point5_play_open_1200_extra_lasts_ralv2-2M",
            'bc_valid': "bc/4800_steps-90_sp_point5_play_open_1200_extra_lasts_ralv2",
        },
    }
}


# scale_performance = {
#     'stack_0': {
#         '0.5': {
#             'lfgp_wrs': "lfgp_wrs/400_steps-90_sp_point5_play_open_100_extra_lasts_ralv2-half",
#             'multitask_bc': "multitask_bc/400_steps-90_sp_point5_play_open_100_extra_lasts_ralv2-half",
#             'dac': "dac/2400_steps-90_sp_point5_play_open_600_extra_lastsralv2-half",
#             'bc': "bc/2400_steps-90_sp_point5_play_open_600_extra_lasts_ralv2-half",
#         },
#         '1.0': {
#             'lfgp_wrs': "lfgp_wrs/800_steps-90_sp_point5_play_open_200_extra_lasts_best",
#             'multitask_bc': "multitask_bc/800_steps-90_sp_point5_play_open_200_extra_lasts_ralv2",
#             'dac': "dac/4800_steps-90_sp_point5_play_open_1200_extra_lastsralv2",
#             'bc': "bc/4800_steps-90_sp_point5_play_open_1200_extra_lasts_ralv2-2M",
#         },
#         '1.5': {
#             'lfgp_wrs': "lfgp_wrs/1200_steps-90_sp_point5_play_open_300_extra_lasts_ralv2-1point5",
#             'multitask_bc': "multitask_bc/1200_steps-90_sp_point5_play_open_300_extra_lasts_ralv2-1point5",
#             'dac': "dac/7200_steps-90_sp_point5_play_open_1800_extra_lasts_ralv2-1point5",
#             'bc': "bc/7200_steps-90_sp_point5_play_open_1800_extra_lasts_ralv2-1point5",
#         },
#     }
# }
