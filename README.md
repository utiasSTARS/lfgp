# Learning from Guided Play: Improving Exploration in Adversarial Imitation Learning with Simple Auxiliary Tasks
#### Trevor Ablett*, Bryan Chan*, Jonathan Kelly _(*equal contribution)_
*Submitted to Robotics and Automation Letters (RA-L) with IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS'22) Option*

**Paper website:** https://papers.starslab.ca/lfgp/

*Presented as "Learning from Guided Play: A Scheduled Hierarchical Approach for Improving Exploration in Adversarial Imitation Learning" as a poster at the NeurIPS 2021 Deep Reinforcement Learning Workshop.*

**NeurIPS workshop arXiv paper:** https://arxiv.org/abs/2112.08932

****
<img src="https://raw.githubusercontent.com/utiasSTARS/lfgp/master/system.png" width="75%" >

Adversarial Imitation Learning (AIL) is a technique for learning from demonstrations that helps remedy the distribution shift problem that occurs with supervised learning (Behavioural Cloning). In our paper, we show through many experiments and analysis that, in manipulation environments, AIL suffers from deceptive rewards, leading to suboptimal policies. In this work, we resolve this by enforcing exploration of a set of easy-to-define auxiliary tasks, in addition to a main task.

This repository contains the source code for reproducing our results.

## Setup
We recommend the readers to setup a virtual environment (e.g. `virtualenv`, `conda`, `pyenv`, etc.). Please also ensure to use Python 3.7 as we have not tested in any other Python versions. In the following, we assume the working directory is the directory containing this README:
```
.
├── lfgp_data/
├── pytorch-a2c-ppo-acktr-gail/
├── rl_sandbox/
├── six_state_mdp.py
└── README.md
```

To install, simply clone and install with pip, which will automatically install all dependencies:
```
git clone git@github.com:utiasSTARS/lfgp.git && cd lfgp
pip install rl_sandbox
```

## Environments
In this paper, we evaluated our method in the four environments listed below:
```
bring_0                  # bring blue block to blue zone
stack_0                  # stack blue block onto green block
insert_0                 # insert blue block into blue zone slot
unstack_stack_env_only_0 # remove green block from blue block, and stack blue block onto green block
```

## Trained Models and Expert Data
The expert and trained lfgp models can be found at [this google drive link](https://drive.google.com/file/d/1yxJ4FuDWvFxkDg4rGSrAdwpw5UIw0Gjd/view). The zip file is 570MB. All of our generated expert data is included, but we only include single seeds of each trained model to reduce the size.

### The Data Directory
This subsection provides the desired directory structure that we will be assuming for the remaining README.
The unzipped `lfgp_data` directory follows the structure:
```
.
├── lfgp_data/
│   ├── expert_data/
│   │   ├── unstack_stack_env_only_0-expert_data/
│   │   │   ├── reset/
│   │   │   │   ├── 54000_steps/
│   │   │   │   └── 9000_steps/
│   │   │   └── play/
│   │   │       └── 9000_steps/
│   │   ├── stack_0-expert_data/
│   │   │   └── (same as unstack_stack_env_only_0-expert_data)/
│   │   ├── insert_0-expert_data/
│   │   │   └── (same as unstack_stack_env_only_0-expert_data)/
│   │   └── bring_0-expert_data/
│   │       └── (same as unstack_stack_env_only_0-expert_data)/
│   └── trained_models/
│       ├── experts/
│       │   ├── unstack_stack_env_only_0/
│       │   ├── stack_0/
│       │   ├── insert_0/
│       │   └── bring_0/
│       ├── unstack_stack_env_only_0/
│       │   ├── multitask_bc/
│       │   ├── lfgp_ns/
│       │   ├── lfgp/
│       │   ├── dac/
│       │   ├── bc_less_data/
│       │   └── bc/
│       ├── stack_0/
│       │   └── (same as unstack_stack_env_only_0)
│       ├── insert_0/
│       │   └── (same as unstack_stack_env_only_0)
│       └── bring_0/
│           └── (same as unstack_stack_env_only_0)
├── liegroups/
├── manipulator-learning/
├── pytorch-a2c-ppo-acktr-gail/
├── rl_sandbox/
├── README.md
└── requirements.txt
```

## Create Expert and Generate Expert Demonstrations
Readers can generate their own experts and expert demonstrations by executing the scripts in the `rl_sandbox/rl_sandbox/examples/lfgp/experts` directory. More specifically, `create_expert.py` and `create_expert_data.py` respectively train the expert and generate the expert demonstrations. We note that training the expert is time consuming and may take up to multiple days.

To create an expert, you can run the following command:
```
# Create a stack expert using SAC-X with seed 0. --gpu_buffer would store the replay buffer on the GPU.
# For more details, please use --help command for more options.
python rl_sandbox/rl_sandbox/examples/lfgp/experts/create_expert.py \
    --seed=0 \
    --main_task=stack_0 \
    --device=cuda \
    --gpu_buffer
```

A `results` directory will be generated. A tensorboard, an experiment setting, a training progress file, model checkpoints, and a buffer checkpoint will be created.

To generate play-based and reset-based expert data using a trained model, you can run the following commands:
```
# Generate play-based stack expert data with seed 1. The program halts when one of --num_episodes or --num_steps is satisfied.
# For more details, please use --help command for more options
python rl_sandbox/rl_sandbox/examples/lfgp/experts/create_expert_data.py \
--model_path=lfgp_data/trained_models/experts/stack_0/state_dict.pt \
--config_path=lfgp_data/trained_models/experts/stack_0/sacx_experiment_setting.pkl \
--save_path=./test_expert_data/ \
--num_episodes=10000000 \
--num_steps=9000 \
--seed=1 \
--render \
--scheduler_period=90 \
--success_only \
--reset_on_success  # note that this does not actually reset the environment without reset_between_intentions, it just flips the scheduler

# Generate reset-based stack expert data with seed 1. Note that --num_episodes will need to be scaled by number of tasks (i.e. num_episodes * num_tasks).
python rl_sandbox/rl_sandbox/examples/lfgp/experts/create_expert_data.py \
--model_path=lfgp_data/trained_models/experts/stack_0/state_dict.pt \
--config_path=lfgp_data/trained_models/experts/stack_0/sacx_experiment_setting.pkl \
--save_path=./test_expert_data/ \
--num_episodes=10000000 \
--num_steps=9000 \
--seed=1 \
--render \
--forced_schedule="{0: {0: ([0, 1, 2, 3, 4, 5], [.15, .15, .25, .15, .15, .15], ['k', 'd', 'd', 'd', 'd', 'd']), 45: 0}, 1: {0: 3, 15: 1}}" \
--scheduler_period=90 \
--success_only \
--reset_on_success \
--reset_between_intentions 

```

The generated expert data will be stored under `--save_path`, in separate files `int_0.gz, ..., int_{num_tasks - 1}.gz`.

## Training the Models with Imitation Learning
The following commands assume you're using the provided expert data and the directory structure outlined above.
The training scripts `run_*.py` are stored in `rl_sandbox/rl_sandbox/examples/lfgp` directory. There are five `run` scripts, each corresponding to a variant of the compared methods (except for behavioural cloning less data, since the change is only in the expert data). The runs will be saved in the same `results` directory mentioned previously. Note that the default hyperparameters specified in the scripts are listed on the appendix.

### Behavioural Cloning (BC)
There are two scripts for single-task and multitask BC: `run_bc.py` and `run_multitask_bc.py`. You can run the following commands:
```
# Train single-task BC agent to stack with using reset-based data.
# NOTE: intention 2 is the main intention (i.e. stack intention). The main intention is indexed at 2 for all environments.
python rl_sandbox/rl_sandbox/examples/lfgp/run_bc.py \
--seed=0 \
--expert_path=lfgp_data/expert_data/stack_0-expert_data/reset/54000_steps/int_2.gz \
--main_task=stack_0 \
--render \
--device=cuda

# Train multitask BC agent to stack with using reset-based data.
python rl_sandbox/rl_sandbox/examples/lfgp/run_multitask_bc.py \
--seed=0 \
--expert_paths="lfgp_data/expert_data/stack_0-expert_data/reset/9000_steps/int_0.gz,\
lfgp_data/expert_data/stack_0-expert_data/reset/9000_steps/int_1.gz,\
lfgp_data/expert_data/stack_0-expert_data/reset/9000_steps/int_2.gz,\
lfgp_data/expert_data/stack_0-expert_data/reset/9000_steps/int_3.gz,\
lfgp_data/expert_data/stack_0-expert_data/reset/9000_steps/int_4.gz,\
lfgp_data/expert_data/stack_0-expert_data/reset/9000_steps/int_5.gz" \
--main_task=stack_0 \
--render \
--device=cuda

```

### Adversarial Imitation learning (AIL)
There are three scripts for Discriminator-Actor-Critic (DAC), Learning from Guided Play (LfGP), and LfGP-NS (No Schedule): `run_dac.py`, `run_lfgp.py`, `run_lfgp_ns.py`. You can run the following commands:
```
# Train DAC agent to stack with using reset-based data.
python rl_sandbox/rl_sandbox/examples/lfgp/run_dac.py \
--seed=0 \
--expert_path=lfgp_data/expert_data/stack_0-expert_data/reset/9000_steps/int_2.gz \
--main_task=stack_0 \
--render \
--device=cuda

# Train LfGP agent to stack with using reset-based data.
python rl_sandbox/rl_sandbox/examples/lfgp/run_lfgp.py \
--seed=0 \
--expert_paths="lfgp_data/expert_data/stack_0-expert_data/reset/9000_steps/int_0.gz,\
lfgp_data/expert_data/stack_0-expert_data/reset/9000_steps/int_1.gz,\
lfgp_data/expert_data/stack_0-expert_data/reset/9000_steps/int_2.gz,\
lfgp_data/expert_data/stack_0-expert_data/reset/9000_steps/int_3.gz,\
lfgp_data/expert_data/stack_0-expert_data/reset/9000_steps/int_4.gz,\
lfgp_data/expert_data/stack_0-expert_data/reset/9000_steps/int_5.gz" \
--main_task=stack_0 \
--device=cuda \
--render

# Train LfGP-NS agent to stack with using reset-based data.
python rl_sandbox/rl_sandbox/examples/lfgp/run_lfgp_ns.py \
--seed=0 \
--expert_paths="lfgp_data/expert_data/stack_0-expert_data/reset/9000_steps/int_0.gz,\
lfgp_data/expert_data/stack_0-expert_data/reset/9000_steps/int_1.gz,\
lfgp_data/expert_data/stack_0-expert_data/reset/9000_steps/int_2.gz,\
lfgp_data/expert_data/stack_0-expert_data/reset/9000_steps/int_3.gz,\
lfgp_data/expert_data/stack_0-expert_data/reset/9000_steps/int_4.gz,\
lfgp_data/expert_data/stack_0-expert_data/reset/9000_steps/int_5.gz" \
--main_task=stack_0 \
--device=cuda \
--render

```

We train the GAIL agent using the [`pytorch-a2c-ppo-acktr-gail`](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) repository. Install following the instructions provided by `pytorch-a2c-ppo-acktr-gail/README.md`. We simply convert the expert data using `pytorch-a2c-ppo-acktr-gail/gail_experts/convert_lfgp_expert_data.py` and execute GAIL through `pytorch-a2c-ppo-acktr-gail/scripts/gail.bash`. You can run the following commands:
```
# Convert main tasks' expert data into desired format
cd pytorch-a2c-ppo-acktr-gail/gail_experts
python convert_lfgp_expert_data.py

# Train GAIL agent to stack
cd ../scripts
./gail.bash stack_0 # (Supports: stack_0, bring_0, insert_0, unstack_stack_env_only_0)
```

Note that `convert_lfgp_expert_data.py` assumes the datasets are following `pytorch-a2c-ppo-acktr-gail/gail_experts/data/<main_task>-expert_data/reset/int_2.gz`. This means it is looking for the reset-based variant of the main task's expert dataset.

## Evaluating the Models
The readers may load up trained agents and evaluate them using the `evaluate.py` script under the `rl_sandbox/rl_sandbox/examples/eval_tools` directory.

```
# To evaluate a single tasks (2 is stack)
python rl_sandbox/rl_sandbox/examples/eval_tools/evaluate.py \
--seed=1 \
--model_path=lfgp_data/trained_models/experts/stack_0/state_dict.pt \
--config_path=lfgp_data/trained_models/experts/stack_0/sacx_experiment_setting.pkl \
--num_episodes=10 \
--intention=2 \
--render \
--device=cuda

# To run all intentions for multitask agents (e.g. SAC-X)
python rl_sandbox/rl_sandbox/examples/eval_tools/evaluate.py \
--seed=1 \
--model_path=lfgp_data/trained_models/experts/stack_0/state_dict.pt \
--config_path=lfgp_data/trained_models/experts/stack_0/sacx_experiment_setting.pkl \
--num_episodes=10 \
--intention=-1 \
--render \
--device=cuda
```

## Citation
If you use this in your work, please cite:
<pre>
@misc{ablett2021learning,
      title={Learning from Guided Play: A Scheduled Hierarchical Approach for Improving Exploration in Adversarial Imitation Learning}, 
      author={Trevor Ablett and Bryan Chan and Jonathan Kelly},
      year={2021},
      eprint={2112.08932},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
</pre>
