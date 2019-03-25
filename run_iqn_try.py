
# @title Necessary imports and globals.

import numpy as np
import os
from dopamine.agents.implicit_quantile import implicit_quantile_agent
from dopamine.discrete_domains import run_experiment
from dopamine.colab import utils as colab_utils
from absl import flags
import gin.tf

BASE_PATH = '/tmp/colab_dope_run'  # @param
GAME = 'Asterix'  # @param

# @title Load the configuration for DQN.

IQN_PATH = os.path.join(BASE_PATH, 'implicit_quantile')
# Modified from dopamine/agents/dqn/config/dqn_cartpole.gin
iqn_config = """

import dopamine.agents.implicit_quantile.implicit_quantile_agent
import dopamine.agents.rainbow.rainbow_agent
import dopamine.discrete_domains.atari_lib
import dopamine.discrete_domains.run_experiment
import dopamine.replay_memory.prioritized_replay_buffer
import gin.tf.external_configurables

ImplicitQuantileAgent.kappa = 1.0
ImplicitQuantileAgent.num_tau_samples = 64
ImplicitQuantileAgent.num_tau_prime_samples = 64
ImplicitQuantileAgent.num_quantile_samples = 32
RainbowAgent.gamma = 0.99
RainbowAgent.update_horizon = 1
RainbowAgent.min_replay_history = 500 # agent steps
RainbowAgent.update_period = 4
RainbowAgent.target_update_period = 100 # agent steps
RainbowAgent.epsilon_train = 0.01
RainbowAgent.epsilon_eval = 0.001
RainbowAgent.epsilon_decay_period = 250000  # agent steps
# IQN currently does not support prioritized replay.
RainbowAgent.replay_scheme = 'uniform'
RainbowAgent.tf_device = '/gpu:0'  # '/cpu:*' use for non-GPU version
RainbowAgent.optimizer = @tf.train.AdamOptimizer()

tf.train.AdamOptimizer.learning_rate = 0.00005
tf.train.AdamOptimizer.epsilon = 0.0003125

atari_lib.create_atari_environment.game_name = 'Pong'
# Sticky actions with probability 0.25, as suggested by (Machado et al., 2017).
atari_lib.create_atari_environment.sticky_actions = True
create_agent.agent_name = 'implicit_quantile'
Runner.num_iterations = 50
Runner.training_steps = 1000
Runner.evaluation_steps = 1000
Runner.max_steps_per_episode = 200  # Default max episode length.

WrappedPrioritizedReplayBuffer.replay_capacity = 50000
WrappedPrioritizedReplayBuffer.batch_size = 128
"""
gin.parse_config(iqn_config, skip_unknown=False)

# @title Train IQN
iqn_runner = run_experiment.create_runner(IQN_PATH, schedule='continuous_train')
print('Will train IQN agent, please be patient, may be a while...')
iqn_runner.run_experiment()
print('Done training!')