import gym
from gym.envs.registration import register

from .cliff_walking import *
from .frozen_lake import *
from .taxi import *
from .cognitive_radio import *

__all__ = ['RewardingFrozenLakeEnv', 'WindyCliffWalkingEnv']

register(
    id='RewardingFrozenLake-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '4x4'},
)

register(
    id='RewardingFrozenLake8x8-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '8x8'}
)

register(
    id='RewardingFrozenLakeNoRewards20x20-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '20x20', 'rewarding': False}
)

register(
    id='RewardingFrozenLakeNoRewards8x8-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '8x8', 'rewarding': False}
)

register(
    id='WindyCliffWalking-v0',
    entry_point='environments:WindyCliffWalkingEnv',
)

register(
    id='Taxi5x5-v0',
    entry_point='environments:TaxiEnv'
)

register(
    id='CognitiveRadio5x20-v0',
    entry_point='environments:CognitiveRadio',
    kwargs={'map_name': '5x20', 'simple_tdma': True, 'max_tune_dist': 4}
)

register(
    id='CognitiveRadio8x30-v0',
    entry_point='environments:CognitiveRadio',
    kwargs={'map_name': '8x30', 'simple_tdma': False, 'max_tune_dist': 3}
)

register(
    id='CognitiveRadio8x30Simple-v0',
    entry_point='environments:CognitiveRadio',
    kwargs={'map_name': '8x30', 'simple_tdma': True, 'max_tune_dist': 3, 'collision_reward': 4}
)

register(
    id='CognitiveRadio10x40-v0',
    entry_point='environments:CognitiveRadio',
    kwargs={'map_name': '10x40', 'simple_tdma': False, 'max_tune_dist': 4}
)

def get_complex_cognitive_radio_environment():
    return gym.make('CognitiveRadio8x30-v0')

def get_large_complex_cognitive_radio_environment():
    return gym.make('CognitiveRadio10x40-v0')

def get_simple_cognitive_radio_environment():
    return gym.make('CognitiveRadio8x30Simple-v0')

def get_taxi_environment():
    return gym.make('Taxi5x5-v0')


def get_rewarding_frozen_lake_environment():
    return gym.make('RewardingFrozenLake8x8-v0')


def get_frozen_lake_environment():
    return gym.make('FrozenLake-v0')


def get_rewarding_no_reward_frozen_lake_environment():
    return gym.make('RewardingFrozenLakeNoRewards8x8-v0')


def get_large_rewarding_no_reward_frozen_lake_environment():
    return gym.make('RewardingFrozenLakeNoRewards20x20-v0')


def get_cliff_walking_environment():
    return gym.make('CliffWalking-v0')


def get_windy_cliff_walking_environment():
    return gym.make('WindyCliffWalking-v0')
