from gym.envs.registration import register

from gym.envs.registration import registry

env_id='diaspora-v0'
if env_id in registry.env_specs.keys():
    del registry.env_specs[env_id]

register(
    id='diaspora-v0',
    entry_point='DiasporaGym.env:DiasporaEnv',
)

