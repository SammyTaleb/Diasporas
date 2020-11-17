from gym.envs.registration import register
import random
#a=random.randint(3,10000000000)
register(
    id=f'diaspora-v10',
    entry_point='DiasporaGym.env:DiasporaEnv',
)

