# %%
import os
import gym
from datetime import datetime
from stable_baselines3 import A2C, PPO

# %% Input variables
MODEL = 'A2C'
MODELS_DIR = './LunarLanderV2_models/' + MODEL
LOG_DIR = './LunarLanderV2_logs/'
SAVE_EVERY = 10000 # Save once every N iterations
TOTAL_TIMESTEPS = SAVE_EVERY * 30

# %% Get timestamp string
now = datetime.now()
current_time_code = now.strftime("%Y%m%d_%H%M%S")
print('Current time code:', current_time_code)

# %% Create directories if nonexistent
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
if not os.path.exists(os.path.join(MODELS_DIR, current_time_code)):
    os.makedirs(os.path.join(MODELS_DIR, current_time_code))
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# %% Train the model
env = gym.make('LunarLander-v2')
env.reset()

if MODEL == 'A2C':
    model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=LOG_DIR + current_time_code)
elif MODEL == 'PPO':
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=LOG_DIR + current_time_code)

for i in range(TOTAL_TIMESTEPS//SAVE_EVERY):
    model.learn(total_timesteps=SAVE_EVERY, reset_num_timesteps=False, tb_log_name=MODEL)
    model.save(f'{MODELS_DIR}/{current_time_code}/{MODEL}_{SAVE_EVERY*i}')

print('Sample Action:', env.action_space.sample())
print('Observation Space Shape: ', env.observation_space.shape)
print('Sample Observation', env.observation_space.sample())

# %% Try out the final model
episodes = 10
for ep in range(episodes):
    obs = env.reset()
    done = False

    while not done:
        env.render()
        # action = env.action_space.sample()
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)

env.close()
