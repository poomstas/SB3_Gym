# %%
from matplotlib import animation
import matplotlib.pyplot as plt
from stable_baselines3 import A2C, PPO, SAC
import gym 

# %%
def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
    def animate(i):
        patch.set_data(frames[i])
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=500)
    anim.save(path + filename, writer='imagemagick', fps=60)

# %% Specify gym environment and model
ENV_NAME = 'BipedalWalker-v3'
MODEL_PATH = '/Users/Brian/Dropbox/SideProjects/20220207_Stable_Baselines_3_Gym/SB3_Gym/bipedalwalker_models/SAC_910000.zip'
MODEL_TYPE = 'SAC'

# %% Create gym environment and load model
env = gym.make(ENV_NAME)
if MODEL_TYPE == 'PPO':
    model = PPO.load(MODEL_PATH)
elif MODEL_TYPE == 'A2C':
    model = A2C.load(MODEL_PATH)
elif MODEL_TYPE == 'SAC':
    model = SAC.load(MODEL_PATH)

# %% Run the env
observation = env.reset()
frames = []
for t in range(1000):
    frames.append(env.render(mode="rgb_array"))
    action, _ = model.predict(observation)
    observation, _, done, _ = env.step(action)
    if done:
        break
env.close()
save_frames_as_gif(frames)
