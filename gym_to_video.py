import gym
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3 import A2C, PPO, SAC


def record_video(env_name, model, video_length=500, prefix='', video_folder='videos/'):
    """
        :param env_name: (str) e.g. 'BipedalWalker-v3'
        :param model: (RL model) e.g. 'SAC'
        :param model_weight: (file) e.g. './models/SAC/'
        :param video_length: (int)
        :param prefix: (str)
        :param video_folder: (str)
    """
    eval_env = DummyVecEnv([lambda: gym.make(env_name)])
    eval_env = VecVideoRecorder(eval_env,               # Start video at step 0 and record 500 steps
                                video_folder = video_folder,
                                record_video_trigger = lambda step: step==0, 
                                video_length = video_length,
                                name_prefix = prefix)
    obs = eval_env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs)
        obs, _, _, _ = eval_env.step(action)
    eval_env.close()


if __name__=='__main__':
    MODEL = 'SAC'
    ENV_NAME = 'BipedalWalker-v3'
    MODEL_PATH = '/Users/Brian/Dropbox/SideProjects/20220207_Stable_Baselines_3_Gym/SB3_Gym/bipedalwalker_models/SAC_910000.zip'

    if MODEL == 'PPO':
        model = PPO.load(MODEL_PATH)
    elif MODEL == 'A2C':
        model = A2C.load(MODEL_PATH)
    elif MODEL == 'SAC':
        model = SAC.load(MODEL_PATH)
    
    record_video(ENV_NAME, model)