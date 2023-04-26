from project.envs.sorter import SorterEnv, OBSERVATION_POSES
from project.ppo.model import Actor

from PIL import Image
from typing import Dict, List
from time import sleep

FPS = 32
FRAME_DELAY = 1.0/FPS
MAX_EPISODE_LENGTH_SECONDS = 5
RECORDING_EPISODE_COUNT = 10
MODEL_PATH = "./training/actor.pth"
OUTPUT_FILE = "./playback.gif"

env = SorterEnv(
    OBSERVATION_POSES,
    3,
    render_mode="rgb_array",
    renderer="OpenGL",
    blocker_bar=True
)

actor = Actor(env)
actor.load(MODEL_PATH)

episode = 0
episode_length = 0
frames = []
observation, info = env.reset()

while episode < RECORDING_EPISODE_COUNT:
    episode_length += 1
    if isinstance(observation, Dict):
        observation = observation["observation"]
    action = actor(observation).sample().detach().numpy()
    observation, reward, terminated, truncated, info = env.step(action)

    img = Image.fromarray(env.render())
    frames.append(img)

    if episode_length >= FPS * MAX_EPISODE_LENGTH_SECONDS:
        terminated = True

    if terminated or truncated:
        episode += 1
        episode_length = 0
        observation, info = env.reset()

env.close()

Image.new('RGB', frames[0].size).save(
    fp=OUTPUT_FILE,
    format='GIF',
    append_images=frames,
    save_all=True,
    duration=FRAME_DELAY*len(frames),
    loop=0,
)
