# rbe595-rl-project
Reinforcement Learning Project

This project is an attempt to learn PPO (Proximal Policy Optimzation) with a complex robotic arm environment.

<img src="imgs/2_objects_approx_300_average.gif">

<img src="imgs/1_with_blocker_average_200.gif">

Files of note:

* `train_arm.py` - with some modification, it will kick off training a configurable environment of the robotic arm
* `playback.py` - with some modification, it will load a given agent's model file and execute it on a given environment for several episodes, then save the results to a `.gif` file.
* `agents` folder has all trained agents for given environment configurations and experiments