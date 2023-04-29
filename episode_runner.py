import base64
import pickle

from flask import Flask, jsonify, request
from project.ppo.pose_trainer import Trainer

from project.ppo.model import Actor, Critic
from project.envs.sorter import SorterEnv, OBSERVATION_POSES

app = Flask(__name__)




@app.route('/', methods=['POST'])
def main():
    data = request.get_json(force=True)

    # Get the model and convert from base64 to bytes
    model = data['model']
    print("size", len(model))
    model = base64.b64decode(model.strip().encode('utf-8'))
    print("size", len(model))
    print("start", model[0:10], "end", model[-10:])
    model = pickle.loads(model)

    # Get hyperparameters from request
    timesteps = int(data['timesteps'])
    objects = int(data['objects'])
    blocker = bool(data['blocker'])
    gamma = float(data['gamma'])

    print("args", timesteps, objects, blocker, gamma)

    # Create our environment
    env = SorterEnv(
        OBSERVATION_POSES,
        objects,
        render_mode="rgb_array",
        renderer="Tiny",
        blocker_bar=blocker,
    )

    actor = Actor(env)
    actor.model.load_state_dict(model)
    critic = Critic(env)

    trainer = Trainer(
        env,
        actor,
        critic,
        max_timesteps_per_episode=timesteps,
        γ=gamma,
    )
    observations, actions, log_probabilities, discounted_rewards, total_rewards = trainer.run_episode(env)

    # Prepare our json response of all data:
    response = {
        "observations": observations,
        "actions": actions,
        "log_probabilities": log_probabilities,
        "discounted_rewards": discounted_rewards,
        "total_rewards": total_rewards,
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(port=5000)