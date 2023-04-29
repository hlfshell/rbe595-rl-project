import base64
import pickle
import requests

from project.envs.sorter import SorterEnv, OBSERVATION_POSES
from project.ppo.model import Actor

env = SorterEnv(
    OBSERVATION_POSES,
    2,
    render_mode="rgb_array",
    renderer="Tiny",
    blocker_bar=True,
)

actor = Actor(env)

# Get the state dict bytes of the model
serialized_model_weights = pickle.dumps(actor.model.state_dict())
encoded_model_weights = str(base64.b64encode(serialized_model_weights))
gamma = 0.99

print("size", len(encoded_model_weights))
print("??", encoded_model_weights[0:100])
print("size", len(serialized_model_weights))
print("start", serialized_model_weights[0:10], "end", serialized_model_weights[-10:])

# create a json request
request = {
    "model": encoded_model_weights,
    "timesteps": 750,
    "objects": 2,
    "blocker": True,
    "gamma": gamma,
}

# Send the request to the server
response = requests.post(
    "http://localhost:5000/",
    json=request,
)

if response.status_code != 200:
    print("Error: ", response.status_code)
    exit(1)

json_response = response.json()

observations = json_response["observations"]
actions = json_response["actions"]
log_probabilities = json_response["log_probabilities"]
discounted_rewards = json_response["discounted_rewards"]
total_rewards = json_response["total_rewards"]

# Print length of each to confirm
print("observations", len(observations))
print("actions", len(actions))
print("log_probabilities", len(log_probabilities))
print("discounted_rewards", len(discounted_rewards))
print("total_rewards", total_rewards)