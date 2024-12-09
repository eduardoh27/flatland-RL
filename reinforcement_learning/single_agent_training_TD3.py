import random
import sys
from argparse import ArgumentParser, Namespace
from collections import deque
from pathlib import Path
import PIL

from flatland.utils.rendertools import RenderTool

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from reinforcement_learning.td3_agent import TD3Policy
import matplotlib.pyplot as plt
import numpy as np
import torch

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from utils.observation_utils import normalize_observation
from flatland.envs.observations import TreeObsForRailEnv


def train_agent(n_episodes):
    # Environment parameters
    n_agents = 1
    x_dim = 30
    y_dim = 30
    n_cities = 4
    max_rails_between_cities = 2
    max_rails_in_city = 3
    seed = 42

    # Observation parameters
    observation_tree_depth = 2
    #
    #

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) #

    # Observation builder
    tree_observation = TreeObsForRailEnv(max_depth=observation_tree_depth)

    # Setup environment
    env = RailEnv(
        width=x_dim,
        height=y_dim,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            seed=seed,
            grid_mode=False,
            max_rails_between_cities=max_rails_between_cities,
            max_rail_pairs_in_city=max_rails_in_city,
        ),
        line_generator=sparse_line_generator(),
        number_of_agents=n_agents,
        obs_builder_object=tree_observation,
    )

    env.reset(True, True)

    # State and action sizes
    n_features_per_node = env.obs_builder.observation_dim
    n_nodes = sum([4**i for i in range(observation_tree_depth + 1)]) 
    state_size = n_features_per_node * n_nodes
    action_size = 5

    # TD3 parameters
    policy = TD3Policy(state_size, action_size, max_action=10)

    # Max steps per episode
    max_steps = int(100 * (env.height + env.width + (n_agents / n_cities)))

    # Output path
    output_path = Path("./TD3")
    output_path.mkdir(parents=True, exist_ok=True)

    scores = []
    completion = []
    env_renderer = RenderTool(env, gl="PGL")
    frame_list = []

    for episode_idx in range(n_episodes):
        obs, _ = env.reset(regenerate_rail=True, regenerate_schedule=True)
        done = {"__all__": False}
        score = 0
        action_dict = {}

        for step in range(max_steps):
            for agent_handle in env.get_agent_handles():
                if obs[agent_handle]:
                    state = normalize_observation(obs[agent_handle], observation_tree_depth).flatten()
                    action = policy.act(agent_handle, state)
                    action_dict[agent_handle] = action

                    next_obs, rewards, done, _ = env.step(action_dict)
                    next_state = normalize_observation(next_obs[agent_handle], observation_tree_depth).flatten() \
                        if next_obs[agent_handle] else np.zeros_like(state)

                    shaped_reward = rewards[agent_handle] * 0.01 + (1.0 if done[agent_handle] else -0.01)
                    policy.step(agent_handle, state, action, shaped_reward, next_state, done[agent_handle])

                    score += shaped_reward

            if done["__all__"]:
                break

        policy.end_episode(train=True)
        scores.append(score)
        completion.append(done["__all__"])

        # Save GIF every 50 episodes and for the last episode
        if episode_idx % 50 == 0 or episode_idx == n_episodes - 1:
            env_renderer.reset()
            env_renderer.render_env(show=False, show_observations=False, show_predictions=True)
            frame_list.append(PIL.Image.fromarray(env_renderer.gl.get_image()))
            frame_list[0].save(output_path / f"episode_{episode_idx}.gif", save_all=True, append_images=frame_list[1:], duration=3, loop=0)
            frame_list = []

        # Log progress every 50 episodes
        if episode_idx % 50 == 0:
            avg_score = np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores)
            print(f"Episode {episode_idx}/{n_episodes} - Score: {score:.2f}, Avg Score (last 50): {avg_score:.2f}")

    # Save training plots
    plt.figure()
    plt.plot(scores)
    plt.title("Training Scores")
    plt.savefig(output_path / "scores.png")

    plt.figure()
    plt.plot(completion)
    plt.title("Completion Rates")
    plt.savefig(output_path / "completion.png")

    print("Training complete. Results saved in TD3 directory.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-n", "--n_episodes", type=int, default=200, help="Number of episodes")
    args = parser.parse_args()
    train_agent(args.n_episodes)