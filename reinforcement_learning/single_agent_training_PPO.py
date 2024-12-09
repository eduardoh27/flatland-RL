import random
import sys
from argparse import ArgumentParser, Namespace
from collections import deque
from pathlib import Path
import PIL

from flatland.utils.rendertools import RenderTool

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

# Importamos PPOPolicy desde ppo_agent.py
from reinforcement_learning.ppo_agent import PPOPolicy
import matplotlib.pyplot as plt
import numpy as np
import torch

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from utils.observation_utils import normalize_observation
from flatland.envs.observations import TreeObsForRailEnv


def train_agent(n_episodes):
    # Mismos parámetros de entorno
    n_agents = 1
    x_dim = 30
    y_dim = 30
    n_cities = 4
    max_rails_between_cities = 2
    max_rails_in_city = 3
    seed = 42

    # Observaciones
    observation_tree_depth = 2
    observation_radius = 10

    # Semillas
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Obsevador
    tree_observation = TreeObsForRailEnv(max_depth=observation_tree_depth)

    # Crear entorno
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
        obs_builder_object=tree_observation
    )

    env.reset(True, True)

    # Cálculo tamaños
    n_features_per_node = env.obs_builder.observation_dim
    n_nodes = sum([4**i for i in range(observation_tree_depth + 1)])
    state_size = n_features_per_node * n_nodes
    action_size = 5

    # Pasos máximos
    max_steps = int(100 * (env.height + env.width + (n_agents / n_cities)))

    action_dict = dict()

    scores_window = deque(maxlen=100)
    completion_window = deque(maxlen=100)
    scores = []
    completion = []
    agent_obs = [None] * env.get_num_agents()

    # Parámetros PPO
    ppo_parameters = Namespace(
        hidden_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        buffer_size=10000,
        batch_size=128,
        K_epoch=6,
        use_gpu=torch.cuda.is_available(),
        surrogate_eps_clip=0.2,
        weight_entropy=0.01,
    )

    # Crear política PPO
    policy = PPOPolicy(state_size, action_size, in_parameters=ppo_parameters)

    record_images = False
    frame_list = []

    for episode_idx in range(n_episodes):
        score = 0
        obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
    
        # Cada 50 episodios guardar GIF
        if episode_idx % 50 == 0:
            record_images = True
            env_renderer = RenderTool(env, gl="PGL", )
            env_renderer.reset()
            frame_list = []
        else:
            record_images = False
    
        # Observaciones iniciales
        for agent_handle in env.get_agent_handles():
            if obs[agent_handle] is not None:
                agent_obs[agent_handle] = normalize_observation(
                    obs[agent_handle], observation_tree_depth, observation_radius=observation_radius
                ).flatten()
            else:
                agent_obs[agent_handle] = np.zeros(state_size)
    
        # Ejecutar episodio
        for step in range(max_steps * 3 - 1):
            # Seleccionar acciones
            for agent_handle in env.get_agent_handles():
                if info['action_required'][agent_handle]:
                    action = policy.act(agent_handle, agent_obs[agent_handle])
                else:
                    action = 0
                action_dict[agent_handle] = action
    
            next_obs, all_rewards, done, info = env.step(action_dict)
    
            if record_images:
                env_renderer.render_env(show=False, show_observations=False, show_predictions=True)
                frame_list.append(PIL.Image.fromarray(env_renderer.gl.get_image()))
    
            # Actualizar PPO
            for agent_handle in env.get_agent_handles():
                if next_obs[agent_handle] is not None:
                    next_state = normalize_observation(
                        next_obs[agent_handle], observation_tree_depth, observation_radius=observation_radius
                    ).flatten()
                else:
                    next_state = np.zeros(state_size)
    
                reward = all_rewards[agent_handle]
                done_flag = done[agent_handle]
    
                policy.step(agent_handle,
                            agent_obs[agent_handle],
                            action_dict[agent_handle],
                            reward,
                            next_state,
                            done_flag)
    
                score += reward
                agent_obs[agent_handle] = next_state
    
            if done['__all__']:
                break
            
        # Entrenar PPO al final del episodio
        policy.end_episode(train=True)
    
        # Guardar GIF cada 50 episodios
        if episode_idx % 50 == 0 and record_images and len(frame_list) > 0:
            frame_list[0].save(f"./PPO/flatland_single_agent_{episode_idx}.gif",
                               save_all=True, append_images=frame_list[1:], duration=3, loop=0)
            frame_list = []
    
        # Métricas
        tasks_finished = np.sum([int(done[idx]) for idx in env.get_agent_handles()])
        completion_window.append(tasks_finished / max(1, env.get_num_agents()))
        scores_window.append(score / (max_steps * env.get_num_agents()))
        completion.append(np.mean(completion_window))
        scores.append(np.mean(scores_window))
    
        # Log cada 50 episodios
        if episode_idx % 50 == 0:
            avg_score = np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores)
            print(f"Episode {episode_idx}/{n_episodes} - Score: {score:.2f}, Avg(last 50): {avg_score:.2f}")

    # Evaluación final
    obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
    env_renderer = RenderTool(env, gl="PGL")
    env_renderer.reset()
    frame_list = []
    score = 0
    for step in range(max_steps - 1):
        env_renderer.render_env(show=False, show_observations=False, show_predictions=True)
        frame_list.append(PIL.Image.fromarray(env_renderer.gl.get_image()))

        for agent_handle in env.get_agent_handles():
            if obs[agent_handle] is not None:
                agent_obs[agent_handle] = normalize_observation(
                    obs[agent_handle], observation_tree_depth, observation_radius=observation_radius
                ).flatten()
            else:
                agent_obs[agent_handle] = np.zeros(state_size)

            if info['action_required'][agent_handle]:
                action = policy.act(agent_handle, agent_obs[agent_handle])
            else:
                action = 0
            action_dict[agent_handle] = action

        obs, all_rewards, done, info = env.step(action_dict)
        for agent_handle in env.get_agent_handles():
            score += all_rewards[agent_handle]

        if done['__all__']:
            if len(frame_list) > 0:
                frame_list[0].save(f"./PPO/flatland_single_agent.gif",
                                   save_all=True, append_images=frame_list[1:], duration=3, loop=0)
            frame_list = []
            break

    normalized_score = score / (max_steps * env.get_num_agents())
    print(normalized_score)

    tasks_finished = sum(done[idx] for idx in env.get_agent_handles())
    final_completion = tasks_finished / max(1, env.get_num_agents())
    print(final_completion)

    # Graficar
    plt.figure()
    plt.plot(scores)
    plt.title("Training Scores (PPO)")
    plt.savefig('./PPO/scores.png')

    plt.figure()
    plt.plot(completion)
    plt.title("Completion Rates (PPO)")
    plt.savefig('./PPO/completion.png')

    print("Training complete. Results saved in PPO directory.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-n", "--n_episodes", type=int, default=1000, help="Number of episodes")
    args = parser.parse_args()
    train_agent(args.n_episodes)