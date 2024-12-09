import random
import sys
from argparse import ArgumentParser, Namespace
from collections import deque
from pathlib import Path
import PIL
import os

from flatland.utils.rendertools import RenderTool

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from utils.observation_utils import normalize_observation
from flatland.envs.observations import TreeObsForRailEnv


# Redes para SAC
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=512):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)


class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=512):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action_one_hot):
        x = torch.cat([state, action_one_hot], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.memory = []
        self.idx = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.memory) < self.buffer_size:
            self.memory.append(None)
        self.memory[self.idx] = (state, action, reward, next_state, done)
        self.idx = (self.idx + 1) % self.buffer_size

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class SACPolicy:
    def __init__(self, state_size, action_size, args):
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.update_every = args.update_every
        self.gamma = args.gamma
        self.tau = args.tau
        self.lr = args.learning_rate
        self.hidden_size = args.hidden_size
        self.device = torch.device("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu")

        self.alpha = 0.2  # Temperatura de entropía

        self.actor = Actor(state_size, action_size, self.hidden_size).to(self.device)
        self.critic_1 = Critic(state_size, action_size, self.hidden_size).to(self.device)
        self.critic_2 = Critic(state_size, action_size, self.hidden_size).to(self.device)
        self.target_critic_1 = Critic(state_size, action_size, self.hidden_size).to(self.device)
        self.target_critic_2 = Critic(state_size, action_size, self.hidden_size).to(self.device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=self.lr
        )

        self.memory = ReplayBuffer(self.buffer_size)
        self.t_step = 0
        self.critic_loss_fn = nn.MSELoss()

    def act(self, agent_id, state, eps=0.0):
        # Uso de eps-greedy reducido: menor exploración al principio
        if random.random() < eps:
            return random.randint(0, self.action_size - 1)
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.actor(state)
        action_probs_np = action_probs.cpu().numpy().squeeze()
        action = np.random.choice(self.action_size, p=action_probs_np)
        return action

    def step(self, agent_id, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step += 1
        if len(self.memory) > self.batch_size and self.t_step % self.update_every == 0:
            self.learn()

    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)

        # Calcular target Q
        with torch.no_grad():
            next_action_probs = self.actor(next_states)
            next_action_dist = torch.distributions.Categorical(next_action_probs)
            next_actions = next_action_dist.sample().unsqueeze(-1)
            next_actions_one_hot = F.one_hot(next_actions.squeeze(-1), num_classes=self.action_size).float()
            next_q1 = self.target_critic_1(next_states, next_actions_one_hot)
            next_q2 = self.target_critic_2(next_states, next_actions_one_hot)
            next_q = torch.min(next_q1, next_q2)
            next_log_prob = next_action_dist.log_prob(next_actions.squeeze(-1)).unsqueeze(-1)
            target_q = rewards + self.gamma * (1 - dones) * (next_q - self.alpha * next_log_prob)

        # Actualizar críticos
        actions_one_hot = F.one_hot(actions.squeeze(-1), num_classes=self.action_size).float()
        q1 = self.critic_1(states, actions_one_hot)
        q2 = self.critic_2(states, actions_one_hot)
        critic_loss = self.critic_loss_fn(q1, target_q) + self.critic_loss_fn(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actualizar actor
        action_probs = self.actor(states)
        action_dist = torch.distributions.Categorical(action_probs)
        sampled_actions = action_dist.sample().unsqueeze(-1)
        sampled_actions_one_hot = F.one_hot(sampled_actions.squeeze(-1), num_classes=self.action_size).float()
        q1_pi = self.critic_1(states, sampled_actions_one_hot)
        q2_pi = self.critic_2(states, sampled_actions_one_hot)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * action_dist.log_prob(sampled_actions.squeeze(-1)) - min_q_pi.squeeze(-1)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft updates con tau reducido
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'target_critic_1': self.target_critic_1.state_dict(),
            'target_critic_2': self.target_critic_2.state_dict()
        }, filename)


def train_agent(n_episodes):
    os.makedirs("SAC", exist_ok=True)

    # Parámetros del entorno
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

    # Parámetros de exploración ajustados
    eps_start = 0.5
    eps_end = 0.01
    eps_decay = 0.9995

    # Semillas
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tree_observation = TreeObsForRailEnv(max_depth=observation_tree_depth)
    env = RailEnv(
        width=x_dim,
        height=y_dim,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            seed=seed,
            grid_mode=False,
            max_rails_between_cities=max_rails_between_cities,
            max_rail_pairs_in_city=max_rails_in_city
        ),
        line_generator=sparse_line_generator(),
        number_of_agents=n_agents,
        obs_builder_object=tree_observation
    )

    env.reset(True, True)

    n_features_per_node = env.obs_builder.observation_dim
    n_nodes = 0
    for i in range(observation_tree_depth + 1):
        n_nodes += np.power(4, i)
    state_size = n_features_per_node * n_nodes
    action_size = 5

    max_steps = int(100 * (env.height + env.width + (n_agents / n_cities)))

    action_dict = dict()
    scores_window = deque(maxlen=100)
    completion_window = deque(maxlen=100)
    scores = []
    completion = []
    action_count = [0] * action_size
    agent_obs = [None] * env.get_num_agents()
    agent_prev_obs = [None] * env.get_num_agents()
    agent_prev_action = [2] * env.get_num_agents()
    update_values = False

    training_parameters = {
        'buffer_size': int(1e5),
        'batch_size': 64,
        'update_every': 8,
        'learning_rate': 3e-4,
        'tau': 0.005,
        'gamma': 0.99,
        'buffer_min_size': 0,
        'hidden_size': 512,
        'use_gpu': False
    }

    policy = SACPolicy(state_size, action_size, Namespace(**training_parameters))

    # Ajustar el número de episodios a 10,000 y GIFs cada 500 episodios
    n_episodes = 10000
    record_images = False
    for episode_idx in range(n_episodes):
        score = 0
        # Guardar GIF cada 500 episodios y en el último
        if (episode_idx + 1) % 500 == 0 or episode_idx == n_episodes - 1:
            record_images = True
        else:
            record_images = False

        obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
        frame_list = []
        if record_images:
            env_renderer = RenderTool(env, gl="PGL")
            env_renderer.reset()

        for agent in env.get_agent_handles():
            if obs[agent]:
                agent_obs[agent] = normalize_observation(obs[agent], observation_tree_depth, observation_radius=observation_radius)
                agent_prev_obs[agent] = agent_obs[agent].copy()

        for step in range(max_steps * 3 - 1):
            for agent in env.get_agent_handles():
                if info['action_required'][agent]:
                    update_values = True
                    action = policy.act(agent, agent_obs[agent], eps=eps_start)
                    action_count[action] += 1
                else:
                    update_values = False
                    action = 0
                action_dict.update({agent: action})

            next_obs, all_rewards, done, info = env.step(action_dict)

            if record_images:
                env_renderer.render_env(show=False, show_observations=False, show_predictions=True)
                frame_list.append(PIL.Image.fromarray(env_renderer.gl.get_image()))

            for agent in env.get_agent_handles():
                if update_values or done[agent]:
                    policy.step(agent, agent_prev_obs[agent], agent_prev_action[agent], all_rewards[agent], agent_obs[agent], done[agent])
                    agent_prev_obs[agent] = agent_obs[agent].copy()
                    agent_prev_action[agent] = action_dict[agent]

                if next_obs[agent]:
                    agent_obs[agent] = normalize_observation(next_obs[agent], observation_tree_depth, observation_radius=10)
                score += all_rewards[agent]

            if done['__all__']:
                if record_images and len(frame_list) > 0:
                    tasks_done = np.sum([int(done[idx]) for idx in env.get_agent_handles()])
                    completed = tasks_done / max(1, env.get_num_agents())
                    frame_list[0].save(f"SAC/flatland_single_agent_{episode_idx}.gif", save_all=True, append_images=frame_list[1:], duration=3, loop=0)
                break

        # Decaimiento de epsilon
        eps_start = max(eps_end, eps_decay * eps_start)

        tasks_finished = np.sum([int(done[idx]) for idx in env.get_agent_handles()])
        completion_window.append(tasks_finished / max(1, env.get_num_agents()))
        scores_window.append(score / (max_steps * env.get_num_agents()))
        completion.append(np.mean(completion_window))
        scores.append(np.mean(scores_window))
        action_probs = action_count / np.sum(action_count)

        if episode_idx % 100 == 0:
            policy.save('SAC/single-' + str(episode_idx) + '.pth')
            action_count = [1] * action_size

        print('\rTraining {} agents on {}x{}\t Episode {}\t Average Score: {:.4f}\tDones: {:.2f}%\tEpsilon: {:.4f} \t Action Probabilities: {}'.format(
            env.get_num_agents(),
            x_dim, y_dim,
            episode_idx,
            np.mean(scores_window),
            100 * np.mean(completion_window),
            eps_start,
            action_probs
        ), end="\n")

    # Evaluación final
    obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
    env_renderer = RenderTool(env, gl="PGL")
    env_renderer.reset()
    frame_list = []
    score = 0
    for step in range(max_steps - 1):
        env_renderer.render_env(show=False, show_observations=False, show_predictions=True)
        frame_list.append(PIL.Image.fromarray(env_renderer.gl.get_image()))
        for agent in env.get_agent_handles():
            if obs[agent]:
                agent_obs[agent] = normalize_observation(obs[agent], observation_tree_depth, observation_radius=observation_radius)
            action = 0
            if info['action_required'][agent]:
                action = policy.act(agent, agent_obs[agent], eps=0.0)
            action_dict.update({agent: action})

        obs, all_rewards, done, info = env.step(action_dict)
        for agent in env.get_agent_handles():
            score += all_rewards[agent]

        if done['__all__']:
            frame_list[0].save("SAC/flatland_single_agent.gif", save_all=True, append_images=frame_list[1:], duration=3, loop=0)
            break

    normalized_score = score / (max_steps * env.get_num_agents())
    print("Normalized Score:", normalized_score)
    tasks_finished = sum(done[idx] for idx in env.get_agent_handles())
    final_completion = tasks_finished / max(1, env.get_num_agents())
    print("Final Completion:", final_completion)

    plt.figure()
    plt.plot(scores)
    plt.title("Average Score")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.savefig('SAC/scores.png')

    plt.figure()
    plt.plot(completion)
    plt.title("Completion")
    plt.xlabel("Episode")
    plt.ylabel("Completion")
    plt.savefig('SAC/completion.png')


if __name__ == "__main__":
    parser = ArgumentParser()
    # Por defecto 10000 episodios
    parser.add_argument("-n", "--n_episodes", dest="n_episodes", help="number of episodes to run", default=10000, type=int)
    args = parser.parse_args()

    train_agent(args.n_episodes)
