# source: https://github.com/sfujim/TD3_BC
# https://arxiv.org/pdf/2106.06860.pdf
import copy
import os
import random
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import d4rl  # noqa
import gym
import numpy as np
import pandas as pd
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from utils.video_recorder import VideoRecorder

TensorBatch = List[torch.Tensor]


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "halfcheetah-medium-expert-v2"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    log_every: int = 100  # How often (time steps) we log
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    render: bool = False  # Whether to save final testing videos
    # TD3
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount ffor
    expl_noise: float = 0.1  # Std of Gaussian exploration noise
    tau: float = 0.005  # Target network update rate
    policy_noise: float = 0.2  # Noise added to target actor during critic update
    noise_clip: float = 0.5  # Range to clip target actor noise
    policy_freq: int = 2  # Frequency of delayed actor updates
    # TD3 + BC
    alpha: float = 2.5  # Coefficient for Q function in actor loss
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    pretrain: Optional[str] = None  # BC or AC
    pretrain_steps: int = 10000  # Number of pretraining steps
    td_component: float = -1.0  # Proportion of TD to use (rather than MC) in pretraining
    pretrain_cql_regulariser: float = -1.0  # CQL regularisation for pretraining
    cql_regulariser: float = -1.0  # CQL regularisation in training
    cql_n_actions: int = 10  # Number of actions to sample for CQL
    actor_LN: bool = True  # Use LayerNorm in actor
    critic_LN: bool = True  # Use LayerNorm in critic
    # Wandb logging
    project: str = "offline-RL-init"
    group: str = "TD3_BC-D4RL"
    name: str = "TD3_BC"

    def __post_init__(self):
        if self.cql_regulariser > 0.0:
            self.name = f"{self.name}-CQL"
        self.name = f"{self.name}-{self.env}"  # -{str(uuid.uuid4())[:8]}
        if self.checkpoints_path is not None:
            time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            self.checkpoints_path = os.path.join(
                self.checkpoints_path, f"{time}_{self.name}"
            )


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def discount_cumsum(x, discount):
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        disc_cumsum[t] = x[t] + discount * disc_cumsum[t + 1]
    return disc_cumsum


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        discount: float,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._returns_to_go = torch.zeros(
            (buffer_size, 1), dtype=torch.float32, device=device
        )
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._discount = discount
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def compute_returns_to_go(self, data: np.ndarray) -> np.ndarray:
        n_transitions = data["observations"].shape[0]
        episode_rewards = []
        returns_to_go = []

        for i in range(n_transitions):
            episode_rewards.append(data["rewards"][i])
            if data["terminals"][i] or data["timeouts"][i] or i == n_transitions - 1:
                if data["timeouts"][i] or i == n_transitions - 1:
                    episode_rewards[-1] = episode_rewards[-1] / (1 - self._discount)
                episode_returns_to_go = discount_cumsum(
                    np.array(episode_rewards), self._discount
                )
                returns_to_go.append(episode_returns_to_go)
                episode_rewards = []
        returns_to_go_array = np.array(
            [
                return_to_go
                for episode_returns in returns_to_go
                for return_to_go in episode_returns
            ]
        ).flatten()

        return returns_to_go_array

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )

        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(
            (data["terminals"] + data["timeouts"])[..., None]
        )
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        data["returns_to_go"] = self.compute_returns_to_go(data)
        self._returns_to_go[:n_transitions] = self._to_tensor(
            data["returns_to_go"][..., None]
        )

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        returns_to_go = self._returns_to_go[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, returns_to_go, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.log_code(".")
    wandb.run.save()


@torch.no_grad()
def eval_actor(
    env: gym.Env,
    actor: nn.Module,
    device: str,
    n_episodes: int,
    seed: int,
    render: bool,
    name: str,
) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    video = VideoRecorder() if render else None
    # Max demonstration lengths for each environment from human data (described in Appendix H of paper)
    # TODO: Create env wrapper for harcoded truncation fix below for code release
    max_demonstration_lengths = {
        "pen": 100,
        "door": 300,
        "hammer": 624,
        "relocate": 527,
    }
    max_demonstration_length = None
    for s in max_demonstration_lengths.keys():
        if s in env.spec.id:
            max_demonstration_length = max_demonstration_lengths[s]
    for i in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        timestep = 0
        while not done:
            action = actor.act(state, device)
            try:
                state, reward, done, _ = env.step(action)
                timestep += 1
            except Exception as e:
                print(e)
                print(
                    "Error with mujoco! Evaluation episode terminated and return set to zero."
                )
                episode_reward = 0
                break
            if (
                max_demonstration_length is not None
                and timestep < max_demonstration_length
            ):
                done = False
            episode_reward += reward
            if video is not None and i == 0:  # Record 1 episode
                video.record(env)
        episode_rewards.append(episode_reward)

    actor.train()
    if video is not None:
        video.save(name, wandb=wandb)
    return np.asarray(episode_rewards)


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, env_name):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        max_episode_steps = 1000
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        actor_LN: bool = True,
    ):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256, elementwise_affine=False) if actor_LN else nn.Identity(),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256, elementwise_affine=False) if actor_LN else nn.Identity(),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256, elementwise_affine=False) if actor_LN else nn.Identity(),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, critic_LN: bool = True):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.LayerNorm(256, elementwise_affine=False) if critic_LN else nn.Identity(),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256, elementwise_affine=False) if critic_LN else nn.Identity(),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], 1)
        return self.net(sa)


class TD3_BC:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_1: nn.Module,
        critic_1_optimizer: torch.optim.Optimizer,
        critic_2: nn.Module,
        critic_2_optimizer: torch.optim.Optimizer,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        alpha: float = 2.5,
        pretrain_steps: int = 10000,
        td_component: float = 0,
        pretrain_cql_regulariser: float = 0,
        cql_regulariser: float = 0,
        cql_n_actions: int = 10,
        device: str = "cpu",
    ):
        self.actor = actor
        self.actor_target = copy.deepcopy(actor)
        self.actor_optimizer = actor_optimizer
        self.critic_1 = critic_1
        self.critic_1_target = copy.deepcopy(critic_1)
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2 = critic_2
        self.critic_2_target = copy.deepcopy(critic_2)
        self.critic_2_optimizer = critic_2_optimizer

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha

        if td_component > 0.0:
            assert (
                td_component <= 1.0
            ), "TD_component_must be between 0 and 1 (default: 0)."
        self.td_component = td_component
        self.pretrain_cql_regulariser = pretrain_cql_regulariser
        self.cql_regulariser = cql_regulariser
        self.cql_n_actions = cql_n_actions

        self.total_it = 0
        self.pretrain_steps = pretrain_steps
        self.device = device

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        state, action, reward, return_to_go, next_state, done = batch
        not_done = 1 - done

        with torch.no_grad():
            # Select action according to actor and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute the target Q value
            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.discount * target_q

        # Get current Q estimates
        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        log_dict["critic_loss"] = critic_loss.item()

        if self.cql_regulariser > 0.0:
            # Compute CQL regularisation
            random_actions = action.new_empty(
                (self.cql_n_actions * action.shape[0], action.shape[1]),
                requires_grad=False,
            ).uniform_(-1, 1)
            repeated_state = state.repeat(self.cql_n_actions, 1)
            q1_random_values = self.critic_1(repeated_state, random_actions).reshape(
                self.cql_n_actions, -1, 1
            )
            q2_random_values = self.critic_2(repeated_state, random_actions).reshape(
                self.cql_n_actions, -1, 1
            )

            cql_regularisation = (
                torch.logsumexp(q1_random_values, dim=0) - current_q1
            ).mean() + (torch.logsumexp(q2_random_values, dim=0) - current_q2).mean()
            log_dict["support_regulariser"] = cql_regularisation.item()
            critic_loss = (
                critic_loss / critic_loss.detach()
                + self.cql_regulariser * cql_regularisation / cql_regularisation.detach()
            )

        # Optimize the critic
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Delayed actor updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            pi = self.actor(state)
            q = self.critic_1(state, pi)
            lmbda = self.alpha / q.abs().mean().detach()

            actor_loss = -lmbda * q.mean() + F.mse_loss(pi, action)
            log_dict["actor_loss"] = actor_loss.item()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        return log_dict

    def pretrain_BC(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        state, action, reward, return_to_go, next_state, done = batch
        # Compute actor loss
        pi = self.actor(state)
        actor_loss = F.mse_loss(pi, action)
        log_dict["actor_loss"] = actor_loss.item()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return log_dict

    def pretrain_actorcritic(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        state, action, reward, return_to_go, next_state, done = batch
        # Compute actor loss
        pi = self.actor(state)
        actor_loss = F.mse_loss(pi, action)
        log_dict["actor_loss"] = actor_loss.item()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Compute critic loss
        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)

        critic_loss = F.mse_loss(current_q1, return_to_go) + F.mse_loss(
            current_q2, return_to_go
        )
        log_dict["MC_critic_loss"] = critic_loss.item()

        if self.td_component > 0.0:
            with torch.no_grad():
                # Select action according to actor and add clipped noise
                noise = (torch.randn_like(action) * self.policy_noise).clamp(
                    -self.noise_clip, self.noise_clip
                )

                next_action = (self.actor_target(next_state) + noise).clamp(
                    -self.max_action, self.max_action
                )

                #     # Compute the target Q value
                target_q1 = self.critic_1_target(next_state, next_action)
                target_q2 = self.critic_2_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2)
                target_q = reward + (1 - done) * self.discount * target_q

            TD_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            log_dict["TD_loss"] = TD_loss.item()
            critic_loss = (
                1 - self.td_component
            ) * critic_loss / critic_loss.detach() + self.td_component * TD_loss / TD_loss.detach()

        if self.pretrain_cql_regulariser > 0.0:
            # Compute regularisation
            random_actions = action.new_empty(
                (self.cql_n_actions * action.shape[0], action.shape[1]),
                requires_grad=False,
            ).uniform_(-1, 1)
            repeated_state = state.repeat(self.cql_n_actions, 1)
            q1_policy_values = self.critic_1(repeated_state, random_actions).reshape(
                self.cql_n_actions, -1, 1
            )
            q2_policy_values = self.critic_2(repeated_state, random_actions).reshape(
                self.cql_n_actions, -1, 1
            )

            cql_regulariser = (
                torch.logsumexp(q1_policy_values, dim=0) - current_q1
            ).mean() + (torch.logsumexp(q2_policy_values, dim=0) - current_q2).mean()
            log_dict["support_regulariser"] = cql_regulariser.item()

            critic_loss = (
                critic_loss / critic_loss.detach()
                + self.pretrain_cql_regulariser
                * cql_regulariser
                / cql_regulariser.detach()
            )

        log_dict["critic_loss"] = critic_loss.item()
        # Optimize the critic
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Update the frozen target models
        soft_update(self.critic_1_target, self.critic_1, self.tau)
        soft_update(self.critic_2_target, self.critic_2, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "critic_1": self.critic_1.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.critic_1.load_state_dict(state_dict["critic_1"])
        self.critic_1_optimizer.load_state_dict(state_dict["critic_1_optimizer"])
        self.critic_1_target = copy.deepcopy(self.critic_1)

        self.critic_2.load_state_dict(state_dict["critic_2"])
        self.critic_2_optimizer.load_state_dict(state_dict["critic_2_optimizer"])
        self.critic_2_target = copy.deepcopy(self.critic_2)

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_target = copy.deepcopy(self.actor)

        self.total_it = state_dict["total_it"]


@pyrallis.wrap()
def train(config: TrainConfig):
    print(config.env)
    env = gym.make(config.env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    dataset = env.get_dataset()

    if config.normalize_reward:
        modify_reward(dataset, config.env)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )

    if "next_observations" not in dataset.keys():
        dataset["next_observations"] = np.roll(
            dataset["observations"], shift=-1, axis=0
        )  # Terminals/timeouts block next observations
        print("Loaded next state observations from current state observations.")

    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.discount,
        config.device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    actor = Actor(state_dim, action_dim, max_action, config.actor_LN).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    critic_1 = Critic(state_dim, action_dim, config.critic_LN).to(config.device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=3e-4)
    critic_2 = Critic(state_dim, action_dim, config.critic_LN).to(config.device)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=3e-4)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "critic_1": critic_1,
        "critic_1_optimizer": critic_1_optimizer,
        "critic_2": critic_2,
        "critic_2_optimizer": critic_2_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # TD3
        "policy_noise": config.policy_noise * max_action,
        "noise_clip": config.noise_clip * max_action,
        "policy_freq": config.policy_freq,
        # TD3 + BC
        "alpha": config.alpha,
        "pretrain_steps": config.pretrain_steps,
        # Regularisation
        "td_component": config.td_component,
        "pretrain_cql_regulariser": config.pretrain_cql_regulariser,
        "cql_regulariser": config.cql_regulariser,
        "cql_n_actions": config.cql_n_actions,
    }

    print("---------------------------------------")
    print(f"Training TD3 + BC, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = TD3_BC(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(asdict(config))

    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        if config.pretrain is not None:
            if t < config.pretrain_steps:
                if config.pretrain == "BC":
                    log_dict = trainer.pretrain_BC(batch)
                elif config.pretrain == "AC":
                    log_dict = trainer.pretrain_actorcritic(batch)
                elif config.pretrain == "C":
                    log_dict = trainer.pretrain_critic(batch)
                else:
                    raise ValueError(f"Pretrain type {config.pretrain} not recognised.")
            else:
                if t == config.pretrain_steps:
                    with torch.no_grad():
                        trainer.pretrained_critic_1 = copy.deepcopy(trainer.critic_1)
                        trainer.pretrained_critic_2 = copy.deepcopy(trainer.critic_2)
                log_dict = trainer.train(batch)
        else:
            log_dict = trainer.train(batch)
        if t % config.log_every == 0:
            wandb.log(log_dict)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_returns = eval_actor(
                env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
                render=False,
                name=config.name,
            )
            eval_log = {
                "eval/reward_mean": np.mean(eval_returns),
                "eval/reward_std": np.std(eval_returns),
                "epoch": int((t + 1) / 1000),
            }
            if hasattr(env, "get_normalized_score"):
                normalized_score = env.get_normalized_score(eval_returns) * 100.0
                eval_log["eval/normalized_score_mean"] = np.mean(normalized_score)
                eval_log["eval/normalized_score_std"] = np.std(normalized_score)

            wandb.log(eval_log)
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{np.mean(eval_returns):.3f} , D4RL score: {np.mean(normalized_score):.3f}"
            )
            print("---------------------------------------")
            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(
                        config.checkpoints_path,
                        f"checkpoint_{int((t+1)/1000)}.pt",
                    ),
                )
                checkpoints = [
                    os.path.join(config.checkpoints_path, file)
                    for file in os.listdir(config.checkpoints_path)
                    if os.path.splitext(file)[-1] == ".pt"
                ]
                checkpoints.sort(key=os.path.getmtime)
                if len(checkpoints) > 10:
                    oldest_checkpoint = checkpoints.pop(0)
                    os.remove(oldest_checkpoint)
                df = pd.DataFrame(
                    {
                        "epoch": int((t + 1) / 1000),
                        "return_mean": np.mean(eval_returns),
                        "return_std": np.std(eval_returns),
                        "normalized_score_mean": np.mean(normalized_score),
                        "normalized_score_std": np.std(normalized_score),
                    },
                    index=[0],
                )
                if not os.path.exists(
                    os.path.join(config.checkpoints_path, "results.csv")
                ):
                    df.to_csv(
                        os.path.join(config.checkpoints_path, "results.csv"),
                        index=False,
                    )
                else:
                    df.to_csv(
                        os.path.join(config.checkpoints_path, "results.csv"),
                        mode="a",
                        header=False,
                        index=False,
                    )

    # testing
    test_returns = eval_actor(
        env=env,
        actor=actor,
        device=config.device,
        n_episodes=100,
        seed=config.seed,
        render=config.render,
        name=config.name,
    )
    test_log = {
        "test/reward_mean": np.mean(test_returns),
        "test/reward_std": np.std(test_returns),
    }
    if hasattr(env, "get_normalized_score"):
        normalized_score = env.get_normalized_score(test_returns) * 100.0
        test_log["test/normalized_score_mean"] = np.mean(normalized_score)
        test_log["test/normalized_score_std"] = np.std(normalized_score)

    wandb.log(test_log)

    wandb.finish()


if __name__ == "__main__":
    train()
