# Inspired by:
# 1. paper for SAC-N: https://arxiv.org/abs/2110.01548
# 2. implementation: https://github.com/snu-mllab/EDAC
from typing import Any, Dict, List, Optional, Tuple, Union
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import os
import random
import uuid

import d4rl
import gym
import numpy as np
import pyrallis
import torch
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

import wandb


@dataclass
class TrainConfig:
    # wandb params
    project: str = "offline-RL-init"
    group: str = "MSG-D4RL"
    name: str = "MSG"
    # model params
    hidden_dim: int = 256
    num_critics: int = 10
    gamma: float = 0.99
    tau: float = 5e-3
    eta: float = 0.0
    beta: float = 0.0
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    max_action: float = 1.0
    # training params
    buffer_size: int = 1_000_000
    env_name: str = "halfcheetah-medium-v2"
    batch_size: int = 256
    num_epochs: int = 3000
    num_updates_on_epoch: int = 1000
    normalize_reward: bool = False
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 5
    # general params
    checkpoints_path: Optional[str] = None
    deterministic_torch: bool = False
    train_seed: int = 10
    eval_seed: int = 42
    log_every: int = 100
    device: str = "cpu"
    pretrain: Optional[str] = None  # BC or AC or C or softAC or softC
    pretrain_epochs: int = 10
    actor_LN: bool = True
    critic_LN: bool = True

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


# general utils
TensorBatch = List[torch.Tensor]


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
    )
    wandb.run.save()


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


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def discount_cumsum(x, discount, include_first=True):
    disc_cumsum = np.zeros_like(x)
    if include_first:
        disc_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            disc_cumsum[t] = x[t] + discount * disc_cumsum[t + 1]
    else:
        disc_cumsum[-1] = 0
        for t in reversed(range(x.shape[0] - 1)):
            disc_cumsum[t] = discount * x[t + 1] + discount * disc_cumsum[t + 1]
    return disc_cumsum


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


# def augment_states_with_timestep(data: np.ndarray) -> np.ndarray:
#     n_transitions = data["observations"].shape[0]
#     episode_step = 0.0
#     data["observations"] = np.append(
#         data["observations"], np.zeros((n_transitions, 1)), axis=1
#     )
#     if "next_observations" in data.keys():
#         data["next_observations"] = np.append(
#             data["next_observations"], np.zeros((n_transitions, 1)), axis=1
#         )
#     for i in range(n_transitions):
#         data["observations"][int(episode_step), -1] = episode_step
#         if "next_observations" in data.keys():
#             data["next_observations"][int(episode_step), -1] = episode_step + 1.0
#         episode_step += 1.0
#         if data["terminals"][i] or data["timeouts"][i] or i == n_transitions - 1:
#             if "next_observations" in data.keys():
#                 data["next_observations"][int(episode_step - 1.0), -1] = 0.0
#             episode_step = 0
#     return data


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std

    def scale_reward(reward):
        return reward_scale * reward

    # env = gym.wrappers.TimeAwareObservation(env)
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
        self._action_dim = action_dim
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
        self._next_returns_to_go = torch.zeros(
            (buffer_size, 1), dtype=torch.float32, device=device
        )
        self._entropy_bonuses = torch.zeros(
            (buffer_size, 1), dtype=torch.float32, device=device
        )
        self._soft_returns_to_go = torch.zeros(
            (buffer_size, 1), dtype=torch.float32, device=device
        )
        self._next_soft_returns_to_go = torch.zeros(
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
            episode_rewards.append(
                data["rewards"][i]
            )  # - 1* self._action_dim* torch.log(1 / torch.sqrt(torch.tensor(2 * np.pi))))
            if (
                data["terminals"][i] or data["timeouts"][i] or i == n_transitions - 1
            ):  # TODO: Deal with incomplete trajectory case
                episode_returns_to_go = discount_cumsum(
                    np.array(episode_rewards), self._discount
                )
                returns_to_go.append(episode_returns_to_go)
                episode_rewards = []

        returns_to_go = np.array(
            [
                return_to_go
                for episode_returns in returns_to_go
                for return_to_go in episode_returns
            ]
        ).flatten()

        next_returns_to_go = np.roll(
            returns_to_go, shift=-1, axis=0
        )  # Terminals/timeouts block next returns to go
        assert next_returns_to_go[0] == returns_to_go[1]

        self._returns_to_go[:n_transitions] = self._to_tensor(returns_to_go[..., None])
        self._next_returns_to_go[:n_transitions] = self._to_tensor(
            next_returns_to_go[..., None]
        )

    def compute_soft_returns_to_go(
        self, data: np.ndarray, alpha: torch.Tensor, actor: "Actor"
    ) -> np.ndarray:
        n_transitions = self._states.shape[0]
        episode_rewards = []
        episode_entropy_bonuses = []
        soft_returns_to_go = []

        entropy_batch_size = 512
        with torch.no_grad():
            for i in range((n_transitions // entropy_batch_size) + 1):
                batch_states = self._states[
                    entropy_batch_size
                    * i : min(entropy_batch_size * (i + 1), n_transitions)
                ]
                pi, log_pi = actor(
                    batch_states,
                    need_log_prob=True,
                )
                self._entropy_bonuses[
                    entropy_batch_size
                    * i : min(entropy_batch_size * (i + 1), n_transitions)
                ] = (-log_pi.detach().unsqueeze(-1).cpu())

        for i in range(n_transitions):
            episode_rewards.append(self._rewards[i].cpu().item())
            episode_entropy_bonuses.append(self._entropy_bonuses[i].cpu().item())
            if self._dones[i] or i == n_transitions - 1:
                episode_returns_to_go = discount_cumsum(
                    np.array(episode_rewards), self._discount
                ) + alpha.detach().cpu().item() * discount_cumsum(
                    np.array(episode_entropy_bonuses),
                    self._discount,
                    include_first=False,
                )
                soft_returns_to_go.append(episode_returns_to_go)
                episode_rewards = []
                episode_entropy_bonuses = []

        soft_returns_to_go = np.array(
            [
                return_to_go
                for episode_returns in soft_returns_to_go
                for return_to_go in episode_returns
            ]
        ).flatten()

        next_soft_returns_to_go = np.roll(
            soft_returns_to_go, shift=-1, axis=0
        )  # Terminals/timeouts block next soft returns to go
        assert next_soft_returns_to_go[0] == soft_returns_to_go[1]

        self._soft_returns_to_go[:n_transitions] = self._to_tensor(
            soft_returns_to_go[..., None]
        )
        self._next_soft_returns_to_go[:n_transitions] = self._to_tensor(
            next_soft_returns_to_go[..., None]
        )
        self._soft_returns_loaded = True

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )

        # import pickle

        # with open("halfcheetah_data.pkl", "wb") as fp:
        #     pickle.dump(data, fp)

        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])

        self._dones[:n_transitions] = self._to_tensor(
            (data["terminals"] + data["timeouts"])[..., None]
        )
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        self.compute_returns_to_go(data)
        self._soft_returns_loaded = False

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        if self._soft_returns_loaded:
            returns_to_go = self._soft_returns_to_go[indices]
            next_returns_to_go = self._next_soft_returns_to_go[indices]
        else:
            returns_to_go = self._returns_to_go[indices]
            next_returns_to_go = self._next_returns_to_go[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [
            states,
            actions,
            rewards,
            returns_to_go,
            next_states,
            next_returns_to_go,
            dones,
        ]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        raise NotImplementedError


# SAC Actor & Critic implementation
class VectorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # default pytorch init for nn.Linear module
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [ensemble_size, batch_size, input_size]
        # weight: [ensemble_size, input_size, out_size]
        # out: [ensemble_size, batch_size, out_size]
        return x @ self.weight + self.bias


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        max_action: float = 1.0,
        actor_LN: bool = True,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, elementwise_affine=False)
            if actor_LN
            else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, elementwise_affine=False)
            if actor_LN
            else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, elementwise_affine=False)
            if actor_LN
            else nn.Identity(),
            nn.ReLU(),
        )
        # with separate layers works better than with Linear(hidden_dim, 2 * action_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_sigma = nn.Linear(hidden_dim, action_dim)

        # init as in the EDAC paper
        for layer in self.trunk[::3]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.mu.bias, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_sigma.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_sigma.bias, -1e-3, 1e-3)

        self.action_dim = action_dim
        self.max_action = max_action

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        need_log_prob: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden = self.trunk(state)
        mu, log_sigma = self.mu(hidden), self.log_sigma(hidden)
        # print(torch.exp(log_sigma).mean())

        # clipping params from EDAC paper, not as in SAC paper (-20, 2)
        log_sigma = torch.clip(log_sigma, -5, 2)
        policy_dist = Normal(mu, torch.exp(log_sigma))

        if deterministic:
            action = mu
        else:
            action = policy_dist.rsample()

        tanh_action, log_prob = torch.tanh(action), None
        if need_log_prob:
            # change of variables formula (SAC paper, appendix C, eq 21)
            log_prob = policy_dist.log_prob(action).sum(axis=-1)
            log_prob = log_prob - torch.log(1 - tanh_action.pow(2) + 1e-6).sum(axis=-1)

        return tanh_action * self.max_action, log_prob

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        hidden = self.trunk(state)
        mu, log_sigma = self.mu(hidden), self.log_sigma(hidden)
        # print(mu.mean())
        # print(torch.exp(log_sigma).mean())

        # clipping params from EDAC paper, not as in SAC paper (-20, 2)
        log_sigma = torch.clip(log_sigma, -5, 2)
        policy_dist = Normal(mu, torch.exp(log_sigma))

        # action_distribution = TransformedDistribution(
        #     policy_dist, TanhTransform(cache_size=1)
        # )
        # return torch.sum(action_distribution.log_prob(action), dim=-1)

        # if torch.any(action <= -1) or torch.any(action >= 1):
        #     print(action)
        log_prob = policy_dist.log_prob(
            torch.arctanh(torch.clip(action, -1 + 1e-6, 1 - 1e-6))
        ).sum(axis=-1)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(axis=-1)
        return log_prob

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str) -> np.ndarray:
        deterministic = not self.training
        state = torch.tensor(state, device=device, dtype=torch.float32)
        action = self(state, deterministic=deterministic)[0].cpu().numpy()
        return action


class VectorizedCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_critics: int,
        critic_LN: bool = True,
    ):
        super().__init__()
        self.critic = nn.Sequential(
            VectorizedLinear(state_dim + action_dim, hidden_dim, num_critics),
            nn.LayerNorm(hidden_dim, elementwise_affine=False)
            if critic_LN
            else nn.Identity(),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.LayerNorm(hidden_dim, elementwise_affine=False)
            if critic_LN
            else nn.Identity(),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.LayerNorm(hidden_dim, elementwise_affine=False)
            if critic_LN
            else nn.Identity(),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, 1, num_critics),
        )
        # init as in the EDAC paper
        for layer in self.critic[::3]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.critic[-1].weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.critic[-1].bias, -3e-3, 3e-3)

        self.num_critics = num_critics

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # [..., batch_size, state_dim + action_dim]
        state_action = torch.cat([state, action], dim=-1)
        if state_action.dim() != 3:
            assert state_action.dim() == 2
            # [num_critics, batch_size, state_dim + action_dim]
            state_action = state_action.unsqueeze(0).repeat_interleave(
                self.num_critics, dim=0
            )
        assert state_action.dim() == 3
        assert state_action.shape[0] == self.num_critics
        # [num_critics, batch_size]
        q_values = self.critic(state_action).squeeze(-1)
        return q_values


class MSG:
    def __init__(
        self,
        actor: Actor,
        pretrain_actor_optimizer: torch.optim.Optimizer,
        actor_optimizer: torch.optim.Optimizer,
        actor_scheduler: torch.optim.lr_scheduler,
        critic: VectorizedCritic,
        pretrain_critic_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        critic_scheduler: torch.optim.lr_scheduler,
        gamma: float = 0.99,
        tau: float = 0.005,
        eta: float = 1.0,
        beta: float = 0.0,
        alpha_learning_rate: float = 1e-4,
        device: str = "cpu",  # noqa
    ):
        self.device = device

        self.actor = actor
        self.critic = critic
        with torch.no_grad():
            self.target_critic = deepcopy(self.critic)

        self.pretrain_actor_optimizer = pretrain_actor_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_scheduler = actor_scheduler
        self.pretrain_critic_optimizer = pretrain_critic_optimizer
        self.critic_optimizer = critic_optimizer
        self.critic_scheduler = critic_scheduler

        self.tau = tau
        self.gamma = gamma
        self.eta = eta
        self.beta = beta

        # adaptive alpha setup
        self.target_entropy = -float(self.actor.action_dim)
        self.log_alpha = torch.tensor(
            [0.0], dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_learning_rate)
        self.alpha = self.log_alpha.exp().detach()

    def _alpha_loss(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action, action_log_prob = self.actor(state, need_log_prob=True)

        loss = (-self.log_alpha * (action_log_prob + self.target_entropy)).mean()

        return loss

    def _actor_loss(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, float, float]:
        pi, log_pi = self.actor(state, need_log_prob=True)  # , deterministic=True

        # log_prob_action = self.actor.log_prob(state, action)

        q_value_dist = self.critic(state, pi)
        assert q_value_dist.shape[0] == self.critic.num_critics
        # q_value_min = q_value_dist.min(0).values
        q_value_min = q_value_dist.mean(0) + self.beta * q_value_dist.std(0)
        # needed for logging
        q_value_std = q_value_dist.std(0).mean().item()
        batch_entropy = -log_pi.mean().item()

        assert log_pi.shape == q_value_min.shape
        loss = (
            self.alpha * log_pi
            - q_value_min
            # - self.beta * log_prob_action
        ).mean()  # (1 / q_value_min.abs().mean().detach()) *

        # self.beta -= 0.0001
        # self.beta = max(self.beta, 0.0)

        return loss, batch_entropy, q_value_std

    def _critic_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_action, next_action_log_prob = self.actor(
                next_state, need_log_prob=True  # , deterministic=True
            )
            q_next = self.target_critic(next_state, next_action)
            # q_nexts = self.target_critic(next_state, next_action)
            # q_next = q_nexts.mean(0) - 2 * q_nexts.std(0)
            q_next = q_next - self.alpha * next_action_log_prob.unsqueeze(0)
            assert q_next.shape[1] == done.shape[0] == reward.shape[0]
            q_target = reward.view(1, -1) + self.gamma * (1 - done.view(1, -1)) * q_next

        q_values = self.critic(state, action)
        # [ensemble_size, batch_size] - [1, batch_size]
        critic_loss = ((q_values - q_target) ** 2).mean(dim=1).sum(dim=0)
        # diversity_loss = self._critic_diversity_loss(state, action)
        pi, _ = self.actor(state, need_log_prob=False)
        q_policy_values = self.critic(state, pi)

        support_regulariser = (q_policy_values - q_values).mean(dim=1).sum(dim=0)

        # loss = (1 / critic_loss.abs().mean().detach()) *
        loss = critic_loss + self.eta * support_regulariser

        return loss

    def pretrain_BC(self, batch: TensorBatch) -> Dict[str, float]:
        state, action, reward, return_to_go, next_state, next_return_to_go, done = batch
        # Alpha update
        alpha_loss = self._alpha_loss(state)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # Compute actor loss
        pi, log_pi = self.actor(state, need_log_prob=True)
        log_prob_action = self.actor.log_prob(state, action)
        actor_loss = (self.alpha * log_pi - log_prob_action).mean()

        # Optimize the actor
        self.pretrain_actor_optimizer.zero_grad()
        actor_loss.backward()
        self.pretrain_actor_optimizer.step()

        log_dict = {
            "alpha_loss": alpha_loss.item(),
            "actor_loss": actor_loss.item(),
            "batch_entropy": -log_pi.mean().item(),
            "alpha": self.alpha.item(),
        }

        return log_dict

    def pretrain_actorcritic(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}

        state, action, reward, return_to_go, next_state, next_return_to_go, done = batch
        # Alpha update
        # alpha_loss = self._alpha_loss(state)
        # self.alpha_optimizer.zero_grad()
        # alpha_loss.backward()
        # self.alpha_optimizer.step()

        # self.alpha = self.log_alpha.exp().detach()
        # Compute actor loss
        pi, action_log_prob = self.actor(state, deterministic=True, need_log_prob=True)
        actor_loss = F.mse_loss(
            pi, action
        ) + self.alpha * self.actor.action_dim * torch.log(
            1 / torch.sqrt(torch.tensor(2 * np.pi))
        )
        # print(f"alpha: {self.alpha}")
        # print(f"action_log_prob: {action_log_prob.mean()}")
        # log_probs = self.actor.log_prob(state, action)
        # actor_loss = (self.alpha * action_log_prob - log_probs).mean()
        log_dict["actor_loss"] = actor_loss.item()
        log_dict["batch_entropy"] = -action_log_prob.mean().item()
        # Optimize the actor
        self.pretrain_actor_optimizer.zero_grad()
        actor_loss.backward()
        self.pretrain_actor_optimizer.step()

        # Compute critic loss
        q = self.critic(state, action)
        diversity_loss = self._critic_diversity_loss(state, action)
        critic_loss = (
            F.mse_loss(
                q,
                return_to_go.squeeze()
                .unsqueeze(0)
                .repeat_interleave(self.critic.num_critics, dim=0),
            )
            + self.eta * diversity_loss
        )
        log_dict["critic_loss"] = critic_loss.item()
        # Optimize the critic
        self.pretrain_critic_optimizer.zero_grad()
        critic_loss.backward()
        self.pretrain_critic_optimizer.step()

        # Update the frozen target models
        with torch.no_grad():
            soft_update(self.target_critic, self.critic, self.tau)

        return log_dict

    def pretrain_critic(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}

        state, action, reward, return_to_go, next_state, next_return_to_go, done = batch

        # Compute critic loss
        q = self.critic(state, action).mean(dim=0)
        diversity_loss = self._critic_diversity_loss(state, action)
        critic_loss = F.mse_loss(q, return_to_go.squeeze()) + self.eta * diversity_loss
        log_dict["critic_loss"] = critic_loss.item()
        # Optimize the critic
        self.pretrain_critic_optimizer.zero_grad()
        critic_loss.backward()
        self.pretrain_critic_optimizer.step()

        # Update the frozen target models
        soft_update(self.target_critic, self.critic, self.tau)

        return log_dict

    def pretrain_soft_critic(
        self, batch: TensorBatch, epoch, pretrain_epochs
    ) -> Dict[str, float]:
        state, action, reward, return_to_go, next_state, next_return_to_go, done = batch

        # Compute critic loss
        # q = self.critic(state, action)  # .mean(0)  # .min(0).values
        # max_action = self.actor.max_action

        # OOD_loss = 0
        # for i in range(100):
        # random_actions = -max_action + 2 * max_action * torch.rand_like(action)
        # random_actions = torch.clip(
        #     action + 0.1 * torch.randn_like(action), min=-max_action, max=max_action
        # )
        # policy_actions = self.actor(state)
        # q_random = self.critic(state, random_actions)
        # OOD_loss = asymmetric_l2_loss(q_random - q, 0.99)

        # diversity_loss = self._critic_diversity_loss(state, action)
        # beta = float(epoch - (pretrain_epochs // 2)) / float(pretrain_epochs // 2)
        beta = 0.0
        # F.mse_loss(
        #     q,
        #     return_to_go.squeeze()
        #     .unsqueeze(0)
        #     .repeat_interleave(self.critic.num_critics, dim=0),
        #     reduction="none",
        # )
        # .mean(dim=1)
        # .sum(dim=0)
        # q = self.critic(state, action)
        # critic_loss = (
        #     # (1 - beta)*
        #     ((q - return_to_go.view(1, -1)) ** 2).mean(dim=1).sum(dim=0)
        #     # ((q.mean(0) - return_to_go) ** 2).mean()
        #     # + beta * ((self._critic_loss(state, action, reward, next_state, done)))
        #     + 100 * self.eta * diversity_loss
        # )
        # + 1000 * OOD_loss
        #
        with torch.no_grad():
            next_action, next_action_log_prob = self.actor(
                next_state, need_log_prob=True  # , deterministic=True
            )
            # q_next_network = (
            #     self.target_critic(next_state, next_action).min(0).values.unsqueeze(-1)
            # )

            q_next = next_return_to_go - self.alpha * next_action_log_prob.unsqueeze(-1)
            q_target = reward + self.gamma * (1 - done) * q_next

        q_values = (
            self.critic(state, action).mean(0)
            + self.beta * self.critic(state, action).std(0)
        ).view(1, -1)
        pi, _ = self.actor(state, need_log_prob=False)
        q_policy_values = self.critic(state, pi)

        support_regulariser = (q_policy_values - q_values).mean(dim=1).sum(dim=0)
        # # [ensemble_size, batch_size] - [1, batch_size]
        critic_loss = (
            ((q_values - q_target.view(1, -1)) ** 2).mean(dim=1).sum(dim=0)
        ) + self.eta * support_regulariser
        # + self.eta * diversity_loss)
        # critic_loss = support_regulariser + 100 * self.eta * diversity_loss

        # critic_loss = q - return_to_go.squeeze().unsqueeze(0).repeat_interleave(
        #     self.critic.num_critics, dim=0
        # )  # + 100 * self.eta * diversity_loss
        # critic_loss = (
        #     asymmetric_l2_loss(critic_loss, 0.99) + 1000 * self.eta * diversity_loss
        # )

        # Optimize the critic
        self.pretrain_critic_optimizer.zero_grad()
        critic_loss.backward()
        self.pretrain_critic_optimizer.step()

        #  Target networks soft update
        with torch.no_grad():
            soft_update(self.target_critic, self.critic, self.tau)
            # for logging, Q-ensemble std estimate with the random actions:
            # a ~ U[-max_action, max_action]
            max_action = self.actor.max_action
            random_actions = -max_action + 2 * max_action * torch.rand_like(action)

            q_random_std = self.critic(state, random_actions).std(0).mean().item()

        log_dict = {
            "critic_loss": critic_loss.item(),
            "beta": beta,
            # "diversity_loss": diversity_loss.item(),
            "q_random_std": q_random_std,
        }

        return log_dict

    def update(self, batch: TensorBatch) -> Dict[str, float]:
        state, action, reward, return_to_go, next_state, next_return_to_go, done = [
            arr.to(self.device) for arr in batch
        ]
        # Usually updates are done in the following order: critic -> actor -> alpha
        # But we found that EDAC paper uses reverse (which gives better results)

        # Alpha update
        alpha_loss = self._alpha_loss(state)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # Actor update
        actor_loss, actor_batch_entropy, q_policy_std = self._actor_loss(state, action)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Critic update
        critic_loss = self._critic_loss(state, action, reward, next_state, done)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #  Target networks soft update
        with torch.no_grad():
            soft_update(self.target_critic, self.critic, tau=self.tau)
            # for logging, Q-ensemble std estimate with the random actions:
            # a ~ U[-max_action, max_action]
            max_action = self.actor.max_action
            random_actions = -max_action + 2 * max_action * torch.rand_like(action)

            q_random_std = self.critic(state, random_actions).std(0).mean().item()

        update_info = {
            "alpha_loss": alpha_loss.item(),
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "batch_entropy": actor_batch_entropy,
            "alpha": self.alpha.item(),
            "q_policy_std": q_policy_std,
            "q_random_std": q_random_std,
            "actor_lr": [group["lr"] for group in self.actor_optimizer.param_groups][0],
            "critic_lr": [group["lr"] for group in self.critic_optimizer.param_groups][
                0
            ],
        }
        return update_info

    def state_dict(self) -> Dict[str, Any]:
        state = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "log_alpha": self.log_alpha.item(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "critic_optim": self.critic_optimizer.state_dict(),
            "alpha_optim": self.alpha_optimizer.state_dict(),
        }
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_critic.load_state_dict(state_dict["target_critic"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optim"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optim"])
        self.alpha_optimizer.load_state_dict(state_dict["alpha_optim"])
        self.log_alpha.data[0] = state_dict["log_alpha"]
        self.alpha = self.log_alpha.exp().detach()


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: Actor, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.array(episode_rewards)


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


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


@pyrallis.wrap()
def train(config: TrainConfig):
    set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)
    wandb_init(asdict(config))

    # data, evaluation, env setup
    # data, evaluation, env setup
    env = gym.make(config.env_name)

    state_dim = env.observation_space.shape[0]  # + 1  # Timestep
    action_dim = env.action_space.shape[0]

    d4rl_dataset = env.get_dataset()

    # d4rl_dataset = augment_states_with_timestep(d4rl_dataset)

    if config.normalize_reward:
        modify_reward(d4rl_dataset, config.env_name)

    state_mean, state_std = compute_mean_std(d4rl_dataset["observations"], eps=1e-3)

    d4rl_dataset["observations"] = normalize_states(
        d4rl_dataset["observations"], state_mean, state_std
    )
    if "next_observations" not in d4rl_dataset.keys():
        d4rl_dataset["next_observations"] = np.roll(
            d4rl_dataset["observations"], shift=-1, axis=0
        )  # Terminals/timeouts block next observations
        print("Loaded next state observations from current state observations.")

    d4rl_dataset["next_observations"] = normalize_states(
        d4rl_dataset["next_observations"], state_mean, state_std
    )

    eval_env = wrap_env(env, state_mean=state_mean, state_std=state_std)

    buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=config.buffer_size,
        discount=config.gamma,
        device=config.device,
    )
    buffer.load_d4rl_dataset(d4rl_dataset)

    # Actor & Critic setup
    actor = Actor(
        state_dim, action_dim, config.hidden_dim, config.max_action, config.actor_LN
    )
    actor.to(config.device)
    pretrain_actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=5 * config.actor_learning_rate
    )
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate)
    # actor_scheduler = torch.optim.lr_scheduler.LinearLR(
    #     actor_optimizer, start_factor=0.01, total_iters=500
    # )
    actor_scheduler = torch.optim.lr_scheduler.ConstantLR(
        actor_optimizer, factor=1, total_iters=25
    )
    critic = VectorizedCritic(
        state_dim, action_dim, config.hidden_dim, config.num_critics, config.critic_LN
    )
    critic.to(config.device)
    pretrain_critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=5 * config.critic_learning_rate
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=config.critic_learning_rate
    )
    # critic_scheduler = torch.optim.lr_scheduler.LinearLR(
    #     critic_optimizer, start_factor=0.01, total_iters=500
    # )
    critic_scheduler = torch.optim.lr_scheduler.ConstantLR(
        critic_optimizer, factor=1, total_iters=25
    )

    trainer = MSG(
        actor=actor,
        pretrain_actor_optimizer=pretrain_actor_optimizer,
        actor_optimizer=actor_optimizer,
        actor_scheduler=actor_scheduler,
        critic=critic,
        pretrain_critic_optimizer=pretrain_critic_optimizer,
        critic_optimizer=critic_optimizer,
        critic_scheduler=critic_scheduler,
        gamma=config.gamma,
        tau=config.tau,
        eta=config.eta,
        beta=config.beta,
        alpha_learning_rate=config.alpha_learning_rate,
        device=config.device,
    )
    # saving config to the checkpoint
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    total_updates = 0.0
    # reset_optimisers = True
    for epoch in trange(config.num_epochs, desc="Training"):
        # training
        for _ in trange(config.num_updates_on_epoch, desc="Epoch", leave=False):
            batch = buffer.sample(config.batch_size)
            if config.pretrain is not None:
                if epoch <= config.pretrain_epochs:
                    if config.pretrain == "BC":
                        update_info = trainer.pretrain_BC(batch)
                    elif config.pretrain == "AC":
                        update_info = trainer.pretrain_actorcritic(batch)
                    elif config.pretrain == "C":
                        update_info = trainer.pretrain_critic(batch)
                    elif config.pretrain == "softAC":
                        if epoch < config.pretrain_epochs // 2:
                            update_info = trainer.pretrain_BC(batch)
                        else:
                            if buffer._soft_returns_loaded == False:
                                buffer.compute_soft_returns_to_go(
                                    data=d4rl_dataset,
                                    alpha=trainer.alpha,
                                    actor=trainer.actor,
                                )
                                print("Soft returns to go loaded for BC actor!")
                            assert buffer._soft_returns_loaded == True
                            update_info = trainer.pretrain_soft_critic(
                                batch, epoch, config.pretrain_epochs
                            )
                    elif config.pretrain == "softC":
                        if buffer._soft_returns_loaded == False:
                            buffer.compute_soft_returns_to_go(
                                data=d4rl_dataset,
                                alpha=trainer.alpha,
                                actor=trainer.actor,
                            )
                            print("Soft returns to go loaded for initialised actor!")
                        assert buffer._soft_returns_loaded == True
                        update_info = trainer.pretrain_soft_critic(
                            batch, epoch, config.pretrain_epochs
                        )
                    else:
                        raise ValueError(
                            f"Pretrain type {config.pretrain} not recognised."
                        )
                    # if t == 10000:
                    #     trainer.actor_target = copy.deepcopy(trainer.actor)
                    #     trainer.critic_1_target = copy.deepcopy(trainer.critic_1)
                    #     trainer.critic_2_target = copy.deepcopy(trainer.critic_2)
                else:
                    update_info = trainer.update(batch)
                    # if reset_optimisers:
                    #     trainer.actor_optimizer = torch.optim.Adam(
                    #         actor.parameters(), lr=config.actor_learning_rate
                    #     )
                    #     trainer.critic_optimizer = critic_optimizer = torch.optim.Adam(
                    #         critic.parameters(), lr=config.critic_learning_rate
                    #     )
                    #     reset_optimisers = False
            else:
                update_info = trainer.update(batch)

            if total_updates % config.log_every == 0:
                wandb.log({"epoch": epoch, **update_info})

            total_updates += 1

        if config.pretrain is not None:
            if epoch > config.pretrain_epochs:
                trainer.actor_scheduler.step()
                trainer.critic_scheduler.step()

        # evaluation
        if epoch % config.eval_every == 0 or epoch == config.num_epochs - 1:
            eval_returns = eval_actor(
                env=eval_env,
                actor=actor,
                n_episodes=config.eval_episodes,
                seed=config.eval_seed,
                device=config.device,
            )
            eval_log = {
                "eval/reward_mean": np.mean(eval_returns),
                "eval/reward_std": np.std(eval_returns),
                "epoch": epoch,
            }
            if hasattr(eval_env, "get_normalized_score"):
                normalized_score = eval_env.get_normalized_score(eval_returns) * 100.0
                eval_log["eval/normalized_score_mean"] = np.mean(normalized_score)
                eval_log["eval/normalized_score_std"] = np.std(normalized_score)

            wandb.log(eval_log)

            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"{epoch}.pt"),
                )

    wandb.finish()


if __name__ == "__main__":
    train()
