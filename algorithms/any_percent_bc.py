from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import asdict, dataclass
from datetime import datetime
import os
from pathlib import Path
import random
import uuid

import d4rl
import gym
import numpy as np
import pandas as pd
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import wandb

TensorBatch = List[torch.Tensor]


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "halfcheetah-medium-expert-v2"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    log_every: int = 100 # How often (time steps) we log
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    # BC
    buffer_size: int = 2_000_000  # Replay buffer size
    frac: float = 0.1  # Best data fraction to use
    max_traj_len: int = 1000  # Max trajectory length
    normalize: bool = True  # Normalize states
    actor_LN: bool = True  # Use LayerNorm for actor
    soft: bool = False  # Use soft BC
    # Wandb logging
    project: str = "offline-RL-init"
    group: str = "BC-D4RL"
    name: str = "BC"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}"#{str(uuid.uuid4())[:8]}
        if self.checkpoints_path is not None:
            time=datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            self.checkpoints_path = os.path.join(self.checkpoints_path, f"{time}_{self.name}")


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


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
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

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
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

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
    wandb.run.save()


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    # Max demonstration lengths for each environment from human data
    max_demonstration_lengths = {'pen': 100, 'door': 300, 'hammer': 624, 'relocate': 527}
    max_demonstration_length = None
    for s in max_demonstration_lengths.keys():
        if s in env.spec.id:
            max_demonstration_length = max_demonstration_lengths[s]
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        timestep=0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = env.step(action)
            timestep+=1
            episode_reward += reward
            if max_demonstration_length is not None and timestep < max_demonstration_length:
                done = False
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


def keep_best_trajectories(
    dataset: Dict[str, np.ndarray],
    frac: float,
    discount: float,
    max_episode_steps: int = 1000,
):
    ids_by_trajectories = []
    returns = []
    cur_ids = []
    cur_return = 0
    reward_scale = 1.0
    for i, (reward, done) in enumerate(zip(dataset["rewards"], dataset["terminals"])):
        cur_return += reward_scale * reward
        cur_ids.append(i)
        reward_scale *= discount
        if done == 1.0 or len(cur_ids) == max_episode_steps:
            ids_by_trajectories.append(list(cur_ids))
            returns.append(cur_return)
            cur_ids = []
            cur_return = 0
            reward_scale = 1.0

    sort_ord = np.argsort(returns, axis=0)[::-1].reshape(-1)
    top_trajs = sort_ord[: int(frac * len(sort_ord))]

    order = []
    for i in top_trajs:
        order += ids_by_trajectories[i]
    order = np.array(order)

    dataset["observations"] = dataset["observations"][order]
    dataset["actions"] = dataset["actions"][order]
    dataset["next_observations"] = dataset["next_observations"][order]
    dataset["rewards"] = dataset["rewards"][order]
    dataset["terminals"] = dataset["terminals"][order]


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        actor_LN: bool = True,
        soft: bool = True,
    ):
        super(Actor, self).__init__()
        self.action_dim = action_dim
        self.soft = soft

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
            # nn.Linear(256, action_dim),
            # nn.Tanh(),
        )
        self.mu = nn.Linear(256, action_dim)
        self.log_sigma = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state: torch.Tensor, deterministic=False) -> torch.Tensor:
        if self.soft:
            hidden = self.net(state)
            mu = self.mu(hidden)
            if deterministic:
                scaled_action = self.max_action * torch.tanh(mu)
                log_prob = None
            else:
                log_sigma = self.log_sigma(hidden)
                log_sigma = torch.clip(log_sigma, -5, 2)
                policy_dist = Normal(mu, torch.exp(log_sigma))
                action = policy_dist.rsample()
                tanh_action = torch.tanh(action)
                # change of variables formula (SAC paper, appendix C, eq 21)
                log_prob = policy_dist.log_prob(action).sum(axis=-1)
                log_prob = log_prob - torch.log(1 - tanh_action.pow(2) + 1e-6).sum(
                    axis=-1
                )
                scaled_action = self.max_action * tanh_action
        else:
            hidden = self.net(state)
            scaled_action = torch.tanh(self.mu(hidden))
            log_prob = None
        return scaled_action, log_prob

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        hidden = self.net(state)
        mu = self.mu(hidden)
        log_sigma = self.log_sigma(hidden)
        log_sigma = torch.clip(log_sigma, -5, 2)
        policy_dist = Normal(mu, torch.exp(log_sigma))
        action = torch.clip(action, -self.max_action + 1e-6, self.max_action - 1e-6)
        log_prob = policy_dist.log_prob(torch.arctanh(action)).sum(axis=-1)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(axis=-1)
        return log_prob

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state, deterministic=True)[0].cpu().data.numpy().flatten()


class BC:  # noqa
    def __init__(
        self,
        max_action: np.ndarray,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        discount: float = 0.99,
        soft: bool = False,
        device: str = "cpu",
    ):
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.max_action = max_action
        self.discount = discount
        self.soft = soft

        self.total_it = 0
        self.device = device
        self.log_alpha = torch.tensor(
            [0.0], dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-4)
        self.alpha = self.log_alpha.exp().detach()
        self.target_entropy = -float(self.actor.action_dim)

    def _alpha_loss(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action, action_log_prob = self.actor(state)

        loss = (-self.log_alpha * (action_log_prob + self.target_entropy)).mean()

        return loss

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        state, action, _, _, _ = batch

        # Compute actor loss
        if self.soft:
            alpha_loss = self._alpha_loss(state)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().detach()

            # Compute actor loss
            pi, log_pi = self.actor(state)
            log_prob_action = self.actor.log_prob(state, action)
            actor_loss = (self.alpha * log_pi - log_prob_action).mean()
        else:
            pi, _ = self.actor(state)
            actor_loss = F.mse_loss(pi, action)
        log_dict["actor_loss"] = actor_loss.item()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.total_it = state_dict["total_it"]


@pyrallis.wrap()
def train(config: TrainConfig):
    env = gym.make(config.env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    dataset = env.get_dataset()

    if config.frac != 1.0:
        keep_best_trajectories(dataset, config.frac, config.discount)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    if "next_observations" not in dataset.keys():
        dataset["next_observations"] = np.roll(
            dataset["observations"], shift=-1, axis=0
        )  # Terminals/timeouts block next observations
        print("Loaded next state observations from current state observations.")

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    max_action = float(env.action_space.high[0])

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    actor = Actor(state_dim, action_dim, max_action, config.actor_LN, config.soft).to(
        config.device
    )
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "discount": config.discount,
        "soft": config.soft,
        "device": config.device,
    }

    print("---------------------------------------")
    print(f"Training BC, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize policy
    trainer = BC(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(asdict(config))

    evaluations = []
    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
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
            )
            eval_log = {
                "eval/reward_mean": np.mean(eval_returns),
                "eval/reward_std": np.std(eval_returns),
                "epoch": int((t+1)/1000),
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
                    os.path.join(config.checkpoints_path, f"checkpoint_{int((t+1)/1000)}.pt"),
                )
                checkpoints = [os.path.join(config.checkpoints_path, file) for file in os.listdir(config.checkpoints_path) if os.path.splitext(file)[-1]=='.pt']
                checkpoints.sort(key=os.path.getmtime)
                if len(checkpoints) > 10:
                    oldest_checkpoint = checkpoints.pop(0)
                    os.remove(oldest_checkpoint)
                df = pd.DataFrame({"epoch": int((t+1)/1000), "return_mean": np.mean(eval_returns), "return_std": np.std(eval_returns), "normalized_score_mean": np.mean(normalized_score), "normalized_score_std": np.std(normalized_score)}, index=[0])
                if not os.path.exists(os.path.join(config.checkpoints_path, "results.csv")):
                    df.to_csv(os.path.join(config.checkpoints_path, "results.csv"), index=False)
                else:
                    df.to_csv(os.path.join(config.checkpoints_path, "results.csv"), mode='a', header=False, index=False)


if __name__ == "__main__":
    train()
