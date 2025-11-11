import torch
from torch.distributions import Categorical
from .model import ActorCritic


class PlayerAgent:
    def __init__(self, state_dim, action_dim, n_latent_var, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_latent_var = n_latent_var
        self.device = device

        self.policy = self.create_policy()

    def create_policy(self):
        return ActorCritic(self.state_dim, self.action_dim, self.n_latent_var).to(
            self.device
        )

    def attempt(self, state, memory):
        """
        尝试性的选择一个动作,并将状态、动作、动作概率和对数概率存储到内存中。
        """

        state = torch.tensor(state).float().to(self.device)
        action_probs = self.policy.forward(state)
        dist = Categorical(action_probs)  # 按照给定的概率分布来进行采样
        action = (
            dist.sample()
        )  # 从分布中采样一个动作,并返回动作的索引。这里使用sample而非argmax的原因是因为我们要根据动作的概率分布来进行采样,而不是直接选择概率最高的动作。以一定概率选择非最优动作，尝试未知的可能性，相当于 “冒险尝试新路径，可能发现更好的选择”。

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def act(self, state):
        """
        选择一个最优的动作
        """

        state = torch.tensor(state).float().to(self.device)
        action_probs = self.policy.forward(state)
        action = torch.argmax(action_probs, dim=-1).item()

        return action
