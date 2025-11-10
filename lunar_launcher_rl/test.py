import gymnasium as gym
from .memory import Memory
import torch
from .agent import PlayerAgent
from .device import deault_device

def test():
    ############## Hyperparameters ##############
    env_name = "LunarLander-v3"
    # creating environment
    env = gym.make(env_name, render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = 4
    max_timesteps = 300
    n_latent_var = 64           # number of variables in hidden layer

    device = deault_device
    
    #############################################

    n_episodes = 3


    filename = "PPO_{}.pth".format(env_name)
    directory = "./models/"
    
    memory = Memory()
    player_agent = PlayerAgent(state_dim, action_dim, n_latent_var, device)
    
    player_agent.policy.load_state_dict(torch.load(directory+filename),strict=False)
    
    for ep in range(1, n_episodes+1):
        ep_reward = 0
        state, _ = env.reset()
        for t in range(max_timesteps):
            action = player_agent.act(state, memory)
            observation, reward, terminated, truncated, info = env.step(action)    
            state = observation

            ep_reward += reward
        
            if terminated or truncated:
                observation, info = env.reset()
                break
            
        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        ep_reward = 0

    env.close()
    
if __name__ == '__main__':
    test()
    
    