import gymnasium as gym
from .memory import Memory
from .ppo import PPO
import torch
from .agent import PlayerAgent

def main():
    ############## Hyperparameters ##############
    env_name = "LunarLander-v3"
    # creating environment
    env = gym.make(env_name)
    #月球登陆为8维
    state_dim = env.observation_space.shape[0]
    #行为共4个
    action_dim = 4
    solved_reward = 230         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 50000        # max training episodes
    max_timesteps = 300         # max timesteps in one episode
    n_latent_var = 64           # number of variables in hidden layer
    update_timestep = 2000      # update policy every n timesteps
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    random_seed = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #############################################
    
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
    
    memory = Memory()
    # print(state_dim)

    agent = PlayerAgent(state_dim, action_dim, n_latent_var, device)

    ppo = PPO(agent, lr, betas, gamma, K_epochs, eps_clip, device)
    #print(lr,betas)
    
    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    
    # training loop
    for i_episode in range(1, max_episodes+1): # 最多迭代max_episodes次
        state, _ = env.reset()#初始化（重新玩）
        for t in range(max_timesteps):
            timestep += 1
            
            # Running policy_old:
            #得到老状态的
            action = agent.act(state, memory)
            observation, reward, terminated, truncated, info = env.step(action) #得到（新的状态，奖励，是否终止，额外的调试信息）
            state = observation
            
            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(terminated)
            
            # update if its time
            if timestep % update_timestep == 0:
                ppo.update(memory) # 更新策略
                memory.clear_memory()
                timestep = 0
            
            running_reward += reward

            if terminated or truncated:
                break
                
        avg_length += t
        
        # stop training if avg_reward > solved_reward
        # print(log_interval*solved_reward)
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
            break
            
        # logging
        if i_episode % log_interval == 0: # 每迭代20次打印一次
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

if __name__ == '__main__':
    main()