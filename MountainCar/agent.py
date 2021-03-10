import gym
from time import sleep
import numpy as np
import matplotlib.pyplot as plt

# Basic Q Learning Lesson

# The Algorithm is as follows:
# 1. Initialize Q(s1, s2, a) by setting all of the elements equal to small random values;
# 2. Observe the current state, (s1, s2) 
# 3. Based on the exploration strategy, choose and action to take, "a"
# 4. Take action a and observe the resulting reward, r,  and the new state of the environment, (s1', s2')
# 5. Update Q(s1, s2, a) based on the update rule:
#
# Q'(s1, s2, a) = (1 - w)*Q(s1, s2, a) + w*(r+d*Q(s1', s2', argmax a'Q(s1, s2, s')))
# Where w is the learning rate and d is the discount rate;
#
# 6. Repeat steps 2-5 until convergence.


class Agent:

    def __init__(self):
        self.env = gym.make('MountainCar-v0')
        self.env.reset()
        self.action_set = [0, 1, 2]
        self.reward_list = []
        self.avg_reward_list = []
        self.analyze_environment()
    
    def analyze_environment(self):
        """
        Have the agent learn about it's surroundings
        """
        observation_space = self.env.observation_space
        action_space = self.env.action_space
        self.low_x = observation_space.low[0]
        self.low_y = observation_space.low[1]
        self.high_x = observation_space.high[0]
        self.high_y = observation_space.high[1]

        print(f"------- Analyzing Environment -------")
        print(f"State space: {observation_space}")
        print(f"State space Low: {(self.low_x, self.low_y)}")
        print(f"State space high: {(self.high_x, self.high_y)}")
        print(f"Action space: {action_space}")
        print(f"-------------------------------------")


    # Define Q-Learning function
    def QLearning(self, learning, discount, epsilon, min_eps, episodes):
        # Determine size of discretized state space
        num_states = (self.env.observation_space.high - self.env.observation_space.low)*np.array([10, 100])
        num_states = np.round(num_states, 0).astype(int) + 1

        # Initialize Q table
        Q = np.random.uniform(
            low = -1, high = 1, size = (num_states[0], num_states[1], self.env.action_space.n)
        )

        # Calculate episodic reduction in epsilon
        reduction = (epsilon - min_eps)/episodes

        # Run Q learning algorithm
        for i in range(episodes):
            # Initialize parameters
            done = False
            tot_reward, reward = 0,0
            state = self.env.reset()
            
            # Discretize state
            state_adj = (state - self.env.observation_space.low)*np.array([10, 100])
            state_adj = np.round(state_adj, 0).astype(int)

            while done != True:   
                # Render environment for last five episodes
                if i >= (episodes - 20):
                    sleep(0.2)
                    self.env.render()
                    
                # Determine next action - epsilon greedy strategy
                if np.random.random() < 1 - epsilon:
                    action = np.argmax(Q[state_adj[0], state_adj[1]]) 
                else:
                    action = np.random.randint(0, self.env.action_space.n)
                    
                # Get next state and reward
                state2, reward, done, info = self.env.step(action) 
                
                # Discretize state2
                state2_adj = (state2 - self.env.observation_space.low)*np.array([10, 100])
                state2_adj = np.round(state2_adj, 0).astype(int)
                
                #Allow for terminal states
                if done and state2[0] >= 0.5:
                    Q[state_adj[0], state_adj[1], action] = reward
                    
                # Adjust Q value for current state
                else:
                    delta = learning*(reward + 
                                     discount*np.max(Q[state2_adj[0], 
                                                       state2_adj[1]]) - 
                                     Q[state_adj[0], state_adj[1],action])
                    Q[state_adj[0], state_adj[1],action] += delta
                                         
                # Update variables
                tot_reward += reward
                state_adj = state2_adj
            
            # Decay epsilon
            if epsilon > min_eps:
                epsilon -= reduction
            
            # Track rewards
            self.reward_list.append(tot_reward)
            
            if (i+1) % 100 == 0:
                ave_reward = np.mean(self.reward_list)
                self.avg_reward_list.append(ave_reward)
                self.reward_list = []
                
            if (i+1) % 100 == 0:    
                print('Episode {} Average Reward: {}'.format(i+1, ave_reward))
                    
        self.env.close()

        return self.avg_reward_list

def main():
    
    A = Agent()
    rewards = A.QLearning(0.2, 0.9, 0.8, 0, 5000)

    # Plot Rewards
    plt.plot(100*(np.arange(len(rewards)) + 1), rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Episodes')
    plt.savefig('rewards.jpg')     
    plt.close()  

if __name__=="__main__":
    main()