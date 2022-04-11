import gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
import random
import time

from tqdm.auto import tqdm

env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
env.reset()


def get_actor_model(action_space, freeze_features=True):
    '''Creates an actor model that returns policy for actions in the action_space.'''

    # MobileNet for the features extraction
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    if freeze_features:
        model.features.requires_grad = False # freeze weights

    # custom classifier
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=False),
        torch.nn.Linear(in_features=1280, out_features=1000),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=1000, out_features=action_space)
    )
    
    return model.cuda()


def get_critic_model(freeze_features=True):
    '''Creates critic model.'''

    # loads pretrained MobileNet for features extraction
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    if freeze_features:
        model.features.requires_grad = False # freeze weights
    
    # custom classifier
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=False),
        torch.nn.Linear(in_features=1280, out_features=1000),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=1000, out_features=1),
        torch.nn.ReLU()
    )

    return model.cuda()


def screen_to_tensor(screen):
    '''Converts screenshot to tensor.'''

    return torch.Tensor(screen).permute(2, 0, 1).unsqueeze(0).cuda()


class Trajectory:
    '''Class representing the saved trajectory.'''

    def __init__(self, traj_states, traj_actions, traj_log_probs, traj_rewards, discount_factor=0.998):
        self.states = torch.stack(traj_states)
        self.log_probs = torch.cat([traj_log_probs[x] for x in traj_actions])
        self.actions = torch.stack(traj_actions).view(-1, 1)
        self.rewards = []
        self.action_probs = torch.diagonal(self.log_probs[:, self.actions[:, 0]])

        # calculate rewards
        discounted_reward = 0
        for r in reversed(traj_rewards):
            discounted_reward = discounted_reward * discount_factor + r
            self.rewards.append(discounted_reward)
        self.rewards = torch.Tensor(self.rewards).cuda().view(-1, 1)


class Dataset:
    '''Class responsible for keeping the trajectories' dataset.'''

    def __init__(self, trajectories_limit):
        self.trajectories_limit = trajectories_limit
        self.trajectories = []  
        self._index = 0

    def add(self, new_traj):
        if len(self.trajectories) + 1 == self.trajectories_limit:
            self.trajectories = self.trajectories[1:]
            
        self.trajectories.append(new_traj)       

    def sample(self, batch=16):
        return random.sample(self.trajectories, batch)

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if len(self.trajectories) >= self._index:
            raise StopIteration
        self._index += 1
        return self.trajectories[self._index - 1]


def play_test(actor_model):
    '''Visualize example game using the actor model's policy.'''
    
    actor_model.eval()
    env.reset()
    _, _, _, info = env.step(0)

    state = screen_to_tensor(info['rgb'])
    images = []
    
    while True:    
        with torch.no_grad():
            log_probs = actor_model(state)
            action = np.random.choice(env.action_space.n, p=torch.nn.functional.softmax(log_probs, 1)[0].detach().cpu().numpy())

        observation, reward, done, info = env.step(action)

        state = screen_to_tensor(info['rgb'])
        images.append(info['rgb'])

        if done:
            break

    def animate_func(i):
        img.set_array(images[i])
        return [img]

    fig = plt.figure(figsize=(20, 20))
    img = plt.imshow(images[0], interpolation='none', aspect='auto', animated=True)
    anim = animation.FuncAnimation(fig, animate_func, frames=len(images), interval=1000/120, blit=True, repeat=False)
    plt.axis('off')
    plt.show()
    plt.close()


def collect_training_data(actor_model, critic_model, dataset, iters=1):
    '''Samples trajectories and adds them to the dataset.'''

    print('Playing games...')
    for traj in tqdm(range(iters)):
        states = []
        actions = []
        rewards = []
        log_probs = []

        env.reset()
        _, _, _, info = env.step(0)
        state = screen_to_tensor(info['rgb'])

        while True:
            states.append(state.squeeze(0))

            log_prob = actor_model(state)
            log_probs.append(log_prob.detach())

            action = np.random.choice(env.action_space.n, p=torch.nn.functional.softmax(log_prob, 1)[0].detach().cpu().numpy())
            actions.append(torch.tensor(action).cuda())

            observation, reward, done, info = env.step(action)
            rewards.append(reward)

            state = screen_to_tensor(info['rgb'])

            if done:
                break

        dataset.add(Trajectory(states, actions, log_probs, rewards))


class PPOLoss(torch.nn.Module):
    def __init__(self):
        super(PPOLoss, self).__init__()
    
    def forward(new_policy, old_policy, action_values, eps=0.2):
        ratio = torch.exp(new_policy - old_policy)
        return -min(ratio * action_values, torch.clip(torch.exp(ratio), 1 - eps, 1 + eps) * action_values)


def train_models(actor_model, critic_model, dataset, learning_rate, weight_decay=0.9, steps=1000):
    actor_optim = torch.optim.AdamW(actor_model.parameters(), learning_rate, weight_decay=weight_decay)
    critic_optim = torch.optim.AdamW(critic_model.parameters(), learning_rate, weight_decay=weight_decay)    
    critic_loss = torch.nn.MSELoss()
    actor_loss = PPOLoss()

    for _ in tqdm(range(steps)):
        actor_optim.zero_grad()
        critic_optim.zero_grad()

        for traj in dataset:
            states = traj.states.detach()
            old_policy = traj.action_probs.detach()
            rewards = traj.rewards.detach()
            actions = traj.actions.detach()
            old_action_probs = traj.action_probs.detach()

            action_values = critic_model(states)
            loss = critic_loss(action_values, rewards) / steps # mean
            loss.backward()

            log_probs = actor_model(states)
            new_action_probs = torch.diagonal(log_probs[:, actions[:, 0]])
            loss = actor_loss(new_action_probs, old_action_probs, action_values.detach()) / steps
            loss.backward()

        actor_optim.step()
        critic_optim.step()


ACTOR_MODEL_WEIGHTS = ''
CRITIC_MODEL_WEIGHTS = ''

ITERS = 1
DATA_COLLECTION_STEPS = 100
LR = 1e-4

actor = get_actor_model(env.action_space.n)
critic = get_critic_model()
dataset = Dataset(4098)

for i in tqdm(range(ITERS)):
    collect_training_data(actor, critic, dataset, ITERS)
    train_models(actor, critic, dataset, ITERS)    
    break

env.close()
