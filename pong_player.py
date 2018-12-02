import random
import torch
import torch.nn.functional as F
import numpy as np

from pong_env import PongEnv

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.05
EPS_DECAY = 200
EPS_END = 0.05
TARGET_UPDATE = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MyModelClass(torch.nn.Module):
    
    def __init__(self):
        super(MyModelClass, self).__init__()
        self.linear1 = torch.nn.Linear(7, 32)
        self.linear2 = torch.nn.Linear(32, 32)
        #self.linear3 = torch.nn.Linear(20, 32)
        self.output = torch.nn.Linear(32, 3)
        self.steps = 0
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        #x = F.relu(self.linear3(x))
        return self.output(x)


class PongPlayer:

    def __init__(self, save_path, load=False):
        self.build_model()
        self.build_optimizer()
        self.save_path = save_path
        self.steps = 0
        if load:
            try:
                self.load()
            except FileNotFoundError:
                print("Loading failed. ")

        #  set num_saves
        try:
            state = torch.load(self.save_path)
            self.num_saves = state['num_saves']
        except FileNotFoundError:
            self.num_saves = 0

    def build_model(self):
        self.model = MyModelClass().to(device)

    def build_optimizer(self):
        self.optimizer = torch.optim.RMSprop(self.model.parameters())

    def get_action(self, state, _eval=True):
        eps_thresh = EPS_END + (EPS_START - EPS_END) * np.exp(-1.0 * self.steps / EPS_DECAY)
        self.steps += 1
        if random.random() > eps_thresh or _eval:
            with torch.no_grad():
                index = self.model(state).max(1)[1]
                return index.view(1, 1)
        else:
            return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)

    def load(self):
        state = torch.load(self.save_path)
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])

    def save(self):
        state = {
            'num_saves': self.num_saves + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(state, self.save_path)


def play_game(player, render=True):
    # call this function to run your model on the environment
    # and see how it does
    env = PongEnv()
    state = env.reset()
    action = player.get_action(state)
    done = False
    total_reward = 0
    while not done:
        next_state, reward, done, _ = env.step(action)
        if render:
            env.render()
        action = player.get_action(next_state)
        total_reward += reward
    
    env.close()