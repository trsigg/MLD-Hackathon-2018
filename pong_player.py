import torch
import torch.nn.functional as F

from pong_env import PongEnv

# TODO replace this class with your model
class MyModelClass(torch.nn.Module):
    
    def __init__(self):
        self.linear1 = torch.nn.Linear(7, 32)
        self.linear2 = torch.nn.Linear(32, 32)
        self.output = torch.nn.Linear(32, 3)
        self.steps = 0
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return torch.nn.Softmax(self.output(x))


# TODO fill out the methods of this class
class PongPlayer(object):

    def __init__(self, save_path, load=False):
        self.build_model()
        self.build_optimizer()
        self.save_path = save_path
        if load:
            self.load()

    def build_model(self):
        # TODO: define your model here
        # I would suggest creating another class that subclasses
        # torch.nn.Module. Then you can just instantiate it here.
        # your not required to do this but if you don't you should probably
        # adjust the load and save functions to work with the way you did it.
        self.model = MyModelClass()

    def build_optimizer(self):
        # TODO: define your optimizer here
        optimizer = torch.optim.RMSprop(self.parameters())
        self.optimizer = None

    def get_action(self, state):
        # TODO: this method should return the output of your model
        output = self.model(state)
        index = numpy.random.random_sample()
        for i in output:
            if index < output[i]:
                return i
            index -= output[i]

        raise IndexError

    def reset(self):
        # TODO: this method will be called whenever a game finishes
        # so if you create a model that has state you should reset it here
        # NOTE: this is optional and only if you need it for your model
        pass

    def load(self):
        state = torch.load(self.save_path)
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])

    def save(self):
        state = {
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