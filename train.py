from pong_player import *
from collections import namedtuple
import itertools
from tqdm import tqdm

player = PongPlayer('train2t.pt', True)
env = PongEnv()

target_net = MyModelClass().to(device)
target_net.load_state_dict(player.model.state_dict())
target_net.eval()


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayBuffer:

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.index = 0

    def add(self, state, action, next_state, reward):
        if len(self.buffer) < self.capacity:
            self.buffer.append([])

        self.buffer[self.index] = Transition(state, action, next_state, reward)
        self.index = (self.index + 1) % self.capacity

    def get_sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


replay_buffer = ReplayBuffer(10000)


def train_step():
    if len(replay_buffer.buffer) < BATCH_SIZE:
        return

    samples = replay_buffer.get_sample(BATCH_SIZE)
    batch = Transition(*zip(*samples))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device,
                                  dtype=torch.uint8)

    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_batch_values = player.model.forward(state_batch)

    state_action_values = state_batch_values.gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values,
                            expected_state_action_values.unsqueeze(1))

    # Optimize the model
    player.optimizer.zero_grad()
    loss.backward()
    for param in player.model.parameters():
        # clamp gradients between -1 and 1
        param.grad.data.clamp_(-1, 1)
    player.optimizer.step()


def print_reward_stats(rewards):
    print("Mean reward: ", np.mean(rewards))
    print("Max reward: ", np.max(rewards))
    print("Num rewards: ", len(rewards))


num_episodes = 200
rewards = []
for i_episode in tqdm(range(num_episodes)):
    env.reset()
    state = torch.tensor(env.state, device=device, dtype=torch.float32).view(1, -1)
    reward_sum = 0

    try:
        for t in itertools.count():
            # Select and perform an action
            action = player.get_action(state, False)
            next_state, reward, done, _ = env.step(action.item())
            reward_sum += reward
            reward = torch.tensor([reward], device=device)

            next_state = torch.tensor(next_state, device=device, dtype=torch.float32).view(1, -1)
            # Store the transition in memory
            replay_buffer.add(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            train_step()
            if done:
                rewards.append(reward_sum)
                break
    except KeyboardInterrupt:
        print("Interrupt detected.")
        if input("Save? ") != 'n':
            player.save()
        if input("Kill? ") == 'y':
            exit()

    if i_episode % 10 == 0:
        print_reward_stats(rewards[-10:])
    # Update the target network
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(player.model.state_dict())

player.save()
print_reward_stats(rewards)
print("Num saves: ", player.num_saves)
