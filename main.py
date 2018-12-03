from pong_player import *


if __name__ == '__main__':
    env = PongEnv()
    env.reset()
    state = env.state
    player = PongPlayer('train1t.pt', True)
    done = False
    while not done:
        state_tensor = torch.tensor(state, device=device, dtype=torch.float32).view(1, -1)
        state, _, done, _ = env.step(player.get_action(state_tensor).item())
        env.render()

    env.close()
