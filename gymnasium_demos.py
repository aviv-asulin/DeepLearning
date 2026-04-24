# ----------------------------------------------------------------------------------------
# DDQN with Replay Buffer & Target Network
# Supports: CartPole / MountainCar / LunarLander
# ----------------------------------------------------------------------------------------

import gymnasium as gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import pygame

ENV_NAME = "mountaincar"  # options: "cartpole", "mountaincar", "lunarlander"

if ENV_NAME.lower() == "cartpole":
    ENV = "CartPole-v1"
    STATE_SIZE = 4 # nn input size: cart position and velocity, pole angle and angular velocity
    NUM_ACTIONS = 2 # agent action: left or right
    SOLVED_SCORE = 475
    MAX_STEPS = 500 # not in use, frame_idx is the step counter
    SAVE_PATH = "Data/cartpole_dqn.pth"

elif ENV_NAME.lower() == "mountaincar":
    ENV = "MountainCar-v0"
    STATE_SIZE = 2 # nn input: position and velocity
    NUM_ACTIONS = 3 # 0-push left, 1-do nothing, 2-push right
    SOLVED_SCORE = -110
    MAX_STEPS = 200 # not in use, frame_idx is the step counter
    SAVE_PATH = "Data/mountaincar_dqn.pth"

elif ENV_NAME.lower() == "lunarlander":
    ENV = "LunarLander-v3"
    STATE_SIZE = 8      # nn input: x position, y position, x velocity, y velocity, angle,  
                        # angular velocity, left leg contact, right leg contact
    NUM_ACTIONS = 4     # agent action-nn output: 0-do nothing, 1-fire left engine,
                        # 2-fire main engine, 3-fire right engine
    SOLVED_SCORE = 200
    MAX_STEPS = 1000 # not in use, frame_idx is the step counter
    SAVE_PATH = "Data/lunarlander_dqn.pth"
else:
    raise ValueError(f"Unknown ENV_NAME {ENV_NAME}")


# -------------------------------
# CONSTANTS
# -------------------------------
GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 64
TARGET_UPDATE = 1000
MEMORY_SIZE = 10000 # Replay Buffer size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
EPISODES = 1000
TEST_EPISODES = 50

# GAME = True means play no training, using trained file, False meand train
# GUI = True for visual presenting the training process - consume a lot of training time
GUI = False
GAME = True
# -------------------------------
# REPLAY BUFFER
# -------------------------------
class ReplayBuffer:
    def __init__(self, capacity=MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(next_state),
            np.array(done)
        )
    def __len__(self):
        return len(self.buffer)
# -------------------------------
# DQN MODEL
# -------------------------------
class DQN(nn.Module):
    def __init__(self, state_size=STATE_SIZE, num_actions=NUM_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
    def forward(self, x):
        return self.net(x)
    def copy(self):
        new_model = DQN(state_size=STATE_SIZE, num_actions=NUM_ACTIONS)
        new_model.load_state_dict(self.state_dict())
        return new_model
# -------------------------------
# TRAINING STEP
# -------------------------------
def train_dqn(model_online, model_target, optimizer, batch):
    states, actions, rewards, next_states, dones = batch
    states_v = torch.tensor(states, dtype=torch.float32, device=DEVICE)
    actions_v = torch.tensor(actions, dtype=torch.long, device=DEVICE)
    rewards_v = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
    next_states_v = torch.tensor(next_states, dtype=torch.float32, device=DEVICE)
    dones_v = torch.tensor(dones, dtype=torch.float32, device=DEVICE)
    q_values = model_online(states_v)
    # actions_v.unsqueeze(1) - add batch dimention: action_v is (64,) -> (64,1)
    # gather(1,actions_v.unsqueeze(1)) - extract the q_values of the actions_v index
    # squeeze(1) - reduces the batch dimention -> vector of q_values
    state_action_values = q_values.gather(1, actions_v.unsqueeze(1)).squeeze(1)

    with torch.no_grad():   # do not update target dqn weights AND avoid uneccessay chains
        next_q_online = model_online(next_states_v) # next action's q values
        next_actions = next_q_online.argmax(1)      # pick the best action (max q value index)
        next_q_target = model_target(next_states_v) # get all next actions from target DQN
        # calculate the actions q_alues from the target next action multiply by gamma plus reward
        next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        expected = rewards_v + GAMMA * next_q_values * (1 - dones_v)
    # calculate the loss: expected q values (from the target) vs the online q values
    loss = nn.MSELoss()(state_action_values, expected)

    # backpropogation the online
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
# -------------------------------
# TRAIN FUNCTION
# -------------------------------
def train():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    env = gym.make(ENV, render_mode="human" if GUI else None)
    env.reset(seed=SEED)
    model_online = DQN(state_size=STATE_SIZE, num_actions=NUM_ACTIONS).to(DEVICE)
    optimizer = optim.Adam(model_online.parameters(), lr=LR)
    buffer = ReplayBuffer()
    model_target = model_online.copy().to(DEVICE)
    frame_idx = 0
    reward_window = deque(maxlen=50)
    try:
        for episode in range(EPISODES):
            state, _ = env.reset()
            done = False
            total_reward = 0
            while not done:
                epsilon = 0.05 + 0.9 * np.exp(-frame_idx / 500)
                frame_idx += 1
                if random.random() < epsilon:
                    action = random.randint(0, NUM_ACTIONS - 1)
                else:
                    with torch.no_grad():
                        q_vals = model_online(
                            torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                        )
                        action = torch.argmax(q_vals).item()
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                buffer.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if len(buffer) >= BATCH_SIZE:
                    train_dqn(
                        model_online,
                        model_target,
                        optimizer,
                        buffer.sample(BATCH_SIZE)
                    )
                if GUI:
                    env.render()

                if frame_idx % TARGET_UPDATE == 0:
                    model_target.load_state_dict(model_online.state_dict())
            reward_window.append(total_reward)
            avg_reward = sum(reward_window) / len(reward_window)

            if episode % 4 == 0:
                symbol = "|" if (episode // 4) % 2 == 0 else "-"
                print(f"\r{symbol}", end="", flush=True)
            if episode % 50 == 0:
                print(
                    f"\rEpisode {episode}, reward={total_reward:.2f}, "
                    f"avg50={avg_reward:.2f}, epsilon={epsilon:.3f}"
                )

            if avg_reward >= SOLVED_SCORE:
                print(f"\n🎯 Environment solved at episode {episode}!")
                break
    except KeyboardInterrupt:
        print("\n⏹ Training stopped by Ctrl+C")

    finally:
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        torch.save(model_online.state_dict(), SAVE_PATH)
        print(f"💾 Model saved to {SAVE_PATH}")
        env.close()
# -------------------------------
# TEST FUNCTION
# -------------------------------
def test():
    env = gym.make(ENV)
    model = DQN(state_size=STATE_SIZE, num_actions=NUM_ACTIONS).to(DEVICE)
    if os.path.exists(SAVE_PATH):
        model.load_state_dict(
            torch.load(SAVE_PATH, map_location=DEVICE, weights_only=True)
        )
        print("✅ Loaded pretrained model")
    else:
        print("❌ Model not found")
        return
    model.eval()
    total_scores = []
    for _ in range(TEST_EPISODES):
        state, _ = env.reset()
        done = False
        score = 0
        while not done:
            with torch.no_grad():
                q_vals = model(
                    torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                )
                action = torch.argmax(q_vals).item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
        total_scores.append(score)
    avg_score = np.mean(total_scores)
    print(f"✅ Test: Average Score = {avg_score:.2f}")
    env.close()
# -------------------------------
# PLAY FUNCTION
# -------------------------------
def game():
    env = gym.make(ENV, render_mode="human")
    model = DQN(state_size=STATE_SIZE, num_actions=NUM_ACTIONS).to(DEVICE)

    if os.path.exists(SAVE_PATH):
        model.load_state_dict(
            torch.load(SAVE_PATH, map_location=DEVICE, weights_only=True)
        )
        print("✅ Loaded model from disk")
    else:
        print("❌ Cannot find model file")
        return

    model.eval()
    game_num = 0
    total_scores = []
    run = True
    while run:
        state, _ = env.reset()
        done = False
        total_reward = 0
        print(f"New game: {game_num}")
        game_num += 1

        time.sleep(0.5) # between games

        while not done and run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

            with torch.no_grad():
                q_vals = model(
                    torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                )
                action = torch.argmax(q_vals).item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            env.render()

            time.sleep(0.01)

        print(f"Game finished, reward={total_reward}")
        total_scores.append(total_reward)

        #time.sleep(0.3)

        if game_num >= 10:
            run = False
    print(f"Average score over {game_num} games: {np.mean(total_scores):.2f}")
    env.close()
# -------------------------------
# TRAIN & TEST
# -------------------------------
def train_and_test():
    start_time = time.time()
    print("Starting training...")
    train()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training time: {elapsed_time:.2f} sec ({elapsed_time/60:.2f} min)")
    print("Starting testing...")
    test()
# -------------------------------
# MAIN
# -------------------------------
def main():
    if not GAME:
        train_and_test()
    if GAME:
        game()

if __name__ == "__main__":
    main()