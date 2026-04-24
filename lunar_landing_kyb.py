import gymnasium as gym
import pygame
import time

# --- PARAMETERS ---
NUM_EPISODES = 10
SLEEP_TIME = 1
FPS = 5

# --- Initialize environment ---
env = gym.make("LunarLander-v3", render_mode="human")
state, _ = env.reset()

# --- Initialize Pygame ---
pygame.init()
screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption("LunarLander Manual Control")
clock = pygame.time.Clock()

print("Controls:")
print("LEFT  arrow  -> left engine")
print("RIGHT arrow  -> right engine")
print("UP    arrow  -> main engine")
print("ESC          -> quit")
print("-------------------------------------------------\n")

# --- MAIN LOOP ---
stop = False
for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    done = False
    total_reward = 0
    if stop:
        break

    while not done and not stop:

        # --- Handle quit events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                stop = True
                break
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                done = True

        # --- Continuous keyboard control ---
        keys = pygame.key.get_pressed()

        action = 0  # default: do nothing

        if keys[pygame.K_LEFT]:
            action = 1
        elif keys[pygame.K_UP]:
            action = 2
        elif keys[pygame.K_RIGHT]:
            action = 3

        # --- Step environment ---
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # --- Slow down for visibility ---
        # time.sleep(SLEEP_TIME)
        clock.tick(FPS)

    print(f"Episode {episode+1} ended. Total reward: {total_reward}")
    time.sleep(SLEEP_TIME)

# --- Close everything ---
env.close()
pygame.quit()
print("Demo finished")