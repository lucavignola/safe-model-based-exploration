import pickle

import ffmpeg
import numpy as np
from dm_control import suite

with open('save_trajectory.pkl', 'rb') as file:
    trajectory = pickle.load(file)

trajectory = np.array(trajectory)

# Load the CartPole environment
env = suite.load(domain_name="cartpole", task_name="balance")

# Print qpos and qvel names
print("qpos names:", env.physics.named.data.qpos)  # To identify correct names
print("qvel names:", env.physics.named.data.qvel)  # To identify correct names


# Function to set the state using the correct keys
def set_cartpole_state(env, state):
    physics = env.physics
    # Update these keys based on the actual print output
    physics.data.qpos[0] = state[0]  # cart position
    physics.data.qpos[1] = np.pi - state[1]  # pole angle
    physics.data.qvel[0] = state[2]  # cart velocity
    physics.data.qvel[1] = state[3]  # pole angular velocity
    physics.step()


# Function to visualize the trajectory and capture frames
def capture_frames(env, trajectory, width=640, height=480):
    frames = []
    for idx, state in enumerate(trajectory):
        set_cartpole_state(env, state)
        frame = env.physics.render(camera_id=0, width=width, height=height)
        frames.append(frame)
    return frames


# Capture frames
frames = capture_frames(env, trajectory)

# Save frames as video
video_file = 'unsafe_cartpole_trajectory.mp4'
process = (
    ffmpeg
    .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='640x480')
    .output(video_file, pix_fmt='yuv420p')
    .overwrite_output()
    .run_async(pipe_stdin=True)
)

for frame in frames:
    process.stdin.write(frame.tobytes())

process.stdin.close()
process.wait()
