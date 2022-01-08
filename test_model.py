import gym
import numpy as np
import tensorflow as tf
import cv2

frameSize = (400, 600)
out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'mp4v'), 3, frameSize)

model =tf.saved_model.load("Breakout-test-model")
env = gym.make("LunarLander-v2")
state = env.reset().reshape(1,8)
rewar = 0
ep=0
while ep<2:
    img = env.render(mode="rgb_array")
    out.write(img.astype('uint8'))
    pred = model(state)
    pred = np.argsort(pred)
    action = pred[0][3]
    #action = np.argmax(model.predict(state))
    next_state, reward, terminated, info = env.step(action)
    next_state = next_state.reshape(1,8)
    state = next_state
    rewar+=reward
    if terminated:
        env.reset()
        ep+=1
        print(rewar)
        rewar = 0

