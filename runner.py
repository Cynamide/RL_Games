from environment import Vasuki

from gym import Env
from gym.spaces import Discrete, Box, MultiDiscrete
from nqueens import Queen

import numpy as np
import random
import os

import matplotlib
import matplotlib.pyplot as plt

import cv2

from collections import namedtuple, deque
from itertools import count
from base64 import b64encode

import tensorflow as tf

# ----------------------------------------------------- #


class Runner:
    def __init__(self, model_A, model_B, checkpoint):
        # Path to store the Video
        self.checkpoint = checkpoint
        # Defining the Environment
        config = {
            "n": 8,
            "rewards": {"Food": 4, "Movement": -1, "Illegal": -2},
            "game_length": 100,
        }  # Should not change for evaluation
        self.env = Vasuki(**config)
        self.runs = 1
        # Trained Policies
        self.model_A = model_A  # Loaded model with weights
        self.model_B = model_B  # Loaded model with weights
        # Results
        self.winner = {"Player_A": 0, "Player_B": 0}

    def reset(self):
        self.winner = {"Player_A": 0, "Player_B": 0}

    def evaluate_A(self):
        # Uses self.env as the environment and returns the best action for Player A (Blue)
        distA = []
        closeFoodA = [0, 0]
        for food_loc in self.env.live_foodspawn_space:
            distA.append(
                (
                    (
                        ((self.env.agentA["state"][0] - food_loc[0]) ** 2)
                        + ((self.env.agentA["state"][1] - food_loc[1]) ** 2)
                    )
                    ** 0.5
                )
            )
        closeFoodA[1] = min(distA)
        closeFoodA[0] = distA.index(closeFoodA[1])
        obs = self.get_obs(self.env.agentA, self.env.agentB, "agentA", closeFoodA)
        action_A = self.model_A.predict(obs)[0]
        action_A = np.argmax(action_A)
        return action_A  # Action in {0, 1, 2}

    def evaluate_B(self):
        # Uses self.env as the environment and returns the best action for Player B (Red)
        distB = []
        closeFoodB = [0]
        for food_loc in self.env.live_foodspawn_space:
            distB.append(
                (
                    (
                        ((self.env.agentB["state"][0] - food_loc[0]) ** 2)
                        + ((self.env.agentB["state"][1] - food_loc[1]) ** 2)
                    )
                    ** 0.5
                )
            )
        temp = min(distB)
        closeFoodB[0] = distB.index(temp)
        obs = self.get_obs(self.env.agentA, self.env.agentB, "agentB", closeFoodB)
        action_B = self.model_B.predict(obs)[0]
        action_B = np.argmax(action_B)
        return action_B  # Action in {0, 1, 2}

    def visualize(self, run):
        self.env.reset()
        done = False
        video = []
        while not done:
            # Actions based on the current state using the learned policy
            actionA = self.evaluate_A()
            actionB = self.evaluate_B()
            action = {"actionA": actionA, "actionB": actionB}
            rewardA, rewardB, done, info = self.env.step(action)
            # Rendering the enviroment to generate the simulation
            if len(self.env.history) > 1:
                state = self.env.render(actionA, actionB)
                encoded, _ = self.env.encode()
                state = np.array(state, dtype=np.uint8)
                video.append(state)
        # Recording the Winner
        if self.env.agentA["score"] > self.env.agentB["score"]:
            self.winner["Player_A"] += 1
        elif self.env.agentB["score"] > self.env.agentA["score"]:
            self.winner["Player_B"] += 1
        # Generates a video simulation of the game
        if run % 1 == 0:

            aviname = os.path.join(self.checkpoint, f"game_{run}.avi")
            mp4name = os.path.join(self.checkpoint, f"game_{run}.mp4")
            w, h, _ = video[0].shape
            out = cv2.VideoWriter(aviname, cv2.VideoWriter_fourcc(*"DIVX"), 2, (h, w))
            for state in video:
                assert state.shape == (256, 512, 3)
                out.write(state)

            cv2.destroyAllWindows()
            os.popen("ffmpeg -i {input} {output}".format(input=aviname, output=mp4name))
            # os.popen("rm -f {input}".format(input=aviname))

    def arena(self):
        # Pitching the Agents against each other
        for run in range(1, self.runs + 1, 1):
            self.visualize(run)
        return self.winner

    def get_obs(self, agentA, agentB, agent, closeFood):
        scoreA = agentA["score"]
        scoreB = agentB["score"]
        n = 8
        score = 0
        if agent == "agentA":
            state = agentA["state"]
            opp_state = agentB["state"]
            head = agentA["head"]
            velocity = agentA["velocity"]
            if scoreA > scoreB:
                score = 1
        elif agent == "agentB":
            state = agentB["state"]
            opp_state = agentA["state"]
            head = agentB["head"]
            velocity = agentB["velocity"]
            if scoreB > scoreA:
                score = 1
        danger_top = 0
        danger_left = 0
        danger_right = 0
        if head == 0:  # North
            if state[1] == velocity - 1:  # Left Wall
                danger_left = 1
            if state[0] == velocity - 1:  # Top Wall
                danger_top = 1
            if state[1] == n - velocity:  # Right Wall
                danger_right = 1
        elif head == 1:  # East
            if state[0] == velocity - 1:  # Top Wall
                danger_left = 1
            if state[1] == n - velocity:  # Right Wall
                danger_top = 1
            if state[0] == n - velocity:  # Bottom Wall
                danger_right = 1
        elif head == 2:  # South
            if state[1] == n - velocity:  # Right Wall
                danger_left = 1
            if state[0] == n - velocity:  # Bottom Wall
                danger_top = 1
            if state[1] == velocity - 1:  # Left Wall
                danger_right = 1
        elif head == 3:  # West
            if state[0] == n - velocity:  # Bottom Wall
                danger_left = 1
            if state[1] == velocity - 1:  # Left Wall
                danger_top = 1
            if state[0] == velocity - 1:  # Top Wall
                danger_right = 1
        opponent_top = 0
        opponent_bottom = 0
        opponent_left = 0
        opponent_right = 0
        if opp_state[0] < state[0]:
            opponent_top = 1
        if opp_state[0] > state[0]:
            opponent_bottom = 1
        if opp_state[1] < state[1]:
            opponent_left = 1
        if opp_state[1] > state[1]:
            opponent_right = 1
        food_spawn = self.env.live_foodspawn_space[closeFood[0]]
        food_top = 0
        food_bottom = 0
        food_left = 0
        food_right = 0
        if food_spawn[0] < state[0]:
            food_top = 1
        if food_spawn[0] > state[0]:
            food_bottom = 1
        if food_spawn[1] < state[1]:
            food_left = 1
        if food_spawn[1] > state[1]:
            food_right = 1
        obs = [0] * 16
        obs[head] = 1
        obs[4] = danger_top
        obs[5] = danger_left
        obs[6] = danger_right
        obs[7] = opponent_top
        obs[8] = opponent_bottom
        obs[9] = opponent_left
        obs[10] = opponent_right
        obs[11] = food_top
        obs[12] = food_bottom
        obs[13] = food_left
        obs[14] = food_right
        obs[15] = score

        obs = np.array([obs]).reshape(1, -1)
        return obs


model = tf.keras.models.load_model("model-refined.h5")
runner = Runner(model, model, "./")
runner.arena()
