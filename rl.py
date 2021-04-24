from utils import *

import torch
import torch.nn as nn
import pandas as pd
from torch.distributions import Categorical
import matplotlib.pyplot as plt

from lol import *
"""
Action space = {1, 2, 3, ..., #Champions}
State space = {
        (
            teamA_ban_1,
            teamB_ban_1,
            teamA_ban_2,
            teamB_ban_2,
            teamA_ban_3,
            teamB_ban_3,
            teamA_pick_1,
            teamB_pick_1,
            teamB_pick_2,
            teamA_pick_2,
            teamA_pick_3,
            teamB_pick_3,
            teamB_ban_4,
            teamA_ban_4,
            teamB_ban_5,
            teamA_ban_5,
            teamB_pick_4,
            teamA_pick_4,
            teamA_pick_5,
            teamB_pick_5
        )
    }
State space = {
        (
            teamA_pick_1,
            teamB_pick_1,
            teamB_pick_2,
            teamA_pick_2,
            teamA_pick_3,
            teamB_pick_3,
            teamB_pick_4,
            teamA_pick_4,
            teamA_pick_5,
            teamB_pick_5
        )
    }

"""
# TEAM_A_PICK_INDICES = [6, 9, 10, 17, 18]
# TEAM_B_PICK_INDICES = [7, 8, 11, 16, 19]
# BAN_INDICES = list(range(0, 6)) + list(range(12, 16))
TEAM_A_PICK_INDICES = [0, 3, 4, 7, 8]
TEAM_B_PICK_INDICES = [1, 2, 5, 6, 9]


class Policy(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(Policy, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.layers = nn.Sequential(
            nn.Linear(self.state_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_size),
            nn.Softmax(dim=1)
        )
    def forward(self, input):
        """
        Given the current state, output the action probabilities.
        The input has dim=(self.state_size,). The output has dim=
        (self.action_size,)
        """
        return self.layers(input)

def get_team_indices(team_bit):
    if team_bit == 0:
        return TEAM_A_PICK_INDICES
    else:
        return TEAM_B_PICK_INDICES

def train(
    policy_A,
    policy_B,
    winrate_predictor,
    episodes=10,
    lr=0.01,
):
    policy_A.train()
    policy_B.train()

    optimizer = torch.optim.Adam(policy_A.parameters())
    outcomes = []
    winrates = []
    for i in range(episodes):
        # policy_A is going against policy_B that uses random strategy
        draft, states, actions, penalties, reward = sample_draft(
            policy_A,
            policy_B,
            0,
            winrate_predictor
        )
        outcomes.append(reward)
        action_odds = policy_A(states)
        sampler = Categorical(action_odds)
        log_probs = -sampler.log_prob(actions.add(-1))
        loss = torch.sum(log_probs * reward)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        winrate = sum(outcomes) / len(outcomes)
        winrates.append(winrate)
        
    torch.save(policy_A, "team_a.pt")
    plt.plot(outcomes)
    plt.show()
    plt.plot(winrates)
    plt.show()

def sample_draft(
    policy_A,
    policy_B,
    learning_agent,
    winrate_predictor
):
    """
    Return the sampled draft result according to policy_A and policy_B
    and the total reward for the learning_agent.
    """
    state_size = policy_A.state_size
    action_size = policy_A.action_size
    team_indices = get_team_indices(learning_agent)
    policy_A.eval()
    policy_B.eval()
    winrate_predictor.eval()

    starting_state = -1 * torch.ones((1, state_size), requires_grad=True).float()
    draft = torch.empty(state_size)
    states = []
    actions = []
    penalties = []
    round = 0
    while not is_done(starting_state):
        # action_odds have size (action_size,)
        if round in team_indices:
            action_odds = policy_A(starting_state)
        else:
            action_odds = policy_B(starting_state)

        sampler = Categorical(action_odds)
        action = sampler.sample() + 1

        penalty = torch.tensor(
            [get_penalty(starting_state, action, winrate_predictor)]
        )

        if round in team_indices:
            penalties.append(penalty)
            states.append(starting_state)
            actions.append(action)
        draft[round] = action
        round += 1
        starting_state = transition(starting_state, action)
    result = get_result(starting_state, winrate_predictor)
    if result == learning_agent:
        reward = 1
    else:
        reward = 0

    states = torch.cat(states, dim=0).float()
    actions = torch.cat(actions, dim=0).long()
    penalties = torch.cat(penalties, dim=0).long()
    return draft, states, actions, penalties, reward

def violate_rule(state, action):
    """
    Check if the action is a violation of any rules given the current
    state, ie. picking or banning a champion that already exists in the
    draft.
    """
    if action in state:
        return True
    return False

def penalty_reward(reward):
    """
    Check if the given reward is a result of a rule violation
    """
    if reward < 0:
        return True
    return False

def is_done(state):
    """
    state: A torch.tensor with dim (20,)

    Check if state is done, ie. all picking and banning are finished.
    """
    if -1 not in state:
        return True
    return False 

def get_result(state, winrate_predictor):
    """
    Get result of the game given state.
    0 -> Team A Victory
    1 -> Team B Victory
    """
    teamA_picks = state[:, TEAM_A_PICK_INDICES]
    teamB_picks = state[:, TEAM_B_PICK_INDICES]
    team_comp = torch.cat((teamA_picks, teamB_picks), dim=1)
    winrate = winrate_predictor(team_comp)[0, 0]
    
    if winrate >= 0.5:
        return 0
    return 1 

def get_penalty(state, action, winrate_predictor):
    """
    Return the reward of action given current state.
    """
    if violate_rule(state, action):
        return -1 
    return 0

def transition(state, action):
    """
    Trainsition to the next state given state and action.
    """
    state_cpy = state.clone()
    assert(-1 in state_cpy)
    pos = tuple((state_cpy == -1).nonzero()[0])

    state_cpy[pos] = float(action)
    return state_cpy

if __name__ == "__main__":

    champs = pd.read_csv("champions.csv")
    winrate_predictor = torch.load("winrate_predictor.pt")
    state_size = 10
    hidden_size = 50
    action_size = champs.shape[0]

    torch.manual_seed(11) 
    policy_A = Policy(state_size, action_size, hidden_size)
    policy_B = Policy(state_size, action_size, hidden_size)

    train(
        policy_A,
        policy_B,
        winrate_predictor,
        episodes=10000,
        lr=0.01
    )
