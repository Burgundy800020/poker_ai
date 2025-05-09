from agents.agent import Agent
from gym_env import PokerEnv, WrappedEval
import random
import numpy as np
from treys import Card, Evaluator

action_types = PokerEnv.ActionType

evaluator = WrappedEval()

def compute_probs(observation):
    res = np.ones(27).astype(np.float64)
    for i in observation["my_cards"]: res[i] = 0
    for i in observation["community_cards"]: res[i] = 0
    if (not observation["opp_discarded_card"] == -1):
        res[observation["opp_discarded_card"]] = 0
        res[observation["opp_drawn_card"]] = 0
    return res / np.sum(res)

#Return win ratio
def monte_carlo(observation, iterations=500):
    probs = compute_probs(observation)
    win_count = 0
    lose_count = 0
    com_cards = observation["community_cards"].index(-1) if observation["community_cards"][-1] == -1 else 5
    missing_com_cards = 5 - com_cards
    cards_needed = (2 if observation["opp_drawn_card"] == -1 else 1) + missing_com_cards
    for i in range(iterations):
        cards = np.random.choice(27, cards_needed, p = probs, replace = False)
        
        simulated_community_cards = list(observation["community_cards"][:com_cards]) + list(cards[:missing_com_cards])
        if (observation["opp_drawn_card"] == -1):
            simulated_opponent_cards = cards[missing_com_cards:missing_com_cards+2]
        else:
            simulated_opponent_cards = [observation["opp_drawn_card"], cards[missing_com_cards]]
        simulated_community_cards = list(map(PokerEnv.int_to_card, simulated_community_cards))
        simulated_opponent_cards = list(map(PokerEnv.int_to_card, simulated_opponent_cards))

        my_rank = evaluator.evaluate(list(map(PokerEnv.int_to_card, observation["my_cards"])), simulated_community_cards)
        op_rank = evaluator.evaluate(simulated_opponent_cards, simulated_community_cards)
        if (my_rank < op_rank):
            win_count += 1
        elif (op_rank < my_rank):
            lose_count += 1
    return win_count / (lose_count + 1e-8)

def discard_monte_carlo(observation, iterations=500):
    probs = compute_probs(observation)
    win_count_0 = 0
    lose_count_0 = 0
    win_count_1 = 0
    lose_count_1 = 0
    com_cards = observation["community_cards"].index(-1) if observation["community_cards"][-1] == -1 else 5
    missing_com_cards = 5 - com_cards
    cards_needed = (2 if observation["opp_drawn_card"] == -1 else 1) + missing_com_cards
    for i in range(iterations):
        cards = np.random.choice(27, cards_needed+1, p = probs, replace = False)
        
        simulated_community_cards = list(observation["community_cards"][:com_cards]) + list(cards[:missing_com_cards])
        if (observation["opp_drawn_card"] == -1):
            simulated_opponent_cards = cards[missing_com_cards:missing_com_cards+2]
        else:
            simulated_opponent_cards = [observation["opp_drawn_card"], cards[missing_com_cards]]
        my_new_card = cards[-1]
        simulated_community_cards = list(map(PokerEnv.int_to_card, simulated_community_cards))
        simulated_opponent_cards = list(map(PokerEnv.int_to_card, simulated_opponent_cards))
        my_simulated_hand_0 = list(map(PokerEnv.int_to_card, [observation["my_cards"][0], my_new_card]))
        my_simulated_hand_1 = list(map(PokerEnv.int_to_card, [observation["my_cards"][1], my_new_card]))
        my_rank_0 = evaluator.evaluate(my_simulated_hand_0, simulated_community_cards)
        my_rank_1 = evaluator.evaluate(my_simulated_hand_1, simulated_community_cards)
        op_rank = evaluator.evaluate(simulated_opponent_cards, simulated_community_cards)
        if (my_rank_0 < op_rank):
            win_count_0 += 1
        elif (my_rank_0 > op_rank):
            lose_count_0 += 1
        if (my_rank_1 < op_rank):
            win_count_1 += 1
        elif (my_rank_1 > op_rank):
            lose_count_1 += 1
    ratio_0 = win_count_0 / (lose_count_0 + 1e-8)
    ratio_1 = win_count_1 / (lose_count_1 + 1e-8)
    return 1 if ratio_0 >= ratio_1 else 0, max(ratio_0, ratio_1)


class PlayerAgent(Agent):
    def __name__(self):
        return "PlayerAgent"

    def __init__(self, stream: bool = True):
        super().__init__(stream)
        # Initialize any instance variables here
        self.hand_number = 0
        self.last_action = None
        self.won_hands = 0
        self.winnings = 0

    def act(self, observation, reward, terminated, truncated, info):
        # Example of using the logger
        if observation["street"] == 0 and info["hand_number"] % 50 == 0:
            self.logger.info(f"Hand number: {info['hand_number']}")
        # First, get the list of valid actions we can take
        # valid_actions = observation["valid_actions"]

        if (self.winnings/3 > (1000-info["hand_number"]+1) / 2):
            return action_types.FOLD.value, 0, -1

        win_ratio = monte_carlo(observation)
        if (win_ratio < 0.9 and observation["valid_actions"][action_types.DISCARD.value] and observation["street"]==1):
            best_discard, new_win_ratio = discard_monte_carlo(observation)
            return action_types.DISCARD.value, 0, best_discard
        elif (win_ratio < 0.9 * ((observation["opp_bet"]-observation["my_bet"])/100)):
            if (observation["valid_actions"][action_types.DISCARD.value]): return action_types.DISCARD.value, 0, 0
            return action_types.FOLD.value, 0, -1
        elif (win_ratio > 1.05 and observation["valid_actions"][action_types.RAISE.value]):
            return action_types.RAISE.value, observation["max_raise"], -1
        
        if observation["valid_actions"][action_types.CALL.value]:
            action_type = action_types.CALL.value
        else:
            action_type = action_types.CHECK.value
        raise_amount = 0
        card_to_discard = -1
        return action_type, raise_amount, card_to_discard

    def observe(self, observation, reward, terminated, truncated, info):
        # Log interesting events when observing opponent's actions
        if terminated:
            self.logger.info(f"Game ended with reward: {reward}")
            self.hand_number += 1
            if reward > 0:
                self.won_hands += 1
            self.last_action = None
        else:
            # log observation keys
            self.logger.info(f"Observation keys: {observation}")
    
    def observe(self, observation, reward, terminated, truncated, info):
        if terminated:
            self.winnings += reward