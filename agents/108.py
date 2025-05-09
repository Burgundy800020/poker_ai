from agents.agent import Agent, Observation
from gym_env import PokerEnv, WrappedEval
import random
import numpy as np
from treys import Card, Evaluator

action_types = PokerEnv.ActionType

evaluator = WrappedEval()




def evaluate_hand(cards):
        my_cards, opp_cards, community_cards = cards
        my_cards = list(map(PokerEnv.int_to_card, my_cards))
        opp_cards = list(map(PokerEnv.int_to_card, opp_cards))
        community_cards = list(map(PokerEnv.int_to_card, community_cards))
        my_hand_rank = evaluator.evaluate(my_cards, community_cards)
        opp_hand_rank = evaluator.evaluate(opp_cards, community_cards)
        return my_hand_rank, opp_hand_rank

def fast_monte_carlo(observation, num_simulations = 500):
    my_cards = [int(card) for card in observation["my_cards"]]
    community_cards = [card for card in observation["community_cards"] if card != -1]
    opp_discarded_card = [observation["opp_discarded_card"]] if observation["opp_discarded_card"] != -1 else []
    opp_drawn_card = [observation["opp_drawn_card"]] if observation["opp_drawn_card"] != -1 else []

    # Calculate equity through Monte Carlo simulation
    shown_cards = my_cards + community_cards + opp_discarded_card + opp_drawn_card
    non_shown_cards = [i for i in range(27) if i not in shown_cards]

    # Run Monte Carlo simulation
    win_count = 0
    for i in range(num_simulations):
        if (drawn_cards := random.sample(non_shown_cards, 7 - len(community_cards) - len(opp_drawn_card))):
            my_hand_rank, opp_hand_rank = evaluate_hand((my_cards, opp_drawn_card + drawn_cards[: 2 - len(opp_drawn_card)], community_cards + drawn_cards[2 - len(opp_drawn_card) :]))
            if (my_hand_rank < opp_hand_rank):
                win_count += 1
    return win_count / num_simulations 


def discard_monte_carlo(observation, num_simulations = 500):
    my_cards = [int(card) for card in observation["my_cards"]]
    community_cards = [card for card in observation["community_cards"] if card != -1]
    opp_discarded_card = [observation["opp_discarded_card"]] if observation["opp_discarded_card"] != -1 else []
    opp_drawn_card = [observation["opp_drawn_card"]] if observation["opp_drawn_card"] != -1 else []

    # Calculate equity through Monte Carlo simulation
    shown_cards = my_cards + community_cards + opp_discarded_card + opp_drawn_card
    non_shown_cards = [i for i in range(27) if i not in shown_cards]

    win_rates = [0, 0]
    for discard in (0, 1):     
        win_count = 0
        kept_cards = [my_cards[1-discard]]
        for i in range(num_simulations):
            if (drawn_cards := random.sample(non_shown_cards, 8 - len(community_cards) - len(opp_drawn_card))):
                my_new_cards =  kept_cards + [drawn_cards[-1]]
                my_hand_rank, opp_hand_rank = evaluate_hand((my_new_cards, opp_drawn_card + drawn_cards[: 2 - len(opp_drawn_card)], community_cards + drawn_cards[2 - len(opp_drawn_card):-1]))
                if (my_hand_rank < opp_hand_rank):
                    win_count += 1
        win_rates[discard] = win_count/num_simulations
    return win_rates 
    
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

        self.fold_thresh = [0.44, 0.38, 0.34, 0.28]
        self.raise_thres = [0.51, 0.52, 0.55, 0.64] # 65 percentile
        self.med_equity = [0.48, 0.45, 0.44, 0.47] # 50 percentile
        self.high_equity = [0.53, 0.62, 0.68, 0.78] # 80 percentile
        self.discard_thres = [0.44, 0.38]

    def act(self, observation:Observation, reward, terminated, truncated, info):
        # Example of using the logger
        # print(observation)
        if observation["street"] == 0 and info["hand_number"] % 50 == 0:
            self.logger.info(f"Hand number: {info['hand_number']}")

        # if (self.winnings/3 > (1000-info["hand_number"]+1) / 2 + 1):
        #     return action_types.FOLD.value, 0, -1

        street = observation['street']
        pot = observation['my_bet'] + observation['opp_bet']
        diff = observation['opp_bet'] - observation['my_bet']

        raise_amount = 0

        pot_odd = diff/(pot)
        equity = fast_monte_carlo(observation,num_simulations=1000)

        if observation["valid_actions"][action_types.DISCARD.value] and \
            street == 1 and equity < self.discard_thres[street]:
            # print("case discard")
            discard_equity = discard_monte_carlo(observation, num_simulations=500)
            if max(discard_equity) > equity:
                return action_types.DISCARD.value, 0, int(discard_equity[1] > discard_equity[0])

        if observation["valid_actions"][action_types.FOLD.value] \
            and (equity < pot_odd*self.high_equity[street]):
            # print("case fold")
            return action_types.FOLD.value, 0, -1
        
        if observation["valid_actions"][action_types.RAISE.value] and \
            equity > self.raise_thres[street]:
            # print("case raise")
            raise_amount = max(int(equity*pot/(1.01-equity)), observation['min_raise'])
            raise_amount = min(raise_amount, observation['max_raise'])
            return action_types.RAISE.value, raise_amount, -1
        
        if observation["valid_actions"][action_types.CALL.value]:
            # print("case call")
            action_type = action_types.CALL.value
        else:
            action_type = action_types.CHECK.value

        return action_type, raise_amount, -1
    
    def observe(self, observation, reward, terminated, truncated, info):
        if terminated:
            self.winnings += reward
            if abs(reward) > 20: 
                print(f"Significant hand with reward {reward}")