import numpy as np
from agents.agent import Agent, Observation
from gym_env import PokerEnv
from match import HANDS_PER_MATCH
import random
from treys import Evaluator

action_types = PokerEnv.ActionType
int_to_card = PokerEnv.int_to_card

class PlayerAgent(Agent):
    def __name__(self):
        return "data_collector"

    def __init__(self, stream: bool = False):
        super().__init__(stream)
        self.evaluator = Evaluator()
        self.equities = [[] for _ in range(4)]
        self.win_true = []

    def act(self, observation:Observation, reward, terminated, truncated, info):

        # print(observation)
        #print(info)

        my_cards = [int(card) for card in observation["my_cards"]]
        community_cards = [card for card in observation["community_cards"] if card != -1]
        opp_discarded_card = [observation["opp_discarded_card"]] if observation["opp_discarded_card"] != -1 else []
        opp_drawn_card = [observation["opp_drawn_card"]] if observation["opp_drawn_card"] != -1 else []

        # Calculate equity through Monte Carlo simulation
        shown_cards = my_cards + community_cards + opp_discarded_card + opp_drawn_card
        # print(shown_cards)
        non_shown_cards = [i for i in range(27) if i not in shown_cards]

        def evaluate_hand(cards):
            my_cards, opp_cards, community_cards = cards
            my_cards = list(map(int_to_card, my_cards))
            opp_cards = list(map(int_to_card, opp_cards))
            community_cards = list(map(int_to_card, community_cards))
            my_hand_rank = self.evaluator.evaluate(my_cards, community_cards)
            opp_hand_rank = self.evaluator.evaluate(opp_cards, community_cards)
            return my_hand_rank < opp_hand_rank

        # Run Monte Carlo simulation
        num_simulations = 1000
        
        wins = sum(
            [
            evaluate_hand((my_cards, opp_drawn_card + drawn_cards[: 2 - len(opp_drawn_card)], community_cards + drawn_cards[2 - len(opp_drawn_card) :]))
            for _ in range(num_simulations)
            if (drawn_cards := random.sample(non_shown_cards, 7 - len(community_cards) - len(opp_drawn_card)))
            ]
        )
        equity = wins / num_simulations
        # print(f"equity: {equity}")
        # print(f"street: {observation['street']}")

        self.equities[observation['street']].append(equity) 
        # print(self.equities)

        if observation["valid_actions"][action_types.CALL.value]:
            return action_types.CALL.value, 0, -1

        return action_types.CHECK.value, 0, -1


    def observe(self, observation:Observation, reward, terminated, truncated, info):
        # print(observation)
        # print(info)
        if terminated:
            self.win_true.append(int(reward>0))
            if info['hand_number'] == HANDS_PER_MATCH-2:
                for i in range(4):
                    np.savetxt(f"data/street_{i}_equity.txt", np.array(self.equities[i]), delimiter=",") 
                
                np.savetxt(f"data/win_true.txt", np.array(self.win_true), delimiter=",") 
            
