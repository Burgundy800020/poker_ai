from agents.agent import Agent, Observation
from gym_env import PokerEnv
import random
from treys import Evaluator

action_types = PokerEnv.ActionType
int_to_card = PokerEnv.int_to_card

class PlayerAgent(Agent):
    #parameters
    SMALL_BLUFF = 50 #amount considered small raise, for bluffing
    RAISE_THRES = 0.6 #my equity to raise
    BLUFF_THRES = 0.4 #opponent equity to bluff, < 0.5
    P = BLUFF_THRES/(1-2*BLUFF_THRES)

    def __name__(self):
        return "not_challenge"

    def __init__(self, stream: bool = False):
        super().__init__(stream)
        self.evaluator = Evaluator()
        
        #track opponent
        self.min_opp_equity = 0.0

        self.raised = False 
        self.last_raise_amount = 0
        # print("not challenge inited")

    def act(self, observation:Observation, reward, terminated, truncated, info):
        # print(observation)
        #print(info)

        my_cards = [int(card) for card in observation["my_cards"]]
        community_cards = [card for card in observation["community_cards"] if card != -1]
        opp_discarded_card = [observation["opp_discarded_card"]] if observation["opp_discarded_card"] != -1 else []
        opp_drawn_card = [observation["opp_drawn_card"]] if observation["opp_drawn_card"] != -1 else []

        # Calculate equity through Monte Carlo simulation
        shown_cards = my_cards + community_cards + opp_discarded_card + opp_drawn_card
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
            evaluate_hand((my_cards, opp_drawn_card + drawn_cards[: 2 - len(opp_drawn_card)], community_cards + drawn_cards[2 - len(opp_drawn_card) :]))
            for _ in range(num_simulations)
            if (drawn_cards := random.sample(non_shown_cards, 7 - len(community_cards) - len(opp_drawn_card)))
        )
        equity = wins / num_simulations

        # Calculate pot odds
        continue_cost = observation["opp_bet"] - observation["my_bet"]
        pot_size = observation["my_bet"] + observation["opp_bet"]
        pot_odds = continue_cost / (continue_cost + pot_size) if continue_cost > 0 else 0

        self.logger.debug(f"Equity: {equity:.2f}, Pot odds: {pot_odds:.2f}")

        # Decision making
        raise_amount = 0
        card_to_discard = -1

        #TODO: if we can discard
        #TODO: if we begin betting round
        # if observation["valid_actions"][action_types.DISCARD.value]:
        #     # print("case discard")
        #     action_type = action_types.DISCARD.value
        #     card_to_discard = random.randint(0, 1)
        #     self.logger.debug(f"Discarding card {card_to_discard}: {int_to_card(my_cards[card_to_discard])}")
        #     return action_type, raise_amount, card_to_discard
        if self.raised:
           #opponent just called, update min_opp_equity
           self.min_opp_equity = min(self.last_raise_amount/pot_size, 1.0)
           self.raised = False

        if continue_cost > 0:
            if observation['street'] == 0: 
                if equity > 0.15:
                    return action_types.CALL.value, 0, -1
                return action_types.FOLD.value, 0, -1
            # print("case call")
            self.min_opp_equity = 0.8 
            if equity > 0.9 and observation["valid_actions"][action_types.CALL.value]:
                action_type = action_types.CALL.value
            else:
                action_type = action_types.FOLD.value

        elif observation["valid_actions"][action_types.RAISE.value]:
            # print("case raise")
            opp_equity = (self.min_opp_equity + 0.8)/2*1.1
            odds = equity/opp_equity
            bluff_needed = int(pot_size*opp_equity/(1-2*opp_equity))+2 if opp_equity < 0.49 else 200 
            if odds > 1.15 and observation["street"] > 0:
                action_type = action_types.RAISE.value
                raise_amount = min(int(observation["max_raise"] * odds/2.5), observation['max_raise'])
                raise_amount = max(raise_amount, observation["min_raise"])

            elif bluff_needed <= observation['max_raise']:
                if observation["street"] == 0:
                    if equity > 0.55:
                        raise_amount = max(int(bluff_needed*equity), observation["min_raise"])
                        action_type = action_types.RAISE.value  
                    else: 
                        action_type = action_types.CHECK.value  
                else:
                    raise_amount = max(bluff_needed, observation["min_raise"])
                    action_type = action_types.RAISE.value 
            else:
                action_type = action_types.CHECK.value 
        
        elif observation["valid_actions"][action_types.DISCARD.value]:
            # print("case discard")
            action_type = action_types.DISCARD.value
            card_to_discard = random.randint(0, 1)
            self.logger.debug(f"Discarding card {card_to_discard}: {int_to_card(my_cards[card_to_discard])}")
        else:
            # print("case fold")
            action_type = action_types.FOLD.value
            if observation["opp_bet"] > 20:  # Only log significant folds
                self.logger.info(f"Folding to large bet of {observation['opp_bet']}")

        if action_type == action_types.RAISE.value:
            self.raised = True
            self.last_raise_amount = raise_amount

        return action_type, raise_amount, card_to_discard

    def reset_states(self):
        #track opponent
        self.min_opp_equity = 0.0

        self.raised = False 
        self.last_raise_amount = 0

    def observe(self, observation:Observation, reward, terminated, truncated, info):
        # print(observation)
        # print(info)
        if terminated:  # Only log significant hand results
            if abs(reward) > 20:
                print(f"Significant hand completed with reward: {reward}")
            self.reset_states()
