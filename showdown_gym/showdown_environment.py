import os
import time
from enum import Enum
from typing import Any, Dict

import numpy as np
from poke_env import (
    AccountConfiguration,
    MaxBasePowerPlayer,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.battle import AbstractBattle
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.player.player import Player

from showdown_gym.base_environment import BaseShowdownEnv


class ShowdownEnvironment(BaseShowdownEnv):

    def __init__(
        self,
        battle_format: str = "gen9randombattle",
        account_name_one: str = "train_one",
        account_name_two: str = "train_two",
        team: str | None = None,
    ):
        super().__init__(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )

    def get_additional_info(self) -> Dict[str, Dict[str, Any]]:
        info = super().get_additional_info()

        # Add any additional information you want to include in the info dictionary that is saved in logs
        # For example, you can add the win status

        if self.battle1 is not None:
            agent = self.possible_agents[0]
            info[agent]["win"] = self.battle1.won

        return info

    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        Calculates the reward based on the changes in state of the battle.

        You need to implement this method to define how the reward is calculated

        Args:
            battle (AbstractBattle): The current battle instance containing information
                about the player's team and the opponent's team from the player's perspective.
            prior_battle (AbstractBattle): The prior battle instance to compare against.
        Returns:
            float: The calculated reward based on the change in state of the battle.
        """

        prior_battle = self._get_prior_battle(battle)
        if prior_battle is None:
            return 0.0  # no shaping on the very first observation

        # --- Collect current HP fractions ---
        hp_team_now = [m.current_hp_fraction for m in battle.team.values()]
        hp_opp_now = [m.current_hp_fraction for m in battle.opponent_team.values()]

        # Pad opponent to match team length (random battles can start with unknowns)
        if len(hp_opp_now) < len(hp_team_now):
            hp_opp_now += [1.0] * (len(hp_team_now) - len(hp_opp_now))

        # --- Prior HP fractions ---
        hp_team_prev = [m.current_hp_fraction for m in prior_battle.team.values()]
        hp_opp_prev = [m.current_hp_fraction for m in prior_battle.opponent_team.values()]
        if len(hp_opp_prev) < len(hp_team_prev):
            hp_opp_prev += [1.0] * (len(hp_team_prev) - len(hp_opp_prev))

        # --- Damage deltas (positive means we dealt damage / we took damage) ---
        dmg_dealt = float(np.sum(np.array(hp_opp_prev) - np.array(hp_opp_now)))
        dmg_taken = float(np.sum(np.array(hp_team_prev) - np.array(hp_team_now)))

        # --- KO deltas (positive means a new KO happened since last step) ---
        ko_team_now = sum(1 for m in battle.team.values() if m.fainted)
        ko_team_prev = sum(1 for m in prior_battle.team.values() if m.fainted)
        new_kos_against_us = float(ko_team_now - ko_team_prev)

        ko_opp_now = sum(1 for m in battle.opponent_team.values() if m.fainted)
        ko_opp_prev = sum(1 for m in prior_battle.opponent_team.values() if m.fainted)
        new_kos_we_got = float(ko_opp_now - ko_opp_prev)

        # --- Weights ---
        w_hp = 1.0
        w_ko = 2.0
        win_bonus = 20.0
        loss_bonus = -20.0

        # --- Reward ---
        reward = 0.0
        reward += w_hp * dmg_dealt  # good
        reward -= w_hp * dmg_taken  # bad
        reward += w_ko * new_kos_we_got  # good
        reward -= w_ko * new_kos_against_us  # bad

        if battle.won:
            reward += win_bonus
        elif battle.lost:
            reward += loss_bonus

        return float(reward)

    def _observation_size(self) -> int:
        """
        Returns the size of the observation size to create the observation space for all possible agents in the environment.

        You need to set obvervation size to the number of features you want to include in the observation.
        Annoyingly, you need to set this manually based on the features you want to include in the observation from emded_battle.

        Returns:
            int: The size of the observation space.
        """

        # Simply change this number to the number of features you want to include in the observation from embed_battle.
        # If you find a way to automate this, please let me know!
        return 41

    class PokemonType(Enum):
        NORMAL = 1.0
        FIRE = 2.0
        WATER = 3.0
        ELECTRIC = 4.0
        GRASS = 5.0
        ICE = 6.0
        FIGHTING = 7.0
        POISON = 8.0
        GROUND = 9.0
        FLYING = 10.0
        PSYCHIC = 11.0
        BUG = 12.0
        ROCK = 13.0
        GHOST = 14.0
        DRAGON = 15.0
        DARK = 16.0
        STEEL = 17.0
        FAIRY = 18.0

    def _encode_type(self, poke_type) -> float:
        if poke_type is None or poke_type.name == "???":
            return 0.0
        try:
            return self.PokemonType[poke_type.name.upper()].value
        except KeyError:
            return 0.0

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        Embeds the current state of a Pokémon battle into a numerical vector representation.
        This method generates a feature vector that represents the current state of the battle,
        this is used by the agent to make decisions.

        You need to implement this method to define how the battle state is represented.

        Args:
            battle (AbstractBattle): The current battle instance containing information about
                the player's team and the opponent's team.
        Returns:
            np.float32: A 1D numpy array containing the state you want the agent to observe.
        """

        health_team = [mon.current_hp_fraction for mon in battle.team.values()]
        health_opponent = [
            mon.current_hp_fraction for mon in battle.opponent_team.values()
        ]

        # Ensure health_opponent has 6 components, filling missing values with 1.0 (fraction of health)
        if len(health_opponent) < len(health_team):
            health_opponent.extend([1.0] * (len(health_team) - len(health_opponent)))

        # Fainted flags for my team (6 slots)
        fainted_team = [1.0 if mon.fainted else 0.0 for mon in battle.team.values()]

        # Fainted flags for opponent team (6 slots)
        fainted_opponent = [1.0 if mon.fainted else 0.0 for mon in battle.opponent_team.values()]

        # Pad so both are always length 6
        while len(fainted_team) < 6:
            fainted_team.append(0.0)  # empty slot = not fainted
        while len(fainted_opponent) < 6:
            fainted_opponent.append(0.0)

        # Encode the active pokemon type (2 types, 0 if no type or ???)
        active_poke_types = [
            self._encode_type(battle.active_pokemon.type_1),
            self._encode_type(battle.active_pokemon.type_2),
        ]

        opponent_active_poke_types = [
            self._encode_type(battle.opponent_active_pokemon.type_1),
            self._encode_type(battle.opponent_active_pokemon.type_2),
        ]

        # encode the move types of the active pokemon (4 moves, 0 if no move), negative if unusable
        max_moves = 4
        move_type_ids = [0.0] * max_moves  # 0 = no move
        if battle.active_pokemon is not None:
            for i, move in enumerate(list(battle.active_pokemon.moves.values())[:max_moves]):
                if move.type is not None:
                    type_id = self.PokemonType[move.type.name.upper()].value
                    if move.current_pp > 0:
                        move_type_ids[i] = type_id  # usable
                    else:
                        move_type_ids[i] = -type_id  # revealed but out of PP

        # encode the move types of opponent active pokemon (4 moves, 0 if no move)
        opponent_move_type_ids = [0.0] * max_moves  # 0 = no move
        if battle.opponent_active_pokemon is not None:
            for i, move in enumerate(list(battle.opponent_active_pokemon.moves.values())[:max_moves]):
                if move.type is not None:
                    type_id = self.PokemonType[move.type.name.upper()].value
                    # you usually won’t know opponent PP, but poke-env tracks if revealed
                    if move.current_pp > 0:
                        opponent_move_type_ids[i] = type_id  # revealed and still usable
                    else:
                        opponent_move_type_ids[i] = -type_id  # revealed but out of PP

        # Move effectiveness multipliers against opponent active pokemon (4 moves, 1.0 if no move)
        move_effectiveness = [1.0] * max_moves  # 1.0
        for i, move in enumerate(battle.available_moves[:max_moves]):
            if move.type is not None and battle.opponent_active_pokemon is not None:
                effectiveness = battle.opponent_active_pokemon.damage_multiplier(move.type)
                move_effectiveness[i] = effectiveness

        can_tera = [1.0 if battle.can_tera else 0.0]


        #########################################################################################################
        # Caluclate the length of the final_vector and make sure to update the value in _observation_size above #
        #########################################################################################################

        # Final vector - single array with health of both teams
        final_vector = np.concatenate(
            [
                health_team,  # 6 components for the health of each pokemon
                health_opponent,  # 6 components for the health of opponent pokemon
                fainted_team,  # 6 components for the fainted flags of each pokemon
                fainted_opponent,  # 6 components for the fainted flags of opponent pokemon
                active_poke_types,  # 2 components for the active pokemon types
                opponent_active_poke_types,  # 2 components for the opponent active pokemon types
                move_type_ids,  # 4 components for the move types of the active pokemon
                opponent_move_type_ids,  # 4 components for the move types of the opponent active pokemon
                move_effectiveness,  # 4 components for the move effectiveness against opponent active pokemon
                can_tera,  # 1 component for whether the active pokemon can tera
            ]
        )

        return final_vector


########################################
# DO NOT EDIT THE CODE BELOW THIS LINE #
########################################


class SingleShowdownWrapper(SingleAgentWrapper):
    """
    A wrapper class for the PokeEnvironment that simplifies the setup of single-agent
    reinforcement learning tasks in a Pokémon battle environment.

    This class initializes the environment with a specified battle format, opponent type,
    and evaluation mode. It also handles the creation of opponent players and account names
    for the environment.

    Do NOT edit this class!

    Attributes:
        battle_format (str): The format of the Pokémon battle (e.g., "gen9randombattle").
        opponent_type (str): The type of opponent player to use ("simple", "max", "random").
        evaluation (bool): Whether the environment is in evaluation mode.
    Raises:
        ValueError: If an unknown opponent type is provided.
    """

    def __init__(
        self,
        team_type: str = "random",
        opponent_type: str = "random",
        evaluation: bool = False,
    ):
        opponent: Player
        unique_id = time.strftime("%H%M%S")

        opponent_account = "ot" if not evaluation else "oe"
        opponent_account = f"{opponent_account}_{unique_id}"

        opponent_configuration = AccountConfiguration(opponent_account, None)
        if opponent_type == "simple":
            opponent = SimpleHeuristicsPlayer(
                account_configuration=opponent_configuration
            )
        elif opponent_type == "max":
            opponent = MaxBasePowerPlayer(account_configuration=opponent_configuration)
        elif opponent_type == "random":
            opponent = RandomPlayer(account_configuration=opponent_configuration)
        else:
            raise ValueError(f"Unknown opponent type: {opponent_type}")

        account_name_one: str = "t1" if not evaluation else "e1"
        account_name_two: str = "t2" if not evaluation else "e2"

        account_name_one = f"{account_name_one}_{unique_id}"
        account_name_two = f"{account_name_two}_{unique_id}"

        team = self._load_team(team_type)

        battle_format = "gen9randombattle" if team is None else "gen9ubers"

        primary_env = ShowdownEnvironment(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )

        super().__init__(env=primary_env, opponent=opponent)

    def _load_team(self, team_type: str) -> str | None:
        bot_teams_folders = os.path.join(os.path.dirname(__file__), "teams")

        bot_teams = {}

        for team_file in os.listdir(bot_teams_folders):
            if team_file.endswith(".txt"):
                with open(
                    os.path.join(bot_teams_folders, team_file), "r", encoding="utf-8"
                ) as file:
                    bot_teams[team_file[:-4]] = file.read()

        if team_type in bot_teams:
            return bot_teams[team_type]

        return None
