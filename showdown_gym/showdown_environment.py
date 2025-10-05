import os
import time
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
    TYPES = [
        "water", "normal", "grass", "flying", "psychic", "bug", "fire", "poison",
        "dark", "fighting", "ground", "rock", "steel", "dragon", "electric",
        "ghost", "fairy", "ice"
    ]

    @staticmethod
    def _norm_type_name(t) -> str:
        # Works for PokemonType enums and strings
        return (getattr(t, "name", str(t))).lower()

    def _move_meta_block(self, battle) -> list[float]:
        """
        Returns [stab(4), priority>0(4), category(4: phys=+1, spec=-1, status=0)]
        padded to 4 moves.
        """
        stab, prio, cat = [], [], []

        act = battle.active_pokemon
        act_types = [self._norm_type_name(t) for t in (act.types if act and act.types else [])]

        for m in (battle.available_moves or [])[:4]:
            # --- STAB flag ---
            mt = ""
            try:
                mt = self._norm_type_name(getattr(m, "type", ""))
            except Exception:
                mt = ""
            stab.append(1.0 if mt and mt in act_types else 0.0)

            # --- Priority flag (safe) ---
            p = 0
            try:
                p = int(m.priority)  # may KeyError
            except Exception:
                entry = getattr(m, "entry", {}) or {}
                p = int(entry.get("priority", 0))
            prio.append(1.0 if p > 0 else 0.0)

            # --- Category encoding ---
            cat_attr = getattr(m, "category", None)
            if hasattr(cat_attr, "name"):
                cname = cat_attr.name.lower()
            else:
                cname = str(cat_attr).lower()
            if "physical" in cname:
                cat.append(1.0)
            elif "special" in cname:
                cat.append(-1.0)
            else:
                cat.append(0.0)

        # pad to 4 moves
        while len(stab) < 4:
            stab.append(0.0);
            prio.append(0.0);
            cat.append(0.0)

        return stab + prio + cat  # length 12

    TYPE_TO_INDEX = {t: i for i, t in enumerate(TYPES)}

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
        reward =
          + damage dealt to opponent
          - 0.5 * damage taken
          + 1.0 * (new opponent KOs)
          - 1.0 * (our new KOs)
          + 20.0 on win, -20.0 on loss

        Args:
            battle (AbstractBattle): The current battle instance containing information
                about the player's team and the opponent's team from the player's perspective.
            prior_battle (AbstractBattle): The prior battle instance to compare against.
        Returns:
            float: The calculated reward based on the change in state of the battle.
        """

        prior_battle = self._get_prior_battle(battle)

        reward = 0.0

        health_team = [mon.current_hp_fraction for mon in battle.team.values()]
        health_opponent = [
            mon.current_hp_fraction for mon in battle.opponent_team.values()
        ]

        # If the opponent has less than 6 Pokémon, fill the missing values with 1.0 (fraction of health)
        if len(health_opponent) < len(health_team):
            health_opponent.extend([1.0] * (len(health_team) - len(health_opponent)))

        prior_health_opponent = []
        if prior_battle is not None:
            prior_health_opponent = [
                mon.current_hp_fraction for mon in prior_battle.opponent_team.values()
            ]

        # Ensure health_opponent has 6 components, filling missing values with 1.0 (fraction of health)
        if len(prior_health_opponent) < len(health_team):
            prior_health_opponent.extend(
                [1.0] * (len(health_team) - len(prior_health_opponent))
            )

        diff_health_opponent = np.array(prior_health_opponent) - np.array(
            health_opponent
        )

        sum_diff_health_opponent = np.sum(diff_health_opponent)

        # sum up the damage dealt to opponent
        diff_health_team = np.array([mon.current_hp_fraction for mon in prior_battle.team.values()]) - np.array(
            health_team)
        sum_diff_health_team = np.sum(diff_health_team)

        # Caclulate whether any KOs have happened
        num_ko_team = float(sum(1 for m in battle.team.values() if m.fainted))
        prior_num_ko_team = float(sum(1 for m in prior_battle.team.values() if m.fainted))
        diff_ko_team = num_ko_team - prior_num_ko_team

        # caclulate whether any opponent KOs have happened
        num_ko_opponent = float(sum(1 for m in battle.opponent_team.values() if m.fainted))
        prior_num_ko_opponent = float(sum(1 for m in prior_battle.opponent_team.values() if m.fainted))
        diff_ko_opponent = num_ko_opponent - prior_num_ko_opponent

        # Reward Weightings
        w_dealt = 1.0
        w_taken = -0.5
        w_ko_opponent = 1.0
        w_ko_team = -1.0
        w_win = 20.0
        w_loss = -20.0

        # Reward for reducing the opponent's health
        reward += (w_dealt * sum_diff_health_opponent)  # Reward for damage dealt to opponent
        reward += (w_taken * sum_diff_health_team)  # Penalty for damage taken
        reward += (w_ko_opponent * diff_ko_opponent)  # Reward for opponent KOs
        reward += (w_ko_team * diff_ko_team)  # Penalty for our KOs
        if battle.won:
            reward += w_win
        elif battle.lost:
            reward += w_loss

        return reward

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
        return 67

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

        # My current pokemon type - one hot encoded
        current_type = [0.0] * len(self.TYPES)
        if battle.active_pokemon is not None:
            for t in battle.active_pokemon.types:
                tn = self._norm_type_name(t)
                idx = self.TYPE_TO_INDEX.get(tn)
                if idx is not None:
                    current_type[idx] = 1.0

        # Opponent current pokemon type - one hot encoded
        opponent_current_type = [0.0] * len(self.TYPES)
        if battle.opponent_active_pokemon is not None:
            for t in battle.opponent_active_pokemon.types:
                tn = self._norm_type_name(t)
                idx = self.TYPE_TO_INDEX.get(tn)
                if idx is not None:
                    opponent_current_type[idx] = 1.0

        # Whether can tera or not
        can_tera = [1.0 if battle.can_tera else 0.0]
        num_switches = [float(len(battle.available_switches))]
        num_moves = [float(len(battle.available_moves))]

        # --- move base powers (normalize by 200, pad to 4) ---
        move_bps = []
        for m in (battle.available_moves or []):
            bp = m.base_power if m.base_power is not None else 0
            move_bps.append(float(bp) / 200.0)  # crude normalization
            if len(move_bps) == 4:
                break
        if len(move_bps) < 4:
            move_bps += [0.0] * (4 - len(move_bps))

        move_meta = self._move_meta_block(battle)

        #########################################################################################################
        # Caluclate the length of the final_vector and make sure to update the value in _observation_size above #
        #########################################################################################################

        # Final vector - single array with health of both teams
        final_vector = np.concatenate(
            [
                health_team,  # N components for the health of each pokemon - 6
                health_opponent,  # N components for the health of opponent pokemon - 6
                current_type,  # 18 components for my current pokemon type - 18
                opponent_current_type,  # 18 components for opponent current pokemon type - 18
                can_tera,  # 1 component for whether can tera or not - 1
                num_switches,  # 1 component for number of switches available - 1
                num_moves,  # 1 component for number of moves available - 1
                move_bps,  # 4 components for the base power of each move - 4
                move_meta  # 12 components for move meta info - 12
            ]
        ).astype(np.float32)

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
