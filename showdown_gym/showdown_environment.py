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
        """Shaped reward: damage dealt/taken + KO deltas + terminal win/loss."""
        prior = self._get_prior_battle(battle)
        if prior is None:
            return 0.0

        def hp_vec(side_dict) -> list[float]:
            v = [float(m.current_hp_fraction) for m in side_dict.values()]
            # pad to 6 so newly revealed mons don't look like "healing"
            while len(v) < 6:
                v.append(1.0)
            return v

        # current and prior HP vectors
        team_now = hp_vec(battle.team)
        team_prev = hp_vec(prior.team)
        opp_now = hp_vec(battle.opponent_team)
        opp_prev = hp_vec(prior.opponent_team)

        dmg_dealt = float(np.sum(np.array(opp_prev) - np.array(opp_now)))
        dmg_taken = float(np.sum(np.array(team_prev) - np.array(team_now)))

        def ko_count(side_dict) -> int:
            return sum(1 for m in side_dict.values() if m.fainted)

        new_kos_we_got = float(ko_count(battle.opponent_team) - ko_count(prior.opponent_team))
        new_kos_against_us = float(ko_count(battle.team) - ko_count(prior.team))

        # weights
        w_dealt = 1.0
        w_taken = 0.5
        w_ko = 2.0
        win_bonus = 20.0
        loss_bonus = -20.0

        reward = 0.0
        reward += w_dealt * dmg_dealt
        reward -= w_taken * dmg_taken
        reward += w_ko * new_kos_we_got
        reward -= w_ko * new_kos_against_us

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
        return 18

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """18-D state:[eff_4, rel_4, pstab_4, force_switch, has_switch, my_hp, opp_hp, my_alive/6, opp_alive/6]"""
        max_moves = 4
        eff = [1.0] * max_moves       # neutral if unknown
        rel = [1.0] * max_moves       # accuracy * expected_hits
        pstab = [0.0] * max_moves     # (base_power * STAB) / 200

        active = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        my_types = set(active.types) if active and active.types else set()

        for i, m in enumerate((battle.available_moves or [])[:max_moves]):
            # effectiveness
            if opp is not None:
                try:
                    e = float(opp.damage_multiplier(m))
                except Exception:
                    e = 1.0
                eff[i] = float(min(max(e, 0.0), 4.0))
            else:
                eff[i] = 1.0

            # reliability = accuracy * expected_hits
            acc = getattr(m, "accuracy", 1.0)
            if acc is None:
                acc = 1.0
            acc = float(acc)
            # some moves may store accuracy as 0-100; normalise defensively
            if acc > 1.0:
                acc /= 100.0
            exp_hits = float(getattr(m, "expected_hits", 1.0) or 1.0)
            rel[i] = float(min(max(acc * exp_hits, 0.0), 2.0))

            # power × STAB (normalised)
            bp = float(getattr(m, "base_power", 0.0) or 0.0)
            has_stab = 1.0
            try:
                has_stab = 1.5 if (getattr(m, "type", None) in my_types) else 1.0
            except Exception:
                has_stab = 1.0
            pstab[i] = float(min(max((bp * has_stab) / 200.0, 0.0), 2.0))

        # context
        force_switch = 1.0 if bool(getattr(battle, "force_switch", False)) else 0.0
        has_switch = 1.0 if (len(battle.available_switches or [])) > 0 else 0.0
        my_hp = float(getattr(active, "current_hp_fraction", 1.0) or 1.0)
        opp_hp = float(getattr(opp, "current_hp_fraction", 1.0) or 1.0)
        my_alive = sum(1 for m in battle.team.values() if not m.fainted) / 6.0
        opp_alive = sum(1 for m in battle.opponent_team.values() if not m.fainted) / 6.0

        vec = np.array(
            eff + rel + pstab
            + [force_switch, has_switch, my_hp, opp_hp, my_alive, opp_alive],
            dtype=np.float32,
        )
        return vec



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
