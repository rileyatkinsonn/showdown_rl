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
        """
        Sparse & simple imitation reward:
          + If switched when heuristic recommends switch: +1.0; otherwise -1.0
          + If stayed and used a move:
               +1.0 if chosen move has max effectiveness vs opponent at decision time,
               else - (best_eff - chosen_eff)  (clipped to [-1, 0])
          + Win bonus +10, loss bonus -10
        """
        prior_battle = self._get_prior_battle(battle)
        if prior_battle is None:
            return 0.0

        reward = 0.0
        # --- teacher switch decision evaluated at prior state
        prior_switch_flag = self._switch_recommended_flag(prior_battle)

        # --- did we switch?
        switched = self._did_switch(prior_battle, battle)

        if switched:
            # correct if teacher said switch
            reward += 1.0 if prior_switch_flag >= 0.5 else -1.0
        else:
            # we stayed; see if we used the max-effectiveness move
            opp_prev = getattr(prior_battle, "opponent_active_pokemon", None)
            # build effectiveness for moves available at prior step
            prior_moves = (getattr(prior_battle, "available_moves", None) or [])[:4]
            if opp_prev is not None and prior_moves:
                effs_prev = [self._safe_eff(opp_prev, m) for m in prior_moves]
                best_eff = max(effs_prev) if effs_prev else 1.0

                chosen_move = self._moved_pp_drop(prior_battle, battle)
                if chosen_move is not None:
                    chosen_eff = self._safe_eff(opp_prev, chosen_move)
                    if abs(chosen_eff - best_eff) < 1e-9:
                        reward += 1.0
                    else:
                        # scale penalty to how suboptimal it was, capped at -1.0
                        gap = max(0.0, best_eff - chosen_eff)
                        reward -= min(1.0, gap)
                # if we can't detect the chosen move, leave imitation at 0 for this step

        # terminal shaping (small, simple)
        if battle.won:
            reward += 10.0
        elif battle.lost:
            reward -= 10.0

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
        return 5

    # ---------- simple helpers (minimal + safe) ----------
    def _safe_eff(self, opp, move) -> float:
        """Return effectiveness multiplier vs opponent for 'move'.
        Falls back to 1.0 if anything is missing."""
        try:
            if opp is None or move is None:
                return 1.0
            # poke-env supports opponent.damage_multiplier(move)
            return float(opp.damage_multiplier(move))
        except Exception:
            # fallback via move.type if needed
            try:
                mtype = getattr(move, "type", None)
                return float(opp.damage_multiplier(mtype)) if mtype is not None else 1.0
            except Exception:
                return 1.0

    def _estimate_matchup_simple(self, mon, opp) -> float:
        """Very light matchup score: 'how good my types are vs theirs' minus 'how good theirs are vs mine'.
        Ignores speed/HP to keep state+reward minimal and deterministic."""
        if mon is None or opp is None:
            return 0.0
        try:
            # best we do to them
            good = max([opp.damage_multiplier(t) for t in mon.types if t is not None] or [1.0])
            # best they do to us
            bad = max([mon.damage_multiplier(t) for t in opp.types if t is not None] or [1.0])
            return float(good - bad)
        except Exception:
            return 0.0

    def _switch_recommended_flag(self, battle) -> float:
        """Return 1.0 if switching is clearly better than staying (by a small margin)."""
        act = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        if act is None or opp is None:
            return 0.0
        current = self._estimate_matchup_simple(act, opp)
        best_switch = current
        try:
            for s in (battle.available_switches or []):
                best_switch = max(best_switch, self._estimate_matchup_simple(s, opp))
        except Exception:
            pass
        # margin keeps it simple but decisive
        return 1.0 if (best_switch - current) > 0.5 else 0.0

    def _move_effectiveness_vector(self, battle) -> list[float]:
        """Up to 4 moves vs current opponent; 0.0 if slot empty."""
        effs = [0.0, 0.0, 0.0, 0.0]
        opp = getattr(battle, "opponent_active_pokemon", None)
        moves = (getattr(battle, "available_moves", None) or [])[:4]
        for i, m in enumerate(moves):
            effs[i] = self._safe_eff(opp, m)
        return effs

    def _did_switch(self, prior_battle, battle) -> bool:
        """Detect if we switched between prior and current."""
        try:
            a0 = getattr(prior_battle, "active_pokemon", None)
            a1 = getattr(battle, "active_pokemon", None)
            if a0 is None or a1 is None:
                return False
            # compare species (safer than object identity)
            return str(a0.species) != str(a1.species)
        except Exception:
            return False

    def _moved_pp_drop(self, prior_battle, battle):
        """Return the move (object) we used by finding the move whose PP dropped; else None.
        Only valid when we did NOT switch."""
        try:
            a0 = getattr(prior_battle, "active_pokemon", None)
            a1 = getattr(battle, "active_pokemon", None)
            if a0 is None or a1 is None:
                return None
            # ensure we didn't switch
            if str(a0.species) != str(a1.species):
                return None

            prev_moves = (getattr(a0, "moves", {}) or {})
            curr_moves = (getattr(a1, "moves", {}) or {})

            # keys are move ids; use PP drop to detect selection
            for mid, pm in prev_moves.items():
                cm = curr_moves.get(mid)
                if cm is None:
                    continue
                pp0 = getattr(pm, "current_pp", None)
                pp1 = getattr(cm, "current_pp", None)
                if pp0 is not None and pp1 is not None and pp1 < pp0:
                    return cm  # or pm; id/type same
        except Exception:
            pass
        return None

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
        effs = self._move_effectiveness_vector(battle)  # len 4
        switch_flag = [self._switch_recommended_flag(battle)]  # len 1


        #########################################################################################################
        # Caluclate the length of the final_vector and make sure to update the value in _observation_size above #
        #########################################################################################################

        # Final vector - single array with health of both teams
        final_vector = np.concatenate(
            [
                effs,
                switch_flag,
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
