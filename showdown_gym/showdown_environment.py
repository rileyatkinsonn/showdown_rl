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

from typing import List, Optional, Tuple
import math

from poke_env.battle import AbstractBattle
from poke_env.battle import Move
from poke_env.battle import MoveCategory
from poke_env.battle import Pokemon
from poke_env.battle import SideCondition
from poke_env.battle import Status


from showdown_gym.base_environment import BaseShowdownEnv

# ----------------------------
# Small utilities (pure funcs)
# ----------------------------
def _hp_frac(mon: Optional[Pokemon]) -> float:
    if mon is None:
        return 1.0
    return float(mon.current_hp_fraction if mon.current_hp_fraction is not None else 0.0)

def _is_harmful_status(s: Optional[Status]) -> bool:
    return s in {Status.PSN, Status.TOX, Status.BRN, Status.PAR, Status.SLP, Status.FRZ}

def _side_has_hazards(side_conditions: dict) -> bool:
    if not side_conditions:
        return False
    return any(
        sc in side_conditions
        for sc in (
            SideCondition.SPIKES,
            SideCondition.STEALTH_ROCK,
            SideCondition.STICKY_WEB,
            SideCondition.TOXIC_SPIKES,
        )
    )

def _priority_available(moves: List[Move]) -> bool:
    return any((m.priority or 0) > 0 for m in moves)

def _speed_advantage(us: Optional[Pokemon], them: Optional[Pokemon]) -> bool:
    if us is None or them is None:
        return False
    try:
        # Effective speed (includes boosts).
        return (us.stats.get("spe", 0) or 0) > (them.stats.get("spe", 0) or 0)
    except Exception:
        # Fallback to base stats if effective not available yet
        return (us.base_stats.get("spe", 0) or 0) > (them.base_stats.get("spe", 0) or 0)

def _stab_multiplier(move: Move, mon: Optional[Pokemon]) -> float:
    if mon is None or move.type is None:
        return 1.0
    return 1.5 if move.type in (mon.types or []) else 1.0

def _reliability(move: Move) -> float:
    # accuracy can be None for "always hits" in sim — treat as 1.0
    acc = float(move.accuracy) if move.accuracy is not None else 1.0
    hits = float(move.expected_hits or 1.0)
    return max(0.0, min(1.0, acc)) * max(1.0, hits)

def _effectiveness(move: Move, target: Optional[Pokemon]) -> float:
    if target is None or move.type is None:
        return 1.0
    try:
        return float(target.damage_multiplier(move))
    except Exception:
        # If the engine can't compute yet, be neutral
        return 1.0

def _stab_power(move: Move, mon: Optional[Pokemon]) -> float:
    base = float(move.base_power or 0.0)
    return _stab_multiplier(move, mon) * base

def _expected_damage_frac_const_scaled(move: Move, us: Optional[Pokemon], them: Optional[Pokemon], k: float = 0.0045) -> float:
    """
    Tiny, constant-scaled proxy for expected damage in HP fraction.
    Scales STAB*BP by effectiveness and reliability; k is a global scale.
    """
    return k * _stab_power(move, us) * _effectiveness(move, them) * _reliability(move)

def _can_kill_now(moves: List[Move], us: Optional[Pokemon], them: Optional[Pokemon], them_hp_frac: float) -> bool:
    return any(_expected_damage_frac_const_scaled(m, us, them) >= them_hp_frac - 1e-6 for m in moves)

def _opp_can_kill_now_heuristic(us: Optional[Pokemon], them: Optional[Pokemon]) -> bool:
    """
    Conservative heuristic that avoids needing opponent revealed moves:
    - If we are < 0.6 HP and opponent has SE STAB potential (types super-effective on us), assume danger.
    - Or if opponent has positive offensive boosts, assume danger when we're < 0.5 HP.
    """
    if us is None or them is None:
        return False
    us_hp = _hp_frac(us)
    # Offensive boosts?
    off_boost = (them.boosts.get("atk", 0) or 0) > 0 or (them.boosts.get("spa", 0) or 0) > 0
    # Does any opp type hit us super effectively?
    se_stab_potential = False
    if us.types and them.types:
        # rough check: if for any opp STAB type the multiplier on our active is >= 2
        for t in (them.types or []):
            class _FakeTypeMove:
                def __init__(self, typ): self.type = typ
            try:
                if max(us.damage_multiplier(_FakeTypeMove(t)), 1.0) >= 2.0:
                    se_stab_potential = True
                    break
            except Exception:
                pass
    if us_hp < 0.5 and off_boost:
        return True
    if us_hp < 0.6 and se_stab_potential:
        return True
    return False

def _safer_2hko_exists(moves: List[Move], us: Optional[Pokemon], them: Optional[Pokemon], them_hp_frac: float) -> bool:
    """
    Returns True if there exists a high-reliability move (>=0.95) that deals at least half
    the opponent's remaining HP in expectation (suggesting a safe 2HKO).
    """
    for m in moves:
        rel = _reliability(m)
        if rel >= 0.95:
            dmg = _expected_damage_frac_const_scaled(m, us, them)
            if 0.5 * them_hp_frac <= dmg < them_hp_frac:
                return True
    return False

def _last_turn_opp_switched(prior: Optional[AbstractBattle], now: AbstractBattle) -> bool:
    if prior is None:
        return False
    try:
        prev = prior.opponent_active_pokemon
        cur = now.opponent_active_pokemon
        if prev is None or cur is None:
            return False
        # If species or identifier changed, treat as switch
        return (prev.species != cur.species) or (prev._id != cur._id)
    except Exception:
        return False


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
        Per-step reward with tiny, orthogonal nudges:
          +12  KO they faint
          -12  KO we faint
          +1.5 speed advantage on our acting turn
          +0.004 * (stab_power * eff * reliability) for chosen move quality
          -1   overkill guard when a safer 2HKO exists
          +3   applying useful status (burn/para/tox) when not in immediate kill window
          -2   switch tax, unless forced or dodging lethal
          +0.5 opponent takes residual damage end of turn
          +3/-3 tera discipline (only up-when enabling a kill; down-when wasted)
          ±won_value terminal
        """
        r = 0.0

        # ---- First-step guard when using diffs vs prior ----
        # You can pass prior_battle from your env; if not available, treat as first step.

        us = battle.active_pokemon
        them = battle.opponent_active_pokemon

        our_hp = _hp_frac(us)
        opp_hp = _hp_frac(them)

        # --- KOs this step (diff in fainted counts if prior provided) ---
        if not first_step and prior_battle is not None:
            def _fainted_count(side: dict) -> int:
                return sum(1 for p in side.values() if p.fainted)
            our_fainted_now = _fainted_count(battle.team or {})
            our_fainted_prev = _fainted_count(prior_battle.team or {})
            opp_fainted_now = _fainted_count(battle.opponent_team or {})
            opp_fainted_prev = _fainted_count(prior_battle.opponent_team or {})

            if opp_fainted_now > opp_fainted_prev:
                r += 12.0
            if our_fainted_now > our_fainted_prev:
                r -= 12.0

        # --- Tempo (speed edge) ---
        if _speed_advantage(us, them):
            r += 1.5

        # --- Chosen move quality + overkill guard ---
        if battle.available_moves:
            # Heuristic: pick the actual move we sent if available, else proxy with argmax.
            # (poke-env doesn't expose already-chosen move here reliably; this proxy is fine.)
            chosen = max(battle.available_moves, key=lambda m: _stab_power(m, us) * _effectiveness(m, them) * _reliability(m))
            quality = 0.004 * _stab_power(chosen, us) * _effectiveness(chosen, them) * _reliability(chosen)
            r += float(quality)

            exp_dmg = _expected_damage_frac_const_scaled(chosen, us, them)
            if exp_dmg > 1.5 * opp_hp and _safer_2hko_exists(battle.available_moves, us, them, opp_hp):
                r -= 1.0

        # --- Useful status only when not in immediate kill window ---
        if them is not None:
            inflicted = getattr(them, "status", None)
            if _is_harmful_status(inflicted) and not _can_kill_now(list(battle.available_moves or []), us, them, opp_hp):
                r += 3.0

        # --- Switch tax (contextual) ---
        if getattr(battle, "switched_this_turn", False) or (battle.force_switch and not battle.available_moves):
            # If we know we switched (your env can set a flag), punish unless forced or dodging lethal
            forced = bool(battle.force_switch)
            dodging_lethal = _opp_can_kill_now_heuristic(us, them)
            if not forced and not dodging_lethal:
                r -= 2.0

        # --- Residual realized (poison/burn/sand/screen drop etc.) ---
        # Can't detect all residuals reliably; as a lightweight proxy, reward if opp HP fraction strictly decreased while we didn't act (first step guard protects).
        if not first_step and prior_battle is not None:
            prev_opp = _hp_frac(getattr(prior_battle, "opponent_active_pokemon", None))
            if opp_hp < prev_opp - 1e-6:
                r += 0.5

        # --- Tera discipline ---
        used_tera = bool(getattr(battle, "was_tera_this_turn", False))
        if used_tera:
            if _can_kill_now(list(battle.available_moves or []), us, them, opp_hp) and _effectiveness(chosen, them) > 1.0:
                r += 3.0
            elif _effectiveness(chosen, them) <= 1.0 and not _can_kill_now(list(battle.available_moves or []), us, them, opp_hp):
                r -= 3.0

        # --- Terminal outcome ---
        if battle.finished:
            if battle.won:
                r += float(won_value)
            else:
                r -= float(won_value)

        return float(r)

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
        return 30

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        Returns a compact float32 vector:
          - Per-move block (up to 4 moves): [eff, reliability, stab_power] * 4  => 12 dims
          - Minimal existing 7-style context: [our_hp, opp_hp, alive_frac_us, alive_frac_them, force_switch, has_switch] => 6 dims
          - New compact extras (12 dims):
             [speed_advantage, priority_available, can_kill_now, opp_can_kill_now,
              hazards_us, hazards_them, status_us, status_them,
              opp_boosted_offense, opp_boosted_speed, tera_available, last_turn_opp_switched]
        Total: 12 + 6 + 12 = 30 dims
        """
        us = battle.active_pokemon
        them = battle.opponent_active_pokemon
        our_moves: List[Move] = list(battle.available_moves) if battle.available_moves else []

        # Per-move features (pad to 4)
        per_move_feats: List[float] = []
        for m in our_moves[:4]:
            per_move_feats += [
                _effectiveness(m, them),
                _reliability(m),
                _stab_power(m, us),
            ]
        # pad missing moves with zeros (3 features per slot)
        while len(per_move_feats) < 12:
            per_move_feats.append(0.0)

        # Tiny original context (keep consistent scaling 0..1 where possible)
        our_hp = _hp_frac(us)
        opp_hp = _hp_frac(them)

        def _alive_frac(side: dict) -> float:
            if not side:
                return 1.0
            total = len(side)
            alive = sum(1 for p in side.values() if not p.fainted)
            return float(alive / max(1, total))

        alive_us = _alive_frac(battle.team or {})
        alive_them = _alive_frac(battle.opponent_team or {})
        force_switch = 1.0 if battle.force_switch else 0.0
        has_switch = 1.0 if battle.available_switches else 0.0

        base_context = [our_hp, opp_hp, alive_us, alive_them, force_switch, has_switch]

        # New compact extras (booleans as 0/1 floats)
        speed_edge = 1.0 if _speed_advantage(us, them) else 0.0
        priority_avail = 1.0 if _priority_available(our_moves) else 0.0
        can_kill = 1.0 if _can_kill_now(our_moves, us, them, opp_hp) else 0.0
        opp_can_kill = 1.0 if _opp_can_kill_now_heuristic(us, them) else 0.0

        hazards_us = 1.0 if _side_has_hazards(battle.side_conditions or {}) else 0.0
        hazards_them = 1.0 if _side_has_hazards(battle.opponent_side_conditions or {}) else 0.0
        status_us = 1.0 if _is_harmful_status(getattr(us, "status", None)) else 0.0
        status_them = 1.0 if _is_harmful_status(getattr(them, "status", None)) else 0.0

        opp_boost_off = 1.0 if (them and ((them.boosts.get("atk", 0) or 0) > 0 or (them.boosts.get("spa", 0) or 0) > 0)) else 0.0
        opp_boost_spe = 1.0 if (them and (them.boosts.get("spe", 0) or 0) > 0) else 0.0
        tera_avail = 1.0 if getattr(battle, "can_tera", False) else 0.0
        last_opp_sw = 1.0 if _last_turn_opp_switched(prior_battle, battle) else 0.0

        extras = [
            speed_edge, priority_avail, can_kill, opp_can_kill,
            hazards_us, hazards_them, status_us, status_them,
            opp_boost_off, opp_boost_spe, tera_avail, last_opp_sw,
        ]

        vec = np.asarray(per_move_feats + base_context + extras, dtype=np.float32)
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
