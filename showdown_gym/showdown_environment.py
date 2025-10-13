import os
import time
from typing import Any, Dict, Optional, List

import numpy as np
from poke_env import (
    AccountConfiguration,
    MaxBasePowerPlayer,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.battle import (
    AbstractBattle,
    Pokemon,
    Move,
)
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.player.player import Player

from showdown_gym.base_environment import BaseShowdownEnv


# ----------------------------
# Heuristic helpers
# ----------------------------
def _stab_for(move: Move, mon: Optional[Pokemon]) -> float:
    if mon is None or move.type is None or not mon.types:
        return 1.0
    return 1.5 if move.type in mon.types else 1.0


def _eff_against(move: Move, target: Optional[Pokemon]) -> float:
    if target is None or move.type is None:
        return 1.0
    try:
        return float(target.damage_multiplier(move))
    except Exception:
        return 1.0


def _rel(move: Move) -> float:
    acc = float(move.accuracy) if move.accuracy is not None else 1.0
    if acc > 1.0:  # guard if 0..100
        acc /= 100.0
    hits = float(move.expected_hits or 1.0)
    return max(0.0, min(1.0, acc)) * max(1.0, hits)


def _move_score(move: Move, us: Optional[Pokemon], them: Optional[Pokemon]) -> float:
    bp = float(move.base_power or 0.0)
    return bp * _stab_for(move, us) * _eff_against(move, them) * _rel(move)


_SPEED_TIER_COEF = 0.1
_HP_COEF = 0.4


def _estimate_matchup(mon: Pokemon, opp: Pokemon) -> float:
    our_vs_them = max([opp.damage_multiplier(t) for t in (mon.types or [])] or [1.0])
    them_vs_our = max([mon.damage_multiplier(t) for t in (opp.types or [])] or [1.0])
    score = float(our_vs_them - them_vs_our)
    try:
        if mon.base_stats["spe"] > opp.base_stats["spe"]:
            score += _SPEED_TIER_COEF
        elif opp.base_stats["spe"] > mon.base_stats["spe"]:
            score -= _SPEED_TIER_COEF
    except Exception:
        pass
    score += float(mon.current_hp_fraction or 0.0) * _HP_COEF
    score -= float(opp.current_hp_fraction or 0.0) * _HP_COEF
    return score


def _best_move_idx(b: AbstractBattle) -> Optional[int]:
    moves: List[Move] = list(b.available_moves or [])
    if not moves:
        return None
    us, them = b.active_pokemon, b.opponent_active_pokemon
    k = min(4, len(moves))
    return max(range(k), key=lambda i: _move_score(moves[i], us, them))


def _best_switch_idx(b: AbstractBattle) -> Optional[int]:
    switches = list(b.available_switches or [])
    if not switches:
        return None
    opp = b.opponent_active_pokemon
    if opp is None:
        return 0
    return max(range(len(switches)), key=lambda i: _estimate_matchup(switches[i], opp))


def _hp_frac(mon: Optional[Pokemon]) -> float:
    if mon is None or mon.current_hp_fraction is None:
        return 1.0
    return float(mon.current_hp_fraction)


def _speed_advantage(us: Optional[Pokemon], them: Optional[Pokemon]) -> bool:
    if us is None or them is None:
        return False
    try:
        return (us.stats.get("spe", 0) or 0) > (them.stats.get("spe", 0) or 0)
    except Exception:
        return (us.base_stats.get("spe", 0) or 0) > (them.base_stats.get("spe", 0) or 0)


def _can_kill_now(b: AbstractBattle, i_best: Optional[int]) -> bool:
    if i_best is None or not b.available_moves:
        return False
    us, them = b.active_pokemon, b.opponent_active_pokemon
    m = list(b.available_moves)[i_best]
    # cheap proxy for expected damage fraction
    k = 0.0045
    exp_frac = k * _move_score(m, us, them)
    return them is not None and exp_frac >= _hp_frac(them) - 1e-6


def _opp_can_kill_now_heuristic(b: AbstractBattle) -> bool:
    us, them = b.active_pokemon, b.opponent_active_pokemon
    if us is None or them is None:
        return False
    us_hp = _hp_frac(us)
    off_boost = (them.boosts.get("atk", 0) or 0) > 0 or (them.boosts.get("spa", 0) or 0) > 0
    se_stab_potential = False
    if us.types and them.types:
        class _T:
            def __init__(self, t): self.type = t
        try:
            for t in them.types:
                if max(us.damage_multiplier(_T(t)), 1.0) >= 2.0:
                    se_stab_potential = True
                    break
        except Exception:
            pass
    if us_hp < 0.5 and off_boost:
        return True
    if us_hp < 0.6 and se_stab_potential:
        return True
    return False


# ----------------------------
# Environment
# ----------------------------
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

        # The RL agent is the first account in this head-to-head wrapper
        self.rl_agent = account_name_one

        # Diagnostics / teacher-imitation counters
        self._act_counts = {"attack": 0, "switch": 0}
        self._teacher_hits = 0
        self._teacher_total = 0
        self.last_action_binary: Optional[int] = None  # 0=attack, 1=switch
        self.debug_actions = False

        # Diagnostics to prove process_action is being invoked
        self._process_action_seen = False
        self._process_action_calls = 0

    # 2 actions: 0 = best move, 1 = best switch
    def _get_action_size(self) -> int | None:
        return 2

    def process_action(self, action: np.int64) -> np.int64:
        """
        Map 2-action policy to battle order ids.
        0 -> best attack (move index -> order 6..9)
        1 -> best switch (switch index -> order 0..5)
        """
        b: AbstractBattle = self.battle1

        # Guard: don’t act if the battle is already finished
        if b is None or getattr(b, "finished", False):
            self.last_action_binary = None
            return np.int64(-2)

        # Mark hook usage
        self._process_action_seen = True
        self._process_action_calls += 1

        a = int(action)

        # Forced switch: ignore requested "attack"
        if b.force_switch:
            self.last_action_binary = 1
            j = _best_switch_idx(b)
            if j is not None:
                self._act_counts["switch"] += 1
                out = np.int64(j)
            else:
                out = np.int64(-2)
            if self.debug_actions:
                print(f"[ACTIONS] forced switch -> order_id={int(out)}")
            return out

        # Attack (a == 0)
        if a == 0:
            self.last_action_binary = 0
            self._act_counts["attack"] += 1
            i = _best_move_idx(b)
            out = np.int64(6 + (i or 0)) if i is not None else np.int64(-2)
            if self.debug_actions:
                print(f"[ACTIONS] a=attack i={i} -> order_id={int(out)}")
            return out

        # Switch (a == 1)
        self.last_action_binary = 1
        self._act_counts["switch"] += 1
        j = _best_switch_idx(b)
        if j is not None:
            out = np.int64(j)
            if self.debug_actions:
                print(f"[ACTIONS] a=switch j={j} -> order_id={int(out)}")
            return out

        # No switches available → fallback to best attack
        i = _best_move_idx(b)
        out = np.int64(6 + (i or 0)) if i is not None else np.int64(-2)
        if self.debug_actions:
            print(f"[ACTIONS] a=switch but no switches; fallback attack i={i} -> order_id={int(out)}")
        return out

    def get_additional_info(self) -> Dict[str, Dict[str, Any]]:
        info = super().get_additional_info()

        if self.battle1 is not None:
            agent = self.possible_agents[0]
            info[agent]["win"] = self.battle1.won
            info[agent]["action_counts"] = dict(self._act_counts)
            info[agent]["action_counts_str"] = f"A:{self._act_counts['attack']} S:{self._act_counts['switch']}"
            info[agent]["teacher_match_rate"] = (
                self._teacher_hits / self._teacher_total if self._teacher_total else 0.0
            )
            # Diagnostics to confirm mapping is being used
            info[agent]["process_action_seen"] = self._process_action_seen
            info[agent]["process_action_calls"] = self._process_action_calls

            # When a battle ends, reset counters for next episode
            if self.battle1.finished:
                self._act_counts = {"attack": 0, "switch": 0}
                self._teacher_hits = 0
                self._teacher_total = 0
                self.last_action_binary = None
                self._process_action_seen = False
                self._process_action_calls = 0

        return info

    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        Reward = small imitation tick + lightweight damage shaping + terminal win/loss.
        """
        us, them = battle.active_pokemon, battle.opponent_active_pokemon
        i = _best_move_idx(battle)
        j = _best_switch_idx(battle)

        move_score = 0.0
        if i is not None and battle.available_moves:
            move_score = _move_score(list(battle.available_moves)[i], us, them)
        switch_score = 0.0
        if j is not None and battle.available_switches:
            opp = battle.opponent_active_pokemon
            switch_score = _estimate_matchup(list(battle.available_switches)[j], opp) if opp else 0.0

        teacher_attack = move_score >= switch_score

        # Imitation tick only if we actually chose an action this step
        a_last = getattr(self, "last_action_binary", None)  # 0 attack, 1 switch
        imitation = 1.0 if (a_last is not None and ((a_last == 0) == teacher_attack)) else 0.0
        if a_last is not None:
            self._teacher_total += 1
            if imitation > 0.0:
                self._teacher_hits += 1

        r = 0.5 * imitation

        # Lightweight damage shaping using prior battle snapshot
        prior = self._get_prior_battle(battle)
        if prior is not None:
            def hp_vec(side):
                v = [float(m.current_hp_fraction) for m in side.values()]
                while len(v) < 6:
                    v.append(1.0)
                return v

            dealt = float(np.sum(np.array(hp_vec(prior.opponent_team)) - np.array(hp_vec(battle.opponent_team))))
            taken = float(np.sum(np.array(hp_vec(prior.team)) - np.array(hp_vec(battle.team))))
            r += dealt - 0.5 * taken

        if battle.won:
            r += 20.0
        if battle.lost:
            r -= 20.0

        return float(r)

    def _observation_size(self) -> int:
        # [move_score_n, switch_score_n, speed_edge, can_kill, opp_can_kill,
        #  force_switch, can_tera, my_hp, opp_hp, my_alive/6, opp_alive/6]
        return 11

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        us, them = battle.active_pokemon, battle.opponent_active_pokemon
        i_best = _best_move_idx(battle)
        j_best = _best_switch_idx(battle)

        # raw scores
        if i_best is not None and battle.available_moves:
            m = list(battle.available_moves)[i_best]
            move_score = _move_score(m, us, them)
        else:
            move_score = 0.0

        if j_best is not None and battle.available_switches:
            s = list(battle.available_switches)[j_best]
            switch_score = _estimate_matchup(s, them) if them else 0.0
        else:
            switch_score = 0.0

        # simple normalization
        def nz(x, scale):
            return float(x) / float(scale)

        move_n = nz(move_score, 200.0)   # ~big STAB BP baseline
        switch_n = nz(switch_score, 2.0) # matchup diff is small magnitude

        speed_edge = 1.0 if _speed_advantage(us, them) else 0.0
        can_kill = 1.0 if _can_kill_now(battle, i_best) else 0.0
        opp_can_kill = 1.0 if _opp_can_kill_now_heuristic(battle) else 0.0
        force_switch = 1.0 if battle.force_switch else 0.0
        can_tera = 1.0 if getattr(battle, "can_tera", False) else 0.0
        my_hp = _hp_frac(us)
        opp_hp = _hp_frac(them)
        my_alive = (sum(1 for p in (battle.team or {}).values() if not p.fainted) / 6.0) if battle.team else 1.0
        opp_alive = (sum(1 for p in (battle.opponent_team or {}).values() if not p.fainted) / 6.0) if battle.opponent_team else 1.0

        obs = np.array(
            [move_n, switch_n, speed_edge, can_kill, opp_can_kill,
             force_switch, can_tera, my_hp, opp_hp, my_alive, opp_alive],
            dtype=np.float32,
        )
        return obs


########################################
# DO NOT EDIT THE CODE BELOW THIS LINE #
########################################

class SingleShowdownWrapper(SingleAgentWrapper):
    """
    Simplifies setup of single-agent RL tasks for Pokémon battles.
    Do NOT edit this class.
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
