"""Post-hoc territoriality and turn-taking metrics for the Cleanup reputation
experiment, implementing Eqs. S14-S17 (territoriality, Sec. 6) and Table S2's
recency-weighted turn-taking score (Sec. 7) of McKee et al. (2023), "A
multi-agent reinforcement learning model of reputation and cooperation in
human groups" (supplementary information). The paper's third temporal
metric -- contribution consistency (Eqs. S18-S19) -- is not implemented here;
it's a documented gap, not an oversight.

These are pure functions over logged rollout data (agent positions in the
river region, and the ordered sequence of "turns" entering the river to
clean) -- they don't depend on RLlib or any live environment, so they can be
computed from any rollout log (random-policy smoke test or a trained
checkpoint's rollout) with the same log format produced by
`run_scripts/rollout_reputation_cleanup.py`.
"""

from __future__ import annotations

import numpy as np

# Table S2: recency value as a function of turns elapsed since this group
# member's own last turn. A gap of 4 or more turns maps to 0 (not present in
# the dict; see `_RECENCY_BY_GAP.get(gap, 0.0)` below).
_RECENCY_BY_GAP = {0: 1.0, 1: 0.75, 2: 0.50, 3: 0.25}


def compute_territoriality(river_positions_by_step):
    """Normalized beta diversity over river-region visits (Eq. S14-S17).

    Parameters
    ----------
    river_positions_by_step : list of dict
        One entry per logged timestep the agent was inside the river region;
        each entry maps `agent_id -> (row, col)` for every agent present in
        the river region that step. Steps where no agent is in the river
        region may be omitted or passed as `{}`.

    Returns
    -------
    float
        Normalized beta diversity in the paper's Eq. S17 sense: 1.0 means
        every visited river cell was visited by a completely disjoint set of
        agents (maximal territoriality); low values mean river cells tend to
        be shared by the same group composition (low territoriality). Per
        the paper's own stated bounds, this is lower-bounded by
        `1 / min(gamma_d, N_l)`, not literally 0 -- with `gamma_d` capped at
        the group size (e.g. 5), the practical floor is around 0.2, not 0,
        for typical rollouts. The paper's own qualitative description ("0 =
        identical, 1 = disjoint") is a loose gloss over this, not an exact
        bound; this implements the formula literally.
    """
    location_members: dict[tuple, set] = {}
    for step_positions in river_positions_by_step:
        for agent_id, pos in step_positions.items():
            location_members.setdefault(tuple(pos), set()).add(agent_id)

    if not location_members:
        return 0.0

    num_locations = len(location_members)
    alpha_d = float(np.mean([len(members) for members in location_members.values()]))
    gamma_d = len({agent_id for members in location_members.values() for agent_id in members})

    if alpha_d == 0 or gamma_d == 0:
        return 0.0

    beta_d = gamma_d / alpha_d
    denominator = min(gamma_d, num_locations)
    if denominator == 0:
        return 0.0
    return beta_d / denominator


def compute_turn_taking(turn_sequence):
    """Recency-weighted turn-taking score (Sec. 7 / Table S2).

    Parameters
    ----------
    turn_sequence : list
        Ordered sequence of agent_ids, one entry per "turn" an agent took
        entering the river (the sequence of outside->inside transitions into
        the river region, in the order they happened across the episode; a
        continuous stay counts as a single turn regardless of length).

    Returns
    -------
    float
        Turn-taking score in [0, 1]: near 0 means a single agent dominates
        the river (no rotation); near 1 means agents rotate through the
        river with long gaps between any one agent's own turns.

    Notes
    -----
    An agent's very first turn in the sequence has no prior turn to measure
    a gap from. This implementation treats that case as maximally "stale"
    (gap >= 4, recency 0) rather than excluding it from the average -- a
    documented interpretation choice, since the paper doesn't specify this
    edge case. Figure 2c/3c's illustrative 10-turn excerpts are the *start*
    of a full 1000-step episode's turn sequence, not a self-contained window
    the reported 0.24/0.85 scores can be reproduced from -- this function's
    correctness is validated against boundary cases (single-agent monopoly,
    perfect round-robin), not against those two example scores.
    """
    if not turn_sequence:
        return 0.0

    last_turn_index: dict = {}
    recencies = []
    for i, agent_id in enumerate(turn_sequence):
        if agent_id in last_turn_index:
            gap = i - last_turn_index[agent_id] - 1
        else:
            gap = 4  # no prior turn recorded this episode -> treated as maximally stale
        recencies.append(_RECENCY_BY_GAP.get(gap, 0.0))
        last_turn_index[agent_id] = i

    return 1.0 - float(np.mean(recencies))
