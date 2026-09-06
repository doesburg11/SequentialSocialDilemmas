"""Base map class that defines the rendering process

This is a copy of `map_env.py` (class renamed `MapEnvReputation`) with one
addition on top of the original `MapEnv`: an optional reputation intrinsic
reward, following McKee et al. (2023), "A multi-agent reinforcement learning
model of reputation and cooperation in human groups" (Eq. 2 / Eq. S1-S4):

    r = r_e + r_i
    r_i = -alpha * max(c_bar - c_self, 0) - beta * max(c_self - c_bar, 0)

where `c_self` is an agent's own exponentially-smoothed contribution level
(`c^t_j = lambda * c^{t-1}_j + q^t_j`, `q^t_j in {0, 1}` indicating whether
the agent "contributed" this step) and `c_bar` is the group's mean smoothed
contribution. `alpha`/`beta` are sampled once per agent at construction time
(population heterogeneity, matching the paper's `alpha ~ U(2.4, 3.0)`,
`beta ~ U(0.16, 0.20)`), not resampled across episodes.

The base `MapEnv` has no notion of "contribution" -- that is inherently
environment-specific (e.g. Cleanup's cleaning beam). This class only
maintains the smoothed EMA and applies the reward term; a subclass is
responsible for setting `agent.contributed_this_step = 1` on any agent that
"contributed" on a given step (see `cleanup_reputation.py::CleanupReputationEnv`).
Agents are reset to `contributed_this_step = 0` at the top of every `step()`
call and at `reset()`, so a subclass only ever needs to set the flag to 1.

`identifiable` toggles the paper's core manipulation: in the identifiable
condition the reputation term is computed from the full group's mean
contribution and added to reward; in the anonymous condition the term is
zeroed out entirely (contributions are still tracked, so post-hoc metrics
like territoriality/turn-taking can be computed for both conditions). This
is a simplified, binary reading of the paper's actual anonymous condition,
which instead restricts contribution visibility to a limited radius (Eq.
S5-S6-adjacent visibility range) rather than removing it outright -- see the
reputation experiment's README for this documented simplification.
"""

import matplotlib.pyplot as plt
import numpy as np
try:
    from gymnasium.spaces import Box, Dict
except ImportError:  # pragma: no cover - fallback for legacy gym installs
    from gym.spaces import Box, Dict

try:
    from ray.rllib.algorithms.callbacks import DefaultCallbacks
except ImportError:  # pragma: no cover - fallback for legacy Ray versions
    from ray.rllib.agents.callbacks import DefaultCallbacks

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except ImportError:  # pragma: no cover - fallback for legacy Ray versions
    from ray.rllib.env import MultiAgentEnv

_MAP_ENV_ACTIONS = {
    "MOVE_LEFT": [0, -1],  # Move left
    "MOVE_RIGHT": [0, 1],  # Move right
    "MOVE_UP": [-1, 0],  # Move up
    "MOVE_DOWN": [1, 0],  # Move down
    "STAY": [0, 0],  # don't move
    "TURN_CLOCKWISE": [[0, 1], [-1, 0]],  # Clockwise rotation matrix
    "TURN_COUNTERCLOCKWISE": [[0, -1], [1, 0]],
}  # Counter clockwise rotation matrix
# Positive Theta is in the counterclockwise direction

ORIENTATIONS = {"LEFT": [0, -1], "RIGHT": [0, 1], "UP": [-1, 0], "DOWN": [1, 0]}

DEFAULT_COLOURS = {
    b" ": np.array([0, 0, 0], dtype=np.uint8),  # Black background
    b"0": np.array([0, 0, 0], dtype=np.uint8),  # Black background beyond map walls
    b"": np.array([180, 180, 180], dtype=np.uint8),  # Grey board walls
    b"@": np.array([180, 180, 180], dtype=np.uint8),  # Grey board walls
    b"A": np.array([0, 255, 0], dtype=np.uint8),  # Green apples
    b"F": np.array([255, 255, 0], dtype=np.uint8),  # Yellow firing beam
    b"P": np.array([159, 67, 255], dtype=np.uint8),  # Generic agent (any player)
    # Colours for agents. R value is a unique identifier
    b"1": np.array([0, 0, 255], dtype=np.uint8),  # Pure blue
    b"2": np.array([2, 81, 154], dtype=np.uint8),  # Sky blue
    b"3": np.array([204, 0, 204], dtype=np.uint8),  # Magenta
    b"4": np.array([216, 30, 54], dtype=np.uint8),  # Red
    b"5": np.array([254, 151, 0], dtype=np.uint8),  # Orange
    b"6": np.array([100, 255, 255], dtype=np.uint8),  # Cyan
    b"7": np.array([99, 99, 255], dtype=np.uint8),  # Lavender
    b"8": np.array([250, 204, 255], dtype=np.uint8),  # Pink
    b"9": np.array([238, 223, 16], dtype=np.uint8),  # Yellow
}

# the axes look like this when printed out
# WARNING: increasing array position in the direction of down
# so for example if you move_left when facing left
# your y position decreases.
#         ^
#         |
#         U
#         P
# <--LEFT * RIGHT---->
#         D
#         O
#         W
#         N
#         |


class MapEnvReputation(MultiAgentEnv):
    @property
    def num_agents(self):
        return self._num_agents

    @num_agents.setter
    def num_agents(self, value):
        self._num_agents = value

    def __init__(
        self,
        ascii_map,
        extra_actions,
        view_len,
        num_agents=1,
        color_map=None,
        return_agent_actions=False,
        use_collective_reward=False,
        inequity_averse_reward=False,
        alpha=0.0,
        beta=0.0,
        reputation_reward=False,
        identifiable=True,
        reputation_alpha_range=(2.4, 3.0),
        reputation_beta_range=(0.16, 0.20),
        contribution_ema_lambda=0.97,
        contribution_scale=0.001,
        reputation_seed=None,
    ):
        """

        Parameters
        ----------
        ascii_map: list of strings
            Specify what the map should look like. Look at constant.py for
            further explanation
        extra_actions: dict with action name-value pair
            Environment-specific actions that are not present in _MAP_ENV_ACTIONS
        num_agents: int
            Number of agents to have in the system.
        color_map: dict
            Specifies how to convert between ascii chars and colors
        return_agent_actions: bool
            If true, the observation space will include the actions of other agents
        reputation_reward: bool
            If true, add McKee et al. (2023)'s reputation intrinsic reward on
            top of the environment's own reward (see module docstring).
        identifiable: bool
            McKee et al.'s identifiable/anonymous manipulation. If true, the
            reputation term is computed from the full group's contribution
            mean and added to reward. If false (anonymous), the reputation
            term is not added (contributions are still tracked internally).
        reputation_alpha_range, reputation_beta_range: (float, float)
            Sampling range for each agent's own alpha/beta, drawn once at
            construction (population heterogeneity, not resampled per
            episode). Defaults match the paper's alpha ~ U(2.4, 3.0),
            beta ~ U(0.16, 0.20).
        contribution_ema_lambda: float
            Smoothing factor for the contribution EMA (paper's lambda=0.97).
        contribution_scale: float
            Per-step contribution increment q^t_j when an agent contributes
            (Eq. S3 gives q in {0, 1}; this is the "1"). Default 0.001 =
            1/T for the T=1000-step episode, i.e. contributions measured as
            a fraction of the episode. The paper's Eq. S4 reads literally as
            q=1, but that is irreconcilable with its own figures: with q=1
            and lambda=0.97 the EMA saturates near 33 and a cleaner/non-
            cleaner divergence of order 1-30 would give per-step penalties of
            alpha*divergence ~ 3-90 -- dwarfing the +1/apple extrinsic
            reward, contradicting Fig. S2 (intrinsic ~20 vs extrinsic ~120
            per episode, "agents do not live off intrinsic reward"). Fig. S1
            plots the empirical divergence c_self - c_bar over +-0.004, which
            is exactly the unscaled range (+-4) divided by 1000. So the
            paper's contribution signal is effectively normalized by T; this
            parameter makes that explicit and tunable. Set to 1.0 for the
            literal-but-implausible reading.
        reputation_seed: int or None
            If given, seeds the process-global `np.random` immediately
            before sampling alpha/beta, so alpha/beta are reproducible for a
            given seed. This is intentionally separate from `reset(seed=)`:
            alpha/beta are sampled once here at construction and then held
            fixed for the agent's lifetime (population heterogeneity, not
            re-rolled per episode), so `reset(seed=)` -- which reseeds
            spawn-point/rotation randomness on every episode, matching the
            base `MapEnv`'s own behavior -- has no effect on them.
        """
        self.num_agents = num_agents
        self.base_map = self.ascii_to_numpy(ascii_map)
        self.view_len = view_len
        self.map_padding = view_len
        self.return_agent_actions = return_agent_actions
        self.use_collective_reward = use_collective_reward
        self.inequity_averse_reward = inequity_averse_reward
        self.alpha = alpha
        self.beta = beta
        self.reputation_reward = reputation_reward
        self.identifiable = identifiable
        self.contribution_ema_lambda = contribution_ema_lambda
        self.contribution_scale = contribution_scale
        self.all_actions = _MAP_ENV_ACTIONS.copy()
        self.all_actions.update(extra_actions)
        # Map without agents or beams
        self.world_map = np.full(
            (len(self.base_map), len(self.base_map[0])), fill_value=b" ", dtype="c"
        )
        # Color mapping
        self.color_map = color_map if color_map is not None else DEFAULT_COLOURS.copy()
        # World map image
        self.world_map_color = np.full(
            (len(self.base_map) + view_len * 2, len(self.base_map[0]) + view_len * 2, 3),
            fill_value=0,
            dtype=np.uint8,
        )
        self.beam_pos = []

        self.agents = {}

        # returns the agent at a desired position if there is one
        self.pos_dict = {}
        self.spawn_points = []  # where agents can appear

        self.wall_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == b"P":
                    self.spawn_points.append([row, col])
                elif self.base_map[row, col] == b"@":
                    self.wall_points.append([row, col])
        self.setup_agents()
        self._render_fig = None
        self._render_ax = None
        self._render_im = None

        if self.reputation_reward:
            assert self.num_agents > 1, "Cannot use reputation reward with only one agent!"
            if use_collective_reward or inequity_averse_reward:
                # The reputation term is applied after those two reward
                # transforms (see step()); composing it with either would
                # silently break the "collective" or inequity-computed-from-
                # raw-reward invariant those transforms assume. The base
                # MapEnv doesn't define a meaningful composition between
                # use_collective_reward and inequity_averse_reward either
                # (collectivizing rewards first makes every agent's reward
                # identical, so inequity aversion computed on top is a
                # no-op) -- these reward-shaping mechanisms are each meant
                # to be used on their own, not combined.
                raise ValueError(
                    "reputation_reward cannot be combined with use_collective_reward or "
                    "inequity_averse_reward; use exactly one reward-shaping mechanism."
                )
            if reputation_seed is not None:
                np.random.seed(reputation_seed)
            alpha_low, alpha_high = reputation_alpha_range
            beta_low, beta_high = reputation_beta_range
            self.reputation_alpha = {
                agent_id: np.random.uniform(alpha_low, alpha_high) for agent_id in self.agents
            }
            self.reputation_beta = {
                agent_id: np.random.uniform(beta_low, beta_high) for agent_id in self.agents
            }
            self.contribution_ema = {agent_id: 0.0 for agent_id in self.agents}
            for agent in self.agents.values():
                agent.contributed_this_step = 0

    @property
    def observation_space(self):
        obs_space = {
            "curr_obs": Box(
                low=0,
                high=255,
                shape=(2 * self.view_len + 1, 2 * self.view_len + 1, 3),
                dtype=np.uint8,
            )
        }
        if self.return_agent_actions:
            # Append the actions of other agents
            obs_space = {
                **obs_space,
                "other_agent_actions": Box(
                    low=0,
                    high=len(self.all_actions),
                    shape=(self.num_agents - 1,),
                    dtype=np.uint8,
                ),
                "visible_agents": Box(
                    low=0,
                    high=1,
                    shape=(self.num_agents - 1,),
                    dtype=np.uint8,
                ),
                "prev_visible_agents": Box(
                    low=0,
                    high=1,
                    shape=(self.num_agents - 1,),
                    dtype=np.uint8,
                ),
            }
        obs_space = Dict(obs_space)
        # Change dtype so that ray can put all observations into one flat batch
        # with the correct dtype.
        # See DictFlatteningPreprocessor in ray/rllib/models/preprocessors.py.
        obs_space.dtype = np.uint8
        return obs_space

    def custom_reset(self):
        """Reset custom elements of the map. For example, spawn apples and build walls"""
        pass

    def custom_action(self, agent, action):
        """Execute any custom actions that may be defined, like fire or clean

        Parameters
        ----------
        agent: agent that is taking the action
        action: key of the action to be taken

        Returns
        -------
        updates: list(list(row, col, char))
            List of cells to place onto the map
        """
        pass

    def custom_map_update(self):
        """Custom map updates that don't have to do with agent actions"""
        pass

    def setup_agents(self):
        """Construct all the agents for the environment"""
        raise NotImplementedError

    # FIXME(ev) move this to a utils eventually
    def ascii_to_numpy(self, ascii_list):
        """converts a list of strings into a numpy array


        Parameters
        ----------
        ascii_list: list of strings
            List describing what the map should look like
        Returns
        -------
        arr: np.ndarray
            numpy array describing the map with ' ' indicating an empty space
        """

        arr = np.full((len(ascii_list), len(ascii_list[0])), b" ", dtype="c")
        for row in range(arr.shape[0]):
            for col in range(arr.shape[1]):
                arr[row, col] = ascii_list[row][col]
        return arr

    def step(self, actions):
        """Takes in a dict of actions and converts them to a map update

        Parameters
        ----------
        actions: dict {agent-id: int}
            dict of actions, keyed by agent-id that are passed to the agent. The agent
            interprets the int and converts it to a command

        Returns
        -------
        observations: dict of arrays representing agent observations
        rewards: dict of rewards for each agent
        dones: dict indicating whether each agent is done
        info: dict to pass extra info to gym
        """

        self.beam_pos = []
        if self.reputation_reward:
            # A subclass's custom_action() is responsible for setting this to
            # 1 for any agent that "contributed" this step (e.g. Cleanup's
            # cleaning beam actually removing waste); reset here so an agent
            # that doesn't act on its contribution channel this step reads
            # back as 0, not last step's stale value.
            for agent in self.agents.values():
                agent.contributed_this_step = 0

        agent_actions = {}
        for agent_id, action in actions.items():
            agent_action = self.agents[agent_id].action_map(action)
            agent_actions[agent_id] = agent_action

        # Remove agents from color map
        for agent in self.agents.values():
            row, col = agent.pos[0], agent.pos[1]
            self.single_update_world_color_map(row, col, self.world_map[row, col])

        self.update_moves(agent_actions)

        for agent in self.agents.values():
            pos = agent.pos
            new_char = agent.consume(self.world_map[pos[0], pos[1]])
            self.single_update_map(pos[0], pos[1], new_char)

        # execute custom moves like firing
        self.update_custom_moves(agent_actions)

        # execute spawning events
        self.custom_map_update()

        map_with_agents = self.get_map_with_agents()
        # Add agents to color map
        for agent in self.agents.values():
            row, col = agent.pos[0], agent.pos[1]
            # Firing beams have priority over agents and should cover them
            if self.world_map[row, col] not in [b"F", b"C"]:
                self.single_update_world_color_map(row, col, agent.get_char_id())

        observations = {}
        rewards = {}
        dones = {}
        infos = {}
        for agent in self.agents.values():
            agent.full_map = map_with_agents
            rgb_arr = self.color_view(agent)
            # concatenate on the prev_actions to the observations
            if self.return_agent_actions:
                prev_actions = np.array(
                    [actions[key] for key in sorted(actions.keys()) if key != agent.agent_id]
                ).astype(np.uint8)
                visible_agents = self.find_visible_agents(agent.agent_id)
                observations[agent.agent_id] = {
                    "curr_obs": rgb_arr,
                    "other_agent_actions": prev_actions,
                    "visible_agents": visible_agents,
                    "prev_visible_agents": agent.prev_visible_agents,
                }
                agent.prev_visible_agents = visible_agents
            else:
                observations[agent.agent_id] = {"curr_obs": rgb_arr}
            rewards[agent.agent_id] = agent.compute_reward()
            dones[agent.agent_id] = agent.get_done()
            infos[agent.agent_id] = {}

        if self.use_collective_reward:
            collective_reward = sum(rewards.values())
            for agent in rewards.keys():
                rewards[agent] = collective_reward
        if self.inequity_averse_reward:
            assert self.num_agents > 1, "Cannot use inequity aversion with only one agent!"
            temp_rewards = rewards.copy()
            for agent in rewards.keys():
                diff = np.array([r - rewards[agent] for r in rewards.values()])
                dis_inequity = self.alpha * sum(diff[diff > 0])
                adv_inequity = self.beta * sum(diff[diff < 0])
                temp_rewards[agent] -= (dis_inequity + adv_inequity) / (self.num_agents - 1)
            rewards = temp_rewards
        if self.reputation_reward:
            assert self.num_agents > 1, "Cannot use reputation reward with only one agent!"
            for agent_id, agent in self.agents.items():
                contributed = self.contribution_scale * float(
                    getattr(agent, "contributed_this_step", 0)
                )
                self.contribution_ema[agent_id] = (
                    self.contribution_ema_lambda * self.contribution_ema[agent_id] + contributed
                )
            if self.identifiable:
                mean_contribution = float(np.mean(list(self.contribution_ema.values())))
                for agent_id in rewards.keys():
                    c_self = self.contribution_ema[agent_id]
                    alpha_i = self.reputation_alpha[agent_id]
                    beta_i = self.reputation_beta[agent_id]
                    rewards[agent_id] += -alpha_i * max(
                        mean_contribution - c_self, 0.0
                    ) - beta_i * max(c_self - mean_contribution, 0.0)
            # else (anonymous): contribution EMAs are still tracked above (for
            # post-hoc territoriality/turn-taking metrics) but no reputation
            # term is added to reward -- this repo's simplified reading of
            # McKee et al.'s identifiable/anonymous manipulation.

        dones["__all__"] = np.any(list(dones.values()))
        return observations, rewards, dones, infos

    def reset(self, *, seed=None, options=None):
        """Reset the environment.

        This method is performed in between rollouts. It resets the state of
        the environment.

        Returns
        -------
        observation: dict of numpy ndarray
            the initial observation of the space. The initial reward is assumed
            to be zero.
        """
        if seed is not None:
            self.seed(seed)

        self.beam_pos = []
        self.agents = {}
        self.setup_agents()
        self.reset_map()
        self.custom_map_update()

        if self.reputation_reward:
            # Contribution EMA is episode-local (paper's c^0_j = 0); alpha/beta
            # stay fixed for the agent's lifetime, sampled once in __init__.
            self.contribution_ema = {agent_id: 0.0 for agent_id in self.agents}
            for agent in self.agents.values():
                agent.contributed_this_step = 0

        map_with_agents = self.get_map_with_agents()

        observations = {}
        for agent in self.agents.values():
            agent.full_map = map_with_agents
            rgb_arr = self.color_view(agent)
            # concatenate on the prev_actions to the observations
            if self.return_agent_actions:
                # No previous actions so just pass in "wait" action
                prev_actions = np.array([4 for _ in range(self.num_agents - 1)]).astype(np.uint8)
                visible_agents = self.find_visible_agents(agent.agent_id)
                observations[agent.agent_id] = {
                    "curr_obs": rgb_arr,
                    "other_agent_actions": prev_actions,
                    "visible_agents": visible_agents,
                    "prev_visible_agents": visible_agents,
                }
                agent.prev_visible_agents = visible_agents
            else:
                observations[agent.agent_id] = {"curr_obs": rgb_arr}
        return observations

    def seed(self, seed=None):
        np.random.seed(seed)

    def close(self):
        plt.close()

    @property
    def agent_pos(self):
        return [agent.pos.tolist() for agent in self.agents.values()]

    def get_map_with_agents(self):
        """Gets a version of the environment map where generic
        'P' characters have been replaced with specific agent IDs.

        Returns:
            2D array of strings representing the map.
        """
        grid = np.copy(self.world_map)

        for agent in self.agents.values():
            char_id = agent.get_char_id()

            # If agent is not within map, skip.
            if not (0 <= agent.pos[0] < grid.shape[0] and 0 <= agent.pos[1] < grid.shape[1]):
                continue

            grid[agent.pos[0], agent.pos[1]] = char_id

        # beams should overlay agents
        for beam_pos in self.beam_pos:
            grid[beam_pos[0], beam_pos[1]] = beam_pos[2]

        return grid

    def check_agent_map(self, agent_map):
        """Checks the map to make sure agents aren't duplicated"""
        unique, counts = np.unique(agent_map, return_counts=True)
        count_dict = dict(zip(unique, counts))

        # check for multiple agents
        for i in range(self.num_agents):
            if count_dict[chr(i + 1)] != 1:
                print("Error! Wrong number of agent", i, "in map!")
                return False
        return True

    def full_map_to_colors(self):
        map_with_agents = self.get_map_with_agents()
        rgb_arr = np.zeros((map_with_agents.shape[0], map_with_agents.shape[1], 3), dtype=int)
        return self.map_to_colors(map_with_agents, self.color_map, rgb_arr)

    def color_view(self, agent):
        row, col = agent.pos[0], agent.pos[1]
        view_slice = self.world_map_color[
            row + self.map_padding - self.view_len : row + self.map_padding + self.view_len + 1,
            col + self.map_padding - self.view_len : col + self.map_padding + self.view_len + 1,
        ]
        if agent.orientation == "UP":
            rotated_view = view_slice
        elif agent.orientation == "LEFT":
            rotated_view = np.rot90(view_slice)
        elif agent.orientation == "DOWN":
            rotated_view = np.rot90(view_slice, k=2)
        elif agent.orientation == "RIGHT":
            rotated_view = np.rot90(view_slice, k=1, axes=(1, 0))
        return rotated_view

    def map_to_colors(self, mmap, color_map, rgb_arr, orientation="UP"):
        """Converts a map to an array of RGB values.
        Parameters
        ----------
        mmap: np.ndarray
            map to convert to colors
            Double m to avoid shadowing map.
        color_map: dict
            mapping between array elements and desired colors
        rgb_arr: np.array
            Variable to store the mapping in
        orientation:
            The way in which the output should be oriented.
             UP = no rotation.
             RIGHT = Clockwise 90 degree rotation.
             DOWN = Clockwise 180 degree rotation.
             LEFT = Clockwise 270 degree rotation.
        Returns
        -------
        arr: np.ndarray
            3-dim numpy array consisting of color map
        """
        x_len = mmap.shape[0]
        y_len = mmap.shape[1]
        if orientation == "UP":
            for row_elem in range(x_len):
                for col_elem in range(y_len):
                    rgb_arr[row_elem, col_elem, :] = color_map[mmap[row_elem, col_elem]]
        elif orientation == "LEFT":
            for row_elem in range(x_len):
                for col_elem in range(y_len):
                    rgb_arr[row_elem, col_elem, :] = color_map[mmap[col_elem, x_len - 1 - row_elem]]
        elif orientation == "DOWN":
            for row_elem in range(x_len):
                for col_elem in range(y_len):
                    rgb_arr[row_elem, col_elem, :] = color_map[
                        mmap[x_len - 1 - row_elem, y_len - 1 - col_elem]
                    ]
        elif orientation == "RIGHT":
            for row_elem in range(x_len):
                for col_elem in range(y_len):
                    rgb_arr[row_elem, col_elem, :] = color_map[mmap[y_len - 1 - col_elem, row_elem]]
        else:
            raise ValueError("Orientation {} is not valid".format(orientation))

        return rgb_arr

    def render(self, filename=None, mode="human"):
        """Creates an image of the map to plot or save.

        Args:
            filename: If a string is passed, will save the image
                      to disk at this location.
        """
        rgb_arr = self.full_map_to_colors()
        if mode == "human":
            plt.ion()
            if filename is None:
                needs_init = (
                    self._render_fig is None
                    or self._render_ax is None
                    or self._render_im is None
                    or not plt.fignum_exists(self._render_fig.number)
                )
                if needs_init:
                    self._render_fig, self._render_ax = plt.subplots()
                    self._render_im = self._render_ax.imshow(rgb_arr, interpolation="nearest")
                    self._render_ax.set_axis_off()
                    self._render_fig.tight_layout(pad=0)
                    plt.show(block=False)
                else:
                    self._render_im.set_data(rgb_arr)
                    self._render_fig.canvas.draw_idle()

                # Pump GUI events so interactive windows appear and update.
                self._render_fig.canvas.flush_events()
                plt.pause(0.001)
            else:
                fig, ax = plt.subplots()
                ax.imshow(rgb_arr, interpolation="nearest")
                ax.set_axis_off()
                fig.tight_layout(pad=0)
                fig.savefig(filename)
                plt.close(fig)
            return None
        return rgb_arr

    def update_moves(self, agent_actions):
        """Converts agent action tuples into a new map and new agent positions.
         Also resolves conflicts over multiple agents wanting a cell.

         This method works by finding all conflicts over a cell and randomly assigning them
        to one of the agents that desires the slot. It then sets all of the other agents
        that wanted the cell to have a move of staying. For moves that do not directly
        conflict with another agent for a cell, but may not be temporarily resolvable
        due to an agent currently being in the desired cell, we continually loop through
        the actions until all moves have been satisfied or deemed impossible.
        For example, agent 1 may want to move from [1,2] to [2,2] but agent 2 is in [2,2].
        Agent 2, however, is moving into [3,2]. Agent-1's action is first in the order so at the
        first pass it is skipped but agent-2 moves to [3,2]. In the second pass, agent-1 will
        then be able to move into [2,2].

         Parameters
         ----------
         agent_actions: dict
             dict with agent_id as key and action as value
        """

        reserved_slots = []
        for agent_id, action in agent_actions.items():
            agent = self.agents[agent_id]
            selected_action = self.all_actions[action]
            # TODO(ev) these two parts of the actions
            if "MOVE" in action or "STAY" in action:
                # rotate the selected action appropriately
                rot_action = self.rotate_action(selected_action, agent.get_orientation())
                new_pos = agent.pos + rot_action
                # allow the agents to confirm what position they can move to
                new_pos = agent.return_valid_pos(new_pos)
                reserved_slots.append((*new_pos, b"P", agent_id))
            elif "TURN" in action:
                new_rot = self.update_rotation(action, agent.get_orientation())
                agent.update_agent_rot(new_rot)

        # now do the conflict resolution part of the process

        # helpful for finding the agent in the conflicting slot
        agent_by_pos = {tuple(agent.pos): agent.agent_id for agent in self.agents.values()}

        # agent moves keyed by ids
        agent_moves = {}

        # lists of moves and their corresponding agents
        move_slots = []
        agent_to_slot = []

        for slot in reserved_slots:
            row, col = slot[0], slot[1]
            if slot[2] == b"P":
                agent_id = slot[3]
                agent_moves[agent_id] = [row, col]
                move_slots.append([row, col])
                agent_to_slot.append(agent_id)

        # cut short the computation if there are no moves
        if len(agent_to_slot) > 0:

            # first we will resolve all slots over which multiple agents
            # want the slot

            # shuffle so that a random agent has slot priority
            shuffle_list = list(zip(agent_to_slot, move_slots))
            np.random.shuffle(shuffle_list)
            agent_to_slot, move_slots = zip(*shuffle_list)
            unique_move, indices, return_count = np.unique(
                move_slots, return_index=True, return_counts=True, axis=0
            )
            search_list = np.array(move_slots)

            # first go through and remove moves that can't possible happen. Three types
            # 1. Trying to move into an agent that has been issued a stay command
            # 2. Trying to move into the spot of an agent that doesn't have a move
            # 3. Two agents trying to walk through one another

            # Resolve all conflicts over a space
            if np.any(return_count > 1):
                for move, index, count in zip(unique_move, indices, return_count):
                    if count > 1:
                        # check that the cell you are fighting over doesn't currently
                        # contain an agent that isn't going to move for one of the agents
                        # If it does, all the agents commands should become STAY
                        # since no moving will be possible
                        conflict_indices = np.where((search_list == move).all(axis=1))[0]
                        all_agents_id = [agent_to_slot[i] for i in conflict_indices]
                        # all other agents now stay in place so update their moves
                        # to reflect this
                        conflict_cell_free = True
                        for agent_id in all_agents_id:
                            moves_copy = agent_moves.copy()
                            # TODO(ev) code duplication, simplify
                            if move.tolist() in self.agent_pos:
                                # find the agent that is currently at that spot and make sure
                                # that the move is possible. If it won't be, remove it.
                                conflicting_agent_id = agent_by_pos[tuple(move)]
                                curr_pos = self.agents[agent_id].pos.tolist()
                                curr_conflict_pos = self.agents[conflicting_agent_id].pos.tolist()
                                conflict_move = agent_moves.get(
                                    conflicting_agent_id, curr_conflict_pos
                                )
                                # Condition (1):
                                # a STAY command has been issued
                                if agent_id == conflicting_agent_id:
                                    conflict_cell_free = False
                                # Condition (2)
                                # its command is to stay
                                # or you are trying to move into an agent that hasn't
                                # received a command
                                elif (
                                    conflicting_agent_id not in moves_copy.keys()
                                    or curr_conflict_pos == conflict_move
                                ):
                                    conflict_cell_free = False

                                # Condition (3)
                                # It is trying to move into you and you are moving into it
                                elif conflicting_agent_id in moves_copy.keys():
                                    if (
                                        agent_moves[conflicting_agent_id] == curr_pos
                                        and move.tolist()
                                        == self.agents[conflicting_agent_id].pos.tolist()
                                    ):
                                        conflict_cell_free = False

                        # if the conflict cell is open, let one of the conflicting agents
                        # move into it
                        if conflict_cell_free:
                            self.agents[agent_to_slot[index]].update_agent_pos(move)
                            agent_by_pos = {
                                tuple(agent.pos): agent.agent_id for agent in self.agents.values()
                            }
                        # ------------------------------------
                        # remove all the other moves that would have conflicted
                        remove_indices = np.where((search_list == move).all(axis=1))[0]
                        all_agents_id = [agent_to_slot[i] for i in remove_indices]
                        # all other agents now stay in place so update their moves
                        # to stay in place
                        for agent_id in all_agents_id:
                            agent_moves[agent_id] = self.agents[agent_id].pos.tolist()

            # make the remaining un-conflicted moves
            while len(agent_moves.items()) > 0:
                agent_by_pos = {tuple(agent.pos): agent.agent_id for agent in self.agents.values()}
                num_moves = len(agent_moves.items())
                moves_copy = agent_moves.copy()
                del_keys = []
                for agent_id, move in moves_copy.items():
                    if agent_id in del_keys:
                        continue
                    if move in self.agent_pos:
                        # find the agent that is currently at that spot and make sure
                        # that the move is possible. If it won't be, remove it.
                        conflicting_agent_id = agent_by_pos[tuple(move)]
                        curr_pos = self.agents[agent_id].pos.tolist()
                        curr_conflict_pos = self.agents[conflicting_agent_id].pos.tolist()
                        conflict_move = agent_moves.get(conflicting_agent_id, curr_conflict_pos)
                        # Condition (1):
                        # a STAY command has been issued
                        if agent_id == conflicting_agent_id:
                            del agent_moves[agent_id]
                            del_keys.append(agent_id)
                        # Condition (2)
                        # its command is to stay
                        # or you are trying to move into an agent that hasn't received a command
                        elif (
                            conflicting_agent_id not in moves_copy.keys()
                            or curr_conflict_pos == conflict_move
                        ):
                            del agent_moves[agent_id]
                            del_keys.append(agent_id)
                        # Condition (3)
                        # It is trying to move into you and you are moving into it
                        elif conflicting_agent_id in moves_copy.keys():
                            if (
                                agent_moves[conflicting_agent_id] == curr_pos
                                and move == self.agents[conflicting_agent_id].pos.tolist()
                            ):
                                del agent_moves[conflicting_agent_id]
                                del agent_moves[agent_id]
                                del_keys.append(agent_id)
                                del_keys.append(conflicting_agent_id)
                    # this move is unconflicted so go ahead and move
                    else:
                        self.agents[agent_id].update_agent_pos(move)
                        del agent_moves[agent_id]
                        del_keys.append(agent_id)

                # no agent is able to move freely, so just move them all
                # no updates to hidden cells are needed since all the
                # same cells will be covered
                if len(agent_moves) == num_moves:
                    for agent_id, move in agent_moves.items():
                        self.agents[agent_id].update_agent_pos(move)
                    break

    def update_custom_moves(self, agent_actions):
        """
        This function executes non-movement actions like firing, cleaning etc.
        The order in which agent actions are resolved is random to ensure homogeneity, similar to
        update_moves, otherwise a race condition occurs which prioritizes lower-numbered agents
        """
        agent_ids = list(agent_actions.keys())
        np.random.shuffle(agent_ids)
        for agent_id in agent_ids:
            action = agent_actions[agent_id]
            # check its not a move based action
            if "MOVE" not in action and "STAY" not in action and "TURN" not in action:
                agent = self.agents[agent_id]
                updates = self.custom_action(agent, action)
                if len(updates) > 0:
                    self.update_map(updates)

    def update_map(self, new_points):
        """For points in new_points, place desired char on the map
        Update the color map as well"""
        for point in new_points:
            self.single_update_map(*point)

    def single_update_map(self, row, col, char):
        self.world_map[row, col] = char
        self.world_map_color[row + self.map_padding, col + self.map_padding] = self.color_map[char]

    def single_update_world_color_map(self, row, col, char):
        """Only update the color map. This is done separately when agents move, because their own
        position state is not contained in self.world_map, but in their own Agent objects"""
        self.world_map_color[row + self.map_padding, col + self.map_padding] = self.color_map[char]

    def reset_map(self):
        """Resets the map to be empty as well as a custom reset set by subclasses"""
        self.world_map = np.full((len(self.base_map), len(self.base_map[0])), b" ", dtype="c")
        self.world_map_color = np.full(
            (len(self.base_map) + self.view_len * 2, len(self.base_map[0]) + self.view_len * 2, 3),
            fill_value=0,
            dtype=np.uint8,
        )
        self.build_walls()
        self.custom_reset()

    def update_map_fire(
        self,
        firing_pos,
        firing_orientation,
        fire_len,
        fire_char,
        cell_types=[],
        update_char=[],
        blocking_cells=b"P",
        beam_width=3,
    ):
        """From a firing position, fire a beam that may clean or hit agents

        Notes:
            (1) Beams are blocked by agents
            (2) A beam travels along until it hits a blocking cell at which beam the beam
                covers that cell and stops
            (3) If a beam hits a cell whose character is in cell_types, it replaces it with
                the corresponding index in update_char
            (4) As per the rules, the beams fire from in front of the agent and on its
                sides so the beam that starts in front of the agent travels out one
                cell further than it does along the sides.
            (5) This method updates the beam_pos, an internal representation of how
                which cells need to be rendered with fire_char in the agent view

        Parameters
        ----------
        firing_pos: (list)
            the row, col from which the beam is fired
        firing_orientation: (string)
            the direction the beam is to be fired in
        fire_len: (int)
            the number of cells forward to fire
        fire_char: (bytes)
            the cell that should be placed where the beam goes
        cell_types: (list of bytes)
            the cells that are affected by the beam
        update_char: (list of bytes)
            the character that should replace the affected cells.
        blocking_cells: (list of bytes)
            cells that block the firing beam
        Returns
        -------
        updates: (tuple (row, col, char))
            the cells that have been hit by the beam and what char will be placed there
        """
        agent_by_pos = {tuple(agent.pos): agent_id for agent_id, agent in self.agents.items()}
        start_pos = np.asarray(firing_pos)
        firing_direction = ORIENTATIONS[firing_orientation]
        # compute the other two starting positions
        right_shift = self.rotate_right(firing_direction)
        if beam_width == 1:
            firing_pos = [start_pos]
        elif beam_width == 3:
            firing_pos = [
                start_pos,
                start_pos + right_shift - firing_direction,
                start_pos - right_shift - firing_direction,
            ]
        else:
            raise NotImplementedError()
        firing_points = []
        updates = []
        for pos in firing_pos:
            next_cell = pos + firing_direction
            for i in range(fire_len):
                if (
                    self.test_if_in_bounds(next_cell)
                    and self.world_map[next_cell[0], next_cell[1]] != b"@"
                ):
                    # Update the cell if needed
                    firing_points.append((next_cell[0], next_cell[1], fire_char))
                    for c in range(len(cell_types)):
                        if self.world_map[next_cell[0], next_cell[1]] == cell_types[c]:
                            updates.append((next_cell[0], next_cell[1], update_char[c]))
                            break

                    # agents absorb beams
                    # activate the agents hit function if needed
                    if [next_cell[0], next_cell[1]] in self.agent_pos:
                        agent_id = agent_by_pos[(next_cell[0], next_cell[1])]
                        self.agents[agent_id].hit(fire_char)
                        break

                    # check if the cell blocks beams. For example, waste blocks beams.
                    if self.world_map[next_cell[0], next_cell[1]] in blocking_cells:
                        break

                    # increment the beam position
                    next_cell += firing_direction

                else:
                    break

        self.beam_pos += firing_points
        return updates

    def spawn_point(self):
        """Returns a randomly selected spawn point."""
        spawn_index = 0
        is_free_cell = False
        curr_agent_pos = [agent.pos.tolist() for agent in self.agents.values()]
        np.random.shuffle(self.spawn_points)
        for i, spawn_point in enumerate(self.spawn_points):
            if [spawn_point[0], spawn_point[1]] not in curr_agent_pos:
                spawn_index = i
                is_free_cell = True
        assert is_free_cell, "There are not enough spawn points! Check your map?"
        return np.array(self.spawn_points[spawn_index])

    def spawn_rotation(self):
        """Return a randomly selected initial rotation for an agent"""
        rand_int = np.random.randint(len(ORIENTATIONS.keys()))
        return list(ORIENTATIONS.keys())[rand_int]

    def build_walls(self):
        for i in range(len(self.wall_points)):
            row, col = self.wall_points[i]
            self.single_update_map(row, col, b"@")

    ########################################
    # Utility methods, move these eventually
    ########################################

    # TODO(ev) this can be a general property of map_env or a util
    def rotate_action(self, action_vec, orientation):
        # WARNING: Note, we adopt the physics convention that \theta=0 is in the +y direction
        if orientation == "UP":
            return action_vec
        elif orientation == "LEFT":
            return self.rotate_left(action_vec)
        elif orientation == "RIGHT":
            return self.rotate_right(action_vec)
        else:
            return self.rotate_left(self.rotate_left(action_vec))

    def rotate_left(self, action_vec):
        return np.dot(self.all_actions["TURN_COUNTERCLOCKWISE"], action_vec)

    def rotate_right(self, action_vec):
        return np.dot(self.all_actions["TURN_CLOCKWISE"], action_vec)

    # TODO(ev) this should be an agent property
    def update_rotation(self, action, curr_orientation):
        if action == "TURN_COUNTERCLOCKWISE":
            if curr_orientation == "LEFT":
                return "DOWN"
            elif curr_orientation == "DOWN":
                return "RIGHT"
            elif curr_orientation == "RIGHT":
                return "UP"
            else:
                return "LEFT"
        else:
            if curr_orientation == "LEFT":
                return "UP"
            elif curr_orientation == "UP":
                return "RIGHT"
            elif curr_orientation == "RIGHT":
                return "DOWN"
            else:
                return "LEFT"

    # TODO(ev) this definitely should go into utils or the general agent class
    def test_if_in_bounds(self, pos):
        """Checks if a selected cell is outside the range of the map"""
        return 0 <= pos[0] < self.world_map.shape[0] and 0 <= pos[1] < self.world_map.shape[1]

    def find_visible_agents(self, agent_id):
        """Returns all the agents that can be seen by agent with agent_id
        Args
        ----
        agent_id: str
            The id of the agent whose visible agents we are asking about
        Returns
        -------
        visible_agents: list
            which agents can be seen by the agent with id "agent_id"
        """
        agent_pos = self.agents[agent_id].pos
        upper_lim = int(agent_pos[0] + self.agents[agent_id].row_size)
        lower_lim = int(agent_pos[0] - self.agents[agent_id].row_size)
        left_lim = int(agent_pos[1] - self.agents[agent_id].col_size)
        right_lim = int(agent_pos[1] + self.agents[agent_id].col_size)

        # keep this sorted so the visibility matrix is always in order
        other_agent_pos = [
            self.agents[other_agent_id].pos
            for other_agent_id in sorted(self.agents.keys())
            if other_agent_id != agent_id
        ]
        return np.array(
            [
                1
                if (lower_lim <= agent_tup[0] <= upper_lim and left_lim <= agent_tup[1] <= right_lim)
                else 0
                for agent_tup in other_agent_pos
            ],
            dtype=np.uint8,
        )

    @staticmethod
    def get_environment_callbacks():
        return DefaultCallbacks
