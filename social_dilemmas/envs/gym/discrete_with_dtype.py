import numpy as np

try:
    from gymnasium.spaces import Discrete
except ImportError:  # pragma: no cover - fallback for legacy gym installs
    from gym.spaces import Discrete


class DiscreteWithDType(Discrete):
    def __init__(self, n, dtype):
        assert n >= 0
        # Gymnasium's Discrete defines `start`; initialize through parent when possible.
        try:
            super().__init__(n=n, start=0, dtype=dtype)
        except TypeError:
            try:
                super().__init__(n=n, dtype=dtype)
            except TypeError:
                # Very old gym fallback without dtype support.
                super().__init__(n=n)
                self.dtype = dtype

        if not hasattr(self, "start"):
            self.start = 0

    def contains(self, x):
        """Membership by value, not by dtype-safe-castability.

        Gymnasium's own `Discrete.contains` additionally requires
        `np.can_cast(x.dtype, self.dtype)`, which rejects e.g. a plain
        `np.int32` action against our deliberately narrow `np.uint8`
        action spaces (8-9 actions), even though the *value* is valid.
        `np.int32`/`np.int64` are exactly what PettingZoo's own API
        conformance tests (and many RL frameworks) pass by default, so
        the stricter check makes these environments fail PettingZoo
        compliance over a storage-dtype detail the game logic never
        relies on: actions are read as plain Python ints everywhere
        they're consumed (see `Agent.action_map`).
        """
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and x.shape == () and np.issubdtype(x.dtype, np.integer):
            as_int = int(x)
        else:
            return False
        return bool(self.start <= as_int < self.start + self.n)
