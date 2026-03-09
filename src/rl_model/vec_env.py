from __future__ import annotations

import gymnasium
import numpy as np
import jax
import jax.numpy as jnp
from stable_baselines3.common.vec_env import VecEnv
from rl_model.jax_env import JAXStockTradingEnv, EnvState, EnvParams


class VectorizedStockTradingEnv(VecEnv):
    """Vectorized stock trading env — all N envs step via numpy broadcasting.

    Replaces SubprocVecEnv to eliminate IPC overhead. All envs share the same
    day's data arrays (read-only). Per-env state is stored as (n_envs,) arrays.
    """

    def __init__(
        self,
        n_envs: int,
        normalized_data: np.ndarray,
        raw_close_prices: np.ndarray,
        initial_cash: float = 100_000.0,
        transaction_cost_pct: float = 0.001,
    ):
        observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(normalized_data.shape[1],),
            dtype=np.float32,
        )
        action_space = gymnasium.spaces.Discrete(5)
        super().__init__(n_envs, observation_space, action_space)

        self._normalized_data = normalized_data
        self._raw_close_prices = raw_close_prices
        self._num_steps = len(normalized_data)
        self._initial_cash = initial_cash
        self._transaction_cost_pct = transaction_cost_pct
        self._action_to_position = np.array([0.00, 0.25, 0.50, 0.75, 1.00])

        # Per-env state arrays
        self._current_steps = np.zeros(n_envs, dtype=np.int64)
        self._cash = np.full(n_envs, initial_cash, dtype=np.float64)
        self._shares = np.zeros(n_envs, dtype=np.float64)
        self._portfolio_values = np.full(n_envs, initial_cash, dtype=np.float64)

    def reset(self):
        self._current_steps[:] = 0
        self._cash[:] = self._initial_cash
        self._shares[:] = 0.0
        self._portfolio_values[:] = self._initial_cash
        obs = np.tile(self._normalized_data[0], (self.num_envs, 1))
        return obs

    def step_async(self, actions):
        self._actions = np.asarray(actions)

    def step_wait(self):
        actions = self._actions
        n = self.num_envs

        # Vectorized price lookup
        current_prices = self._raw_close_prices[self._current_steps]
        current_portfolio_values = self._cash + self._shares * current_prices

        # Vectorized rebalance
        target_pcts = self._action_to_position[actions]
        target_stock_values = target_pcts * current_portfolio_values
        target_shares = target_stock_values / current_prices
        shares_deltas = target_shares - self._shares

        # Transaction costs
        trade_values = np.abs(shares_deltas) * current_prices
        costs = trade_values * self._transaction_cost_pct

        self._shares[:] = target_shares
        self._cash[:] = current_portfolio_values - target_stock_values - costs

        # Advance
        self._current_steps += 1
        terminated = self._current_steps >= self._num_steps

        # New prices (clamp for terminated envs)
        safe_steps = np.minimum(self._current_steps, self._num_steps - 1)
        new_prices = self._raw_close_prices[safe_steps]

        new_portfolio_values = self._cash + self._shares * new_prices
        rewards = (new_portfolio_values - self._portfolio_values).astype(np.float32)
        self._portfolio_values[:] = new_portfolio_values

        # Observations (fancy indexing already returns a copy)
        obs = self._normalized_data[safe_steps]

        # Build infos and auto-reset terminated envs
        term_idx = np.flatnonzero(terminated)
        infos: list[dict] = [{} for _ in range(n)]
        if len(term_idx) > 0:
            terminal_obs = obs[term_idx].copy()  # snapshot before reset
            self._current_steps[term_idx] = 0
            self._cash[term_idx] = self._initial_cash
            self._shares[term_idx] = 0.0
            self._portfolio_values[term_idx] = self._initial_cash
            obs[term_idx] = self._normalized_data[0]
            for idx_pos, i in enumerate(term_idx):
                infos[i] = {"terminal_observation": terminal_obs[idx_pos], "TimeLimit.truncated": False}

        # >= comparison already created a fresh array
        dones = terminated
        return obs, rewards, dones, infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        if indices is None:
            indices = range(self.num_envs)
        return [getattr(self, attr_name)] * len(indices)

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *args, indices=None, **kwargs):
        if indices is None:
            indices = range(self.num_envs)
        return [None] * len(indices)

    def env_is_wrapped(self, wrapper_class, indices=None):
        if indices is None:
            indices = range(self.num_envs)
        return [False] * len(indices)


class FastVectorizedStockTradingEnv(VecEnv):
    """JAX-accelerated version of VectorizedStockTradingEnv.
    
    Uses JAX for the entire step function, allowing it to run entirely on GPU
    with unified memory (GB10). This eliminates the Numpy-to-JAX conversion
    overhead during each step of the RL rollout.
    """

    def __init__(
        self,
        n_envs: int,
        normalized_data: np.ndarray,
        raw_close_prices: np.ndarray,
        initial_cash: float = 100_000.0,
        transaction_cost_pct: float = 0.001,
    ):
        observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(normalized_data.shape[1],),
            dtype=np.float32,
        )
        action_space = gymnasium.spaces.Discrete(5)
        super().__init__(n_envs, observation_space, action_space)

        # Use jax.device_put for unified memory optimization on GB10
        self._normalized_data = jax.device_put(jnp.array(normalized_data, dtype=jnp.float32))
        self._raw_close_prices = jax.device_put(jnp.array(raw_close_prices, dtype=jnp.float32))
        
        self.jax_env = JAXStockTradingEnv(self._normalized_data, self._raw_close_prices)
        self.params = EnvParams(initial_cash=initial_cash, transaction_cost_pct=transaction_cost_pct)
        
        # JIT compile the vectorized functions
        self._vmap_reset = jax.jit(jax.vmap(self.jax_env.reset, in_axes=(None,)))
        self._vmap_step = jax.jit(jax.vmap(self.jax_env.step, in_axes=(0, 0, None)))

        # Per-env state
        self._state = self._vmap_reset(self.params)
        self._obs, self._state = self._state

    def reset(self):
        self._obs, self._state = self._vmap_reset(self.params)
        return np.array(self._obs)

    def step_async(self, actions):
        self._actions = jnp.array(actions, dtype=jnp.int32)

    def step_wait(self):
        # The core logic runs on GPU
        obs, state, rewards, dones, infos = self._vmap_step(self._state, self._actions, self.params)
        
        # Auto-reset handled here in a JAX-friendly way
        # If done, we need to reset that environment's state and get initial obs
        if jnp.any(dones):
            init_obs, init_state = self._vmap_reset(self.params)
            
            # Use where to selectively reset states
            def select(reset_val, current_val):
                return jnp.where(dones[:, None] if reset_val.ndim > 1 else dones, reset_val, current_val)
            
            # Special case for NamedTuple members
            new_state_dict = {}
            for field in state._fields:
                new_state_dict[field] = select(getattr(init_state, field), getattr(state, field))
            state = EnvState(**new_state_dict)
            
            obs = select(init_obs, obs)

        self._obs = obs
        self._state = state
        
        # Convert back to Numpy only at the boundary for SB3/SBX compatibility
        # (Though SBX could potentially handle JAX arrays directly)
        return np.array(obs), np.array(rewards), np.array(dones), [{}] * self.num_envs

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [None] * (len(indices) if indices else self.num_envs)

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [None] * (len(indices) if indices else self.num_envs)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * (len(indices) if indices else self.num_envs)
