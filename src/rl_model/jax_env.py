import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple

class EnvState(NamedTuple):
    current_step: jnp.ndarray
    cash: jnp.ndarray
    shares: jnp.ndarray
    portfolio_value: jnp.ndarray
    done: jnp.ndarray

class EnvParams(NamedTuple):
    initial_cash: float = 100000.0
    transaction_cost_pct: float = 0.001

class JAXStockTradingEnv:
    """Pure JAX implementation of the stock trading environment.
    
    This allows for end-to-end JIT compilation when used with JAX-based RL agents (SBX).
    Designed to be used with jax.vmap for vectorization.
    """

    def __init__(self, normalized_data: jnp.ndarray, raw_close_prices: jnp.ndarray):
        self.normalized_data = normalized_data
        self.raw_close_prices = raw_close_prices
        self.num_steps = len(normalized_data)
        self.action_to_position = jnp.array([0.00, 0.25, 0.50, 0.75, 1.00])

    def reset(self, params: EnvParams) -> Tuple[jnp.ndarray, EnvState]:
        state = EnvState(
            current_step=jnp.array(0, dtype=jnp.int32),
            cash=jnp.array(params.initial_cash, dtype=jnp.float32),
            shares=jnp.array(0.0, dtype=jnp.float32),
            portfolio_value=jnp.array(params.initial_cash, dtype=jnp.float32),
            done=jnp.array(False, dtype=jnp.bool_)
        )
        obs = self.normalized_data[0]
        return obs, state

    def step(self, state: EnvState, action: int, params: EnvParams) -> Tuple[jnp.ndarray, EnvState, float, bool, dict]:
        current_price = self.raw_close_prices[state.current_step]
        current_portfolio_value = state.cash + state.shares * current_price

        # Rebalance
        target_pct = self.action_to_position[action]
        target_stock_value = target_pct * current_portfolio_value
        target_shares = target_stock_value / current_price
        shares_delta = target_shares - state.shares

        # Transaction costs
        trade_value = jnp.abs(shares_delta) * current_price
        cost = trade_value * params.transaction_cost_pct

        new_shares = target_shares
        new_cash = current_portfolio_value - target_stock_value - cost

        # Advance step
        next_step = state.current_step + 1
        done = next_step >= self.num_steps
        
        # Calculate new portfolio value with the price at the NEXT step (if not done)
        # In this simplified model, we use the price at current_step + 1 for the reward calculation
        # following the logic of the original environment.
        price_at_next_step = jnp.where(done, current_price, self.raw_close_prices[jnp.minimum(next_step, self.num_steps - 1)])
        
        new_portfolio_value = new_cash + new_shares * price_at_next_step
        reward = new_portfolio_value - state.portfolio_value
        
        new_state = EnvState(
            current_step=next_step,
            cash=new_cash,
            shares=new_shares,
            portfolio_value=new_portfolio_value,
            done=done
        )
        
        obs = self.normalized_data[jnp.minimum(next_step, self.num_steps - 1)]
        
        info = {
            "portfolio_value": new_portfolio_value,
            "step": next_step
        }
        
        return obs, new_state, reward, done, info

def vmap_step(env: JAXStockTradingEnv, states: EnvState, actions: jnp.ndarray, params: EnvParams):
    return jax.vmap(env.step, in_axes=(0, 0, None))(states, actions, params)

def vmap_reset(env: JAXStockTradingEnv, params: EnvParams, num_envs: int):
    return jax.vmap(env.reset, in_axes=(None,))(params) # This needs adjustment for num_envs
