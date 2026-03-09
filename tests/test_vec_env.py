import numpy as np
import pytest

from rl_model.vec_env import VectorizedStockTradingEnv


def _make_data(n_steps=390, n_features=23):
    """Synthetic normalized data and prices."""
    rng = np.random.RandomState(42)
    normalized = rng.randn(n_steps, n_features).astype(np.float32)
    prices = np.linspace(100.0, 105.0, n_steps, dtype=np.float64)
    return normalized, prices


class TestConstructorAndReset:
    def test_obs_shape(self):
        norm, prices = _make_data()
        env = VectorizedStockTradingEnv(4, norm, prices)
        obs = env.reset()
        assert obs.shape == (4, 23)
        assert obs.dtype == np.float32

    def test_obs_is_first_row(self):
        norm, prices = _make_data()
        env = VectorizedStockTradingEnv(4, norm, prices)
        obs = env.reset()
        np.testing.assert_array_equal(obs[0], norm[0])
        np.testing.assert_array_equal(obs[3], norm[0])

    def test_spaces(self):
        norm, prices = _make_data()
        env = VectorizedStockTradingEnv(4, norm, prices)
        assert env.num_envs == 4
        assert env.observation_space.shape == (23,)
        assert env.action_space.n == 5

    def test_double_reset(self):
        norm, prices = _make_data()
        env = VectorizedStockTradingEnv(4, norm, prices)
        obs1 = env.reset()
        obs2 = env.reset()
        np.testing.assert_array_equal(obs1, obs2)


class TestStep:
    def test_step_returns_correct_shapes(self):
        norm, prices = _make_data(n_steps=20)
        env = VectorizedStockTradingEnv(4, norm, prices)
        env.reset()
        obs, rewards, dones, infos = env.step(np.array([0, 1, 2, 3]))
        assert obs.shape == (4, 23)
        assert obs.dtype == np.float32
        assert rewards.shape == (4,)
        assert dones.shape == (4,)
        assert dones.dtype == bool
        assert len(infos) == 4

    def test_hold_no_cost(self):
        """Action 0 (0% stock) when holding nothing = no trade, no cost."""
        norm, prices = _make_data(n_steps=20)
        env = VectorizedStockTradingEnv(1, norm, prices)
        env.reset()
        _, rewards, _, _ = env.step(np.array([0]))
        # No position, price change doesn't matter, portfolio = cash = initial
        assert rewards[0] == pytest.approx(0.0, abs=1e-10)

    def test_buy_increases_shares(self):
        norm, prices = _make_data(n_steps=20)
        env = VectorizedStockTradingEnv(2, norm, prices)
        env.reset()
        env.step(np.array([4, 0]))  # env0: buy 100%, env1: hold 0%
        assert env._shares[0] > 0
        assert env._shares[1] == pytest.approx(0.0, abs=1e-10)

    def test_matches_scalar_env(self):
        """Vectorized env must produce identical results to StockTradingEnv."""
        from rl_model.env import StockTradingEnv

        norm, prices = _make_data(n_steps=50)
        actions = [4, 4, 2, 0, 3, 4, 1, 0, 2, 3]

        # Scalar env
        scalar = StockTradingEnv.from_arrays(norm, prices)
        scalar.reset()
        scalar_rewards = []
        for a in actions:
            _, r, _, _, _ = scalar.step(a)
            scalar_rewards.append(r)

        # Vectorized env (1 env to compare)
        vec = VectorizedStockTradingEnv(1, norm, prices)
        vec.reset()
        vec_rewards = []
        for a in actions:
            _, r, _, _ = vec.step(np.array([a]))
            vec_rewards.append(r[0])

        np.testing.assert_allclose(vec_rewards, scalar_rewards, atol=1e-10)


class TestAutoReset:
    def test_terminal_observation_in_info(self):
        norm, prices = _make_data(n_steps=5)
        env = VectorizedStockTradingEnv(1, norm, prices)
        env.reset()
        for _ in range(4):
            obs, _, dones, infos = env.step(np.array([2]))
            assert not dones[0]
        obs, _, dones, infos = env.step(np.array([2]))
        assert dones[0]
        assert "terminal_observation" in infos[0]
        # Returned obs should be the reset obs (step 0), not terminal
        np.testing.assert_array_equal(obs[0], norm[0])

    def test_continues_after_auto_reset(self):
        norm, prices = _make_data(n_steps=5)
        env = VectorizedStockTradingEnv(1, norm, prices)
        env.reset()
        # Run through 2 full episodes
        done_count = 0
        for _ in range(15):
            obs, _, dones, _ = env.step(np.array([2]))
            if dones[0]:
                done_count += 1
                # After reset, obs should be step 0
                np.testing.assert_array_equal(obs[0], norm[0])
        assert done_count >= 2

    def test_all_envs_terminate_simultaneously(self):
        norm, prices = _make_data(n_steps=5)
        env = VectorizedStockTradingEnv(4, norm, prices)
        env.reset()
        for _ in range(4):
            _, _, dones, _ = env.step(np.zeros(4, dtype=int))
            assert not np.any(dones)
        _, _, dones, infos = env.step(np.zeros(4, dtype=int))
        assert np.all(dones)
        for info in infos:
            assert "terminal_observation" in info

    def test_staggered_resets(self):
        """Envs that reset at different times don't interfere."""
        norm, prices = _make_data(n_steps=5)
        env = VectorizedStockTradingEnv(2, norm, prices)
        env.reset()

        # Advance env 0 by 1 extra step by manually setting its step counter
        env._current_steps[0] = 1

        # env 0 is at step 1, env 1 is at step 0. Run until env 0 terminates.
        for _ in range(3):
            env.step(np.array([2, 2]))

        # env 0 should be at step 4 (last), env 1 at step 3
        assert env._current_steps[0] == 4
        assert env._current_steps[1] == 3

        obs, _, dones, _ = env.step(np.array([2, 2]))
        assert dones[0] and not dones[1]
        # env 0 auto-reset, env 1 still going
        assert env._current_steps[0] == 0
        assert env._current_steps[1] == 4


class TestPipelineIntegration:
    def test_ppo_learns_with_vectorized_env(self):
        """PPO can train on VectorizedStockTradingEnv without errors."""
        from stable_baselines3 import PPO

        norm, prices = _make_data(n_steps=50)
        env = VectorizedStockTradingEnv(4, norm, prices)
        model = PPO(
            "MlpPolicy", env, device="cpu",
            n_steps=32, batch_size=16, n_epochs=2, verbose=0,
        )
        model.learn(total_timesteps=128)
        # Verify model can predict
        obs = env.reset()
        actions, _ = model.predict(obs, deterministic=True)
        assert actions.shape == (4,)
