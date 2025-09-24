"""Tests for American Trend Following System."""

import pytest
from pydantic import ValidationError
from unittest import mock

from trendfollower.core.american_tf import (
    AmericanTFState,
    AmericanTrendFollower,
    MarketData,
    StrategyParams,
    handle_long_position,
    handle_neutral_position,
    handle_short_position,
)


class TestMarketData:
    """Test cases for MarketData model."""

    def test_valid_market_data(self):
        """Test creation of valid MarketData."""
        market = MarketData(price=100.0, fast_ema=102.0, slow_ema=98.0, atr=2.0)
        assert market.price == 100.0
        assert market.fast_ema == 102.0
        assert market.slow_ema == 98.0
        assert market.atr == 2.0

    def test_market_data_immutable(self):
        """Test that MarketData is immutable."""
        market = MarketData(price=100.0, fast_ema=102.0, slow_ema=98.0, atr=2.0)
        with pytest.raises(ValidationError):
            market.price = 150.0

    @pytest.mark.parametrize("field,value", [
        ("price", 0.0),
        ("price", -10.0),
        ("fast_ema", 0.0),
        ("fast_ema", -5.0),
        ("slow_ema", 0.0),
        ("slow_ema", -3.0),
        ("atr", 0.0),
        ("atr", -1.0),
    ])
    def test_market_data_validation_errors(self, field, value):
        """Test MarketData validation for non-positive values."""
        data = {"price": 100.0, "fast_ema": 102.0, "slow_ema": 98.0, "atr": 2.0}
        data[field] = value
        with pytest.raises(ValidationError):
            MarketData(**data)


class TestStrategyParams:
    """Test cases for StrategyParams model."""

    def test_valid_strategy_params(self):
        """Test creation of valid StrategyParams."""
        params = StrategyParams(entry_point_width=0.5, stop_loss_width=2.0, risk_multiple=0.02)
        assert params.entry_point_width == 0.5
        assert params.stop_loss_width == 2.0
        assert params.risk_multiple == 0.02

    def test_strategy_params_immutable(self):
        """Test that StrategyParams is immutable."""
        params = StrategyParams(entry_point_width=0.5, stop_loss_width=2.0, risk_multiple=0.02)
        with pytest.raises(ValidationError):
            params.entry_point_width = 1.0

    @pytest.mark.parametrize("field,value", [
        ("entry_point_width", 0.0),
        ("entry_point_width", -0.1),
        ("stop_loss_width", 0.0),
        ("stop_loss_width", -1.0),
        ("risk_multiple", 0.0),
        ("risk_multiple", -0.01),
    ])
    def test_strategy_params_validation_errors(self, field, value):
        """Test StrategyParams validation for non-positive values."""
        data = {"entry_point_width": 0.5, "stop_loss_width": 2.0, "risk_multiple": 0.02}
        data[field] = value
        with pytest.raises(ValidationError):
            StrategyParams(**data)


class TestAmericanTFState:
    """Test cases for AmericanTFState model."""

    def test_valid_neutral_state(self):
        """Test creation of valid neutral state."""
        state = AmericanTFState()
        assert state.position == 0
        assert state.position_size == 0
        assert state.stop_loss is None
        assert state.entry_price is None
        assert state.close_next_day is False

    def test_valid_long_state(self):
        """Test creation of valid long state."""
        state = AmericanTFState(
            position=1, position_size=10.0, stop_loss=95.0, entry_price=100.0
        )
        assert state.position == 1
        assert state.position_size == 10.0
        assert state.stop_loss == 95.0
        assert state.entry_price == 100.0

    def test_valid_short_state(self):
        """Test creation of valid short state."""
        state = AmericanTFState(
            position=-1, position_size=-10.0, stop_loss=105.0, entry_price=100.0
        )
        assert state.position == -1
        assert state.position_size == -10.0
        assert state.stop_loss == 105.0
        assert state.entry_price == 100.0

    def test_state_immutable(self):
        """Test that AmericanTFState is immutable."""
        state = AmericanTFState()
        with pytest.raises(ValidationError):
            state.position = 1

    @pytest.mark.parametrize("position", [-2, 2, 5])
    def test_invalid_position_values(self, position):
        """Test validation error for invalid position values."""
        with pytest.raises(ValidationError):
            AmericanTFState(position=position)

    def test_invalid_neutral_position_with_position_size(self):
        """Test validation error for neutral position with non-zero position_size."""
        with pytest.raises(ValidationError):
            AmericanTFState(position=0, position_size=10.0)

    def test_invalid_neutral_position_with_stop_loss(self):
        """Test validation error for neutral position with stop_loss."""
        with pytest.raises(ValidationError):
            AmericanTFState(position=0, stop_loss=95.0)

    def test_invalid_neutral_position_with_entry_price(self):
        """Test validation error for neutral position with entry_price."""
        with pytest.raises(ValidationError):
            AmericanTFState(position=0, entry_price=100.0)

    def test_invalid_long_position_negative_size(self):
        """Test validation error for long position with negative position_size."""
        with pytest.raises(ValidationError):
            AmericanTFState(position=1, position_size=-10.0, stop_loss=95.0, entry_price=100.0)

    def test_invalid_long_position_no_stop_loss(self):
        """Test validation error for long position without stop_loss."""
        with pytest.raises(ValidationError):
            AmericanTFState(position=1, position_size=10.0, entry_price=100.0)

    def test_invalid_long_position_no_entry_price(self):
        """Test validation error for long position without entry_price."""
        with pytest.raises(ValidationError):
            AmericanTFState(position=1, position_size=10.0, stop_loss=95.0)

    def test_invalid_short_position_positive_size(self):
        """Test validation error for short position with positive position_size."""
        with pytest.raises(ValidationError):
            AmericanTFState(position=-1, position_size=10.0, stop_loss=105.0, entry_price=100.0)

    def test_invalid_short_position_no_stop_loss(self):
        """Test validation error for short position without stop_loss."""
        with pytest.raises(ValidationError):
            AmericanTFState(position=-1, position_size=-10.0, entry_price=100.0)

    def test_invalid_short_position_no_entry_price(self):
        """Test validation error for short position without entry_price."""
        with pytest.raises(ValidationError):
            AmericanTFState(position=-1, position_size=-10.0, stop_loss=105.0)

    def test_negative_stop_loss(self):
        """Test validation error for negative stop_loss."""
        with pytest.raises(ValidationError):
            AmericanTFState(position=1, position_size=10.0, stop_loss=-5.0, entry_price=100.0)

    def test_negative_entry_price(self):
        """Test validation error for negative entry_price."""
        with pytest.raises(ValidationError):
            AmericanTFState(position=1, position_size=10.0, stop_loss=95.0, entry_price=-100.0)


class TestHandleNeutralPosition:
    """Test cases for handle_neutral_position function."""

    def test_long_signal_entry(self):
        """Test long entry when fast EMA > slow EMA + signal buffer."""
        market = MarketData(price=100.0, fast_ema=105.0, slow_ema=100.0, atr=2.0)
        params = StrategyParams(entry_point_width=2.0, stop_loss_width=3.0, risk_multiple=0.02)
        
        result = handle_neutral_position(market, params)
        
        assert result.position == 1
        assert result.position_size == 1.0  # 0.02 * 100 / 2
        assert result.stop_loss == 94.0  # 100 - 3 * 2
        assert result.entry_price == 100.0
        assert result.close_next_day is False

    def test_short_signal_entry(self):
        """Test short entry when fast EMA < slow EMA - signal buffer."""
        market = MarketData(price=100.0, fast_ema=95.0, slow_ema=100.0, atr=2.0)
        params = StrategyParams(entry_point_width=2.0, stop_loss_width=3.0, risk_multiple=0.02)
        
        result = handle_neutral_position(market, params)
        
        assert result.position == -1
        assert result.position_size == -1.0  # -0.02 * 100 / 2
        assert result.stop_loss == 106.0  # 100 + 3 * 2
        assert result.entry_price == 100.0
        assert result.close_next_day is False

    def test_no_signal_remain_neutral(self):
        """Test remaining neutral when no clear signal."""
        market = MarketData(price=100.0, fast_ema=101.0, slow_ema=100.0, atr=2.0)
        params = StrategyParams(entry_point_width=2.0, stop_loss_width=3.0, risk_multiple=0.02)
        
        result = handle_neutral_position(market, params)
        
        assert result.position == 0
        assert result.position_size == 0.0
        assert result.stop_loss is None
        assert result.entry_price is None
        assert result.close_next_day is False

    def test_boundary_condition_long(self):
        """Test boundary condition for long signal (exactly at threshold)."""
        market = MarketData(price=100.0, fast_ema=104.0, slow_ema=100.0, atr=2.0)
        params = StrategyParams(entry_point_width=2.0, stop_loss_width=3.0, risk_multiple=0.02)
        
        result = handle_neutral_position(market, params)
        
        # fast_ema (104) == slow_ema (100) + signal_buffer (4), so no entry
        assert result.position == 0

    def test_boundary_condition_short(self):
        """Test boundary condition for short signal (exactly at threshold)."""
        market = MarketData(price=100.0, fast_ema=96.0, slow_ema=100.0, atr=2.0)
        params = StrategyParams(entry_point_width=2.0, stop_loss_width=3.0, risk_multiple=0.02)
        
        result = handle_neutral_position(market, params)
        
        # fast_ema (96) == slow_ema (100) - signal_buffer (4), so no entry
        assert result.position == 0


class TestHandleLongPosition:
    """Test cases for handle_long_position function."""

    @pytest.fixture
    def long_state(self):
        """Fixture for a typical long position state."""
        return AmericanTFState(
            position=1, position_size=1.0, stop_loss=95.0, entry_price=100.0
        )

    @pytest.fixture
    def market_data(self):
        """Fixture for typical market data."""
        return MarketData(price=98.0, fast_ema=105.0, slow_ema=100.0, atr=2.0)

    @pytest.fixture
    def strategy_params(self):
        """Fixture for typical strategy parameters."""
        return StrategyParams(entry_point_width=2.0, stop_loss_width=3.0, risk_multiple=0.02)

    def test_close_next_day_flag(self, market_data, strategy_params):
        """Test that position is closed when close_next_day flag is set."""
        state = AmericanTFState(
            position=1, position_size=1.0, stop_loss=95.0, entry_price=100.0, close_next_day=True
        )
        
        result = handle_long_position(market_data, strategy_params, state)
        
        assert result.position == 0
        assert result.position_size == 0.0
        assert result.stop_loss is None
        assert result.entry_price is None
        assert result.close_next_day is False

    def test_stop_loss_breach_signal_off(self, strategy_params, long_state):
        """Test stop loss breach when signal is off (immediate close)."""
        market = MarketData(price=94.0, fast_ema=100.0, slow_ema=100.0, atr=2.0)  # Signal off
        
        result = handle_long_position(market, strategy_params, long_state)
        
        assert result.position == 0
        assert result.position_size == 0.0
        assert result.stop_loss is None
        assert result.entry_price is None
        assert result.close_next_day is False

    def test_stop_loss_breach_signal_on(self, strategy_params, long_state):
        """Test stop loss breach when signal is still on (mark for closure)."""
        market = MarketData(price=94.0, fast_ema=105.0, slow_ema=100.0, atr=2.0)  # Signal on
        
        result = handle_long_position(market, strategy_params, long_state)
        
        assert result.position == 1  # Position unchanged
        assert result.position_size == 1.0  # Position size unchanged
        assert result.stop_loss == 95.0  # Stop loss unchanged
        assert result.entry_price == 100.0  # Entry price unchanged
        assert result.close_next_day is True  # Marked for closure next day

    def test_trailing_stop_loss_update(self, strategy_params, long_state):
        """Test trailing stop loss update when signal is on."""
        market = MarketData(price=102.0, fast_ema=105.0, slow_ema=100.0, atr=2.0)
        
        result = handle_long_position(market, strategy_params, long_state)
        
        expected_new_stop = max(95.0, 102.0 - 3.0 * 2.0)  # max(95, 96) = 96
        assert result.position == 1
        assert result.position_size == 1.0
        assert result.stop_loss == 96.0
        assert result.entry_price == 100.0
        assert result.close_next_day is False

    def test_stop_loss_no_update_when_lower(self, strategy_params, long_state):
        """Test that stop loss doesn't move down."""
        market = MarketData(price=97.0, fast_ema=105.0, slow_ema=100.0, atr=2.0)
        
        result = handle_long_position(market, strategy_params, long_state)
        
        expected_new_stop = max(95.0, 97.0 - 3.0 * 2.0)  # max(95, 91) = 95
        assert result.stop_loss == 95.0  # Unchanged

    def test_invalid_state_no_stop_loss(self, market_data, strategy_params):
        """Test error when long position has no stop loss."""
        # We can't create an invalid state due to Pydantic validation,
        # but we can test the handler's explicit check by directly creating a mock

        # Create a mock state that mimics invalid data
        mock_state = mock.Mock(spec=AmericanTFState)
        mock_state.position = 1
        mock_state.position_size = 1.0
        mock_state.stop_loss = None  
        mock_state.entry_price = 100.0

        with pytest.raises(ValueError, match="stop_loss must be set for open positions"):
            handle_long_position(market_data, strategy_params, mock_state)


class TestHandleShortPosition:
    """Test cases for handle_short_position function."""

    @pytest.fixture
    def short_state(self):
        """Fixture for a typical short position state."""
        return AmericanTFState(
            position=-1, position_size=-1.0, stop_loss=105.0, entry_price=100.0
        )

    @pytest.fixture
    def market_data(self):
        """Fixture for typical market data."""
        return MarketData(price=102.0, fast_ema=95.0, slow_ema=100.0, atr=2.0)

    @pytest.fixture
    def strategy_params(self):
        """Fixture for typical strategy parameters."""
        return StrategyParams(entry_point_width=2.0, stop_loss_width=3.0, risk_multiple=0.02)

    def test_close_next_day_flag(self, market_data, strategy_params):
        """Test that position is closed when close_next_day flag is set."""
        state = AmericanTFState(
            position=-1, position_size=-1.0, stop_loss=105.0, entry_price=100.0, close_next_day=True
        )
        
        result = handle_short_position(market_data, strategy_params, state)
        
        assert result.position == 0
        assert result.position_size == 0
        assert result.stop_loss is None
        assert result.entry_price is None
        assert result.close_next_day is False

    def test_stop_loss_breach_signal_off(self, strategy_params, short_state):
        """Test stop loss breach when signal is off (immediate close)."""
        market = MarketData(price=106.0, fast_ema=100.0, slow_ema=100.0, atr=2.0)  # Signal off
        
        result = handle_short_position(market, strategy_params, short_state)
        
        assert result.position == 0
        assert result.position_size == 0.0
        assert result.stop_loss is None
        assert result.entry_price is None
        assert result.close_next_day is False

    def test_stop_loss_breach_signal_on(self, strategy_params, short_state):
        """Test stop loss breach when signal is still on (mark for closure)."""
        market = MarketData(price=106.0, fast_ema=95.0, slow_ema=100.0, atr=2.0)  # Signal on
        
        result = handle_short_position(market, strategy_params, short_state)
        
        assert result.position == -1  # Position unchanged
        assert result.position_size == -1.0  # Position size unchanged
        assert result.stop_loss == 105.0  # Stop loss unchanged
        assert result.entry_price == 100.0  # Entry price unchanged
        assert result.close_next_day is True  # Marked for closure

    def test_trailing_stop_loss_update(self, strategy_params, short_state):
        """Test trailing stop loss update when price moves favorably."""
        market = MarketData(price=98.0, fast_ema=95.0, slow_ema=100.0, atr=2.0)
        
        result = handle_short_position(market, strategy_params, short_state)
        
        expected_new_stop = min(105.0, 98.0 + 3.0 * 2.0)  # min(105, 104) = 104
        assert result.position == -1
        assert result.position_size == -1.0
        assert result.stop_loss == 104.0
        assert result.entry_price == 100.0
        assert result.close_next_day is False

    def test_stop_loss_no_update_when_higher(self, strategy_params, short_state):
        """Test that stop loss doesn't move up."""
        market = MarketData(price=103.0, fast_ema=95.0, slow_ema=100.0, atr=2.0)
        
        result = handle_short_position(market, strategy_params, short_state)
        
        expected_new_stop = min(105.0, 103.0 + 3.0 * 2.0)  # min(105, 109) = 105
        assert result.stop_loss == 105.0  # Unchanged

    def test_invalid_state_no_stop_loss(self, market_data, strategy_params):
        """Test error when short position has no stop loss."""
        # Create a mock state that mimics invalid data
        mock_state = mock.Mock(spec=AmericanTFState)
        mock_state.position = -1
        mock_state.position_size = -1.0
        mock_state.stop_loss = None  # This is what we want to test
        mock_state.entry_price = 100.0

        with pytest.raises(ValueError, match="stop_loss must be set for open positions"):
            handle_short_position(market_data, strategy_params, mock_state)


class TestAmericanTrendFollower:
    """Test cases for AmericanTrendFollower class."""

    @pytest.fixture
    def strategy(self):
        """Fixture for AmericanTrendFollower instance."""
        return AmericanTrendFollower(
            entry_point_width=2.0, stop_loss_width=3.0, risk_multiple=0.02
        )

    def test_initialization(self, strategy):
        """Test proper initialization of AmericanTrendFollower."""
        assert strategy.params.entry_point_width == 2.0
        assert strategy.params.stop_loss_width == 3.0
        assert strategy.params.risk_multiple == 0.02
        assert len(strategy.handlers) == 3
        assert 0 in strategy.handlers
        assert 1 in strategy.handlers
        assert -1 in strategy.handlers

    def test_initialization_validation_errors(self):
        """Test validation errors during initialization."""
        with pytest.raises(ValidationError):
            AmericanTrendFollower(entry_point_width=0.0, stop_loss_width=3.0, risk_multiple=0.02)
        
        with pytest.raises(ValidationError):
            AmericanTrendFollower(entry_point_width=2.0, stop_loss_width=0.0, risk_multiple=0.02)
        
        with pytest.raises(ValidationError):
            AmericanTrendFollower(entry_point_width=2.0, stop_loss_width=3.0, risk_multiple=0.0)

    def test_update_state_from_neutral_to_long(self, strategy):
        """Test state update from neutral to long position."""
        initial_state = AmericanTFState()
        
        result = strategy.update_state(
            s=100.0, s_fast=105.0, s_slow=100.0, atr=2.0, state=initial_state
        )
        
        assert result.position == 1
        assert result.position_size == 1.0
        assert result.stop_loss == 94.0
        assert result.entry_price == 100.0

    def test_update_state_from_neutral_to_short(self, strategy):
        """Test state update from neutral to short position."""
        initial_state = AmericanTFState()
        
        result = strategy.update_state(
            s=100.0, s_fast=95.0, s_slow=100.0, atr=2.0, state=initial_state
        )
        
        assert result.position == -1
        assert result.position_size == -1.0
        assert result.stop_loss == 106.0
        assert result.entry_price == 100.0

    def test_update_state_remain_neutral(self, strategy):
        """Test state update remaining neutral."""
        initial_state = AmericanTFState()
        
        result = strategy.update_state(
            s=100.0, s_fast=101.0, s_slow=100.0, atr=2.0, state=initial_state
        )
        
        assert result.position == 0
        assert result.position_size == 0.0
        assert result.stop_loss is None
        assert result.entry_price is None

    def test_update_state_long_position_trailing_stop(self, strategy):
        """Test state update for long position with trailing stop."""
        initial_state = AmericanTFState(
            position=1, position_size=1.0, stop_loss=95.0, entry_price=100.0
        )
        
        result = strategy.update_state(
            s=102.0, s_fast=105.0, s_slow=100.0, atr=2.0, state=initial_state
        )
        
        assert result.position == 1
        assert result.stop_loss == 96.0  # Updated trailing stop

    def test_update_state_short_position_trailing_stop(self, strategy):
        """Test state update for short position with trailing stop."""
        initial_state = AmericanTFState(
            position=-1, position_size=-1.0, stop_loss=105.0, entry_price=100.0
        )
        
        result = strategy.update_state(
            s=98.0, s_fast=95.0, s_slow=100.0, atr=2.0, state=initial_state
        )
        
        assert result.position == -1
        assert result.stop_loss == 104.0  # Updated trailing stop

    def test_invalid_market_data_validation(self, strategy):
        """Test that invalid market data raises validation error."""
        initial_state = AmericanTFState()
        
        with pytest.raises(ValidationError):
            strategy.update_state(
                s=0.0, s_fast=105.0, s_slow=100.0, atr=2.0, state=initial_state
            )
        
        with pytest.raises(ValidationError):
            strategy.update_state(
                s=100.0, s_fast=0.0, s_slow=100.0, atr=2.0, state=initial_state
            )
        
        with pytest.raises(ValidationError):
            strategy.update_state(
                s=100.0, s_fast=105.0, s_slow=0.0, atr=2.0, state=initial_state
            )
        
        with pytest.raises(ValidationError):
            strategy.update_state(
                s=100.0, s_fast=105.0, s_slow=100.0, atr=0.0, state=initial_state
            )


class TestEndtoEndScenarios:
    """Test end-to-end scenarios."""

    @pytest.fixture
    def strategy(self):
        """Fixture for AmericanTrendFollower instance."""
        return AmericanTrendFollower(
            entry_point_width=1.0, stop_loss_width=2.0, risk_multiple=0.01
        )

    def test_complete_trading_cycle_long(self, strategy):
        """Test a complete long trading cycle: entry -> trail -> exit."""
        # Start neutral
        state = AmericanTFState()
        
        # 1. Enter long position
        state = strategy.update_state(
            s=100.0, s_fast=102.0, s_slow=100.0, atr=1.0, state=state
        )
        assert state.position == 1
        assert state.stop_loss == 98.0  # 100 - 2*1
        
        # 2. Price moves up, trail stop loss
        state = strategy.update_state(
            s=105.0, s_fast=107.0, s_slow=100.0, atr=1.0, state=state
        )
        assert state.position == 1
        assert state.stop_loss == 103.0  # max(98, 105-2) = 103
        
        # 3. Stop loss breach with signal off - exit
        state = strategy.update_state(
            s=102.0, s_fast=100.0, s_slow=100.0, atr=1.0, state=state
        )
        assert state.position == 0
        assert state.stop_loss is None

    def test_complete_trading_cycle_short(self, strategy):
        """Test a complete short trading cycle: entry -> trail -> exit."""
        # Start neutral
        state = AmericanTFState()
        
        # 1. Enter short position
        state = strategy.update_state(
            s=100.0, s_fast=98.0, s_slow=100.0, atr=1.0, state=state
        )
        assert state.position == -1
        assert state.stop_loss == 102.0  # 100 + 2*1
        
        # 2. Price moves down, trail stop loss
        state = strategy.update_state(
            s=95.0, s_fast=93.0, s_slow=100.0, atr=1.0, state=state
        )
        assert state.position == -1
        assert state.stop_loss == 97.0  # min(102, 95+2) = 97
        
        # 3. Stop loss breach with signal off - exit
        state = strategy.update_state(
            s=98.0, s_fast=100.0, s_slow=100.0, atr=1.0, state=state
        )
        assert state.position == 0
        assert state.stop_loss is None

    def test_stop_loss_breach_with_signal_persistence(self, strategy):
        """Test stop loss breach when signal persists (close_next_day scenario)."""
        # Start with long position
        state = AmericanTFState(
            position=1, position_size=1.0, stop_loss=95.0, entry_price=100.0
        )
        
        # Stop loss breach but signal still on
        state = strategy.update_state(
            s=94.0, s_fast=105.0, s_slow=100.0, atr=1.0, state=state
        )
        assert state.position == 1  # Still long
        assert state.close_next_day is True  # Marked for closure
        
        # Next day - position should be closed
        state = strategy.update_state(
            s=94.0, s_fast=105.0, s_slow=100.0, atr=1.0, state=state
        )
        assert state.position == 0  # Now closed
        assert state.close_next_day is False

    def test_very_small_atr_edge_case(self, strategy):
        """Test behavior with very small ATR values."""
        state = AmericanTFState()
        
        # Very small ATR
        result = strategy.update_state(
            s=100.0, s_fast=102.0, s_slow=100.0, atr=0.01, state=state
        )
        
        assert result.position == 1
        assert result.position_size == 100.0  # 0.01 * 100 / 0.01
        assert result.stop_loss == 99.98  # 100 - 2 * 0.01

    def test_high_volatility_edge_case(self, strategy):
        """Test behavior with high ATR values."""
        state = AmericanTFState()
        result = strategy.update_state(
            s=100.0, s_fast=111.0, s_slow=100.0, atr=10.0, state=state
        )
        
        assert result.position == 1
        assert result.position_size == 0.1  # 0.01 * 100 / 10
        assert result.stop_loss == 80.0  # 100 - 2 * 10
